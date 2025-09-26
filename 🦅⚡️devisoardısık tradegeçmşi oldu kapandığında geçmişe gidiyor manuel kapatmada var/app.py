"""
🚀 Ana Uygulama - Kripto AI Trading Sistemi + Real-Time WebSocket
TradingView benzeri gerçek zamanlı indikatör güncellemeleri
🔥 DÜZELTME: Ticker + Kline hibrit stream ile milisaniye seviyesi güncelleme
"""

import threading
import pandas as pd
import websocket
import json
import time
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

# Kendi modüllerimizi import et
import config
from config import (
    initialize, DASH_CONFIG, TABLE_REFRESH_INTERVAL,
    DEFAULT_TIMEFRAME, DEFAULT_MIN_STREAK, DEFAULT_MIN_PCT, 
    DEFAULT_MIN_VOLR, DEFAULT_MIN_AI_SCORE, LOCAL_TZ,
    MAX_OPEN_POSITIONS, BASE
)
from ui.components import create_layout, create_ai_score_bar
from trading.analyzer import analyze_symbol_with_ai
from core.indicators import compute_consecutive_metrics
from core.ai_model import ai_model
from data.fetch_data import fetch_klines, get_current_price
from data.database import (
    setup_csv_files, load_trades_from_csv, 
    calculate_performance_metrics
)

# 🔥 LIVE TRADING IMPORT
try:
    from trading.live_trader import (
        start_live_trading, stop_live_trading, is_live_trading_active,
        get_live_trading_status, get_live_bot_status_for_symbol, get_auto_sltp_count
    )
    LIVE_TRADER_AVAILABLE = True
except ImportError:
    LIVE_TRADER_AVAILABLE = False
    print("⚠️ Live trader modülü bulunamadı - sadece analiz modu")

# 🔥 BINANCE CLIENT IMPORT
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ python-binance kurulu değil - sembol seçimi manuel yapılacak")

# Sistem başlatma
logger, session = initialize()

# Global değişkenler
auto_scan_active = False
websocket_manager = None
current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_streak': DEFAULT_MIN_STREAK,
    'min_pct': DEFAULT_MIN_PCT,
    'min_volr': DEFAULT_MIN_VOLR,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}


def get_smart_50_symbols():
    """
    Akıllı emtia seçimi: Binance Futures'tan 500K+ USDT hacimli ilk 50 emtia
    🔧 DÜZELTME: Futures API'yi doğru kullan
    """
    try:
        if not BINANCE_AVAILABLE:
            logger.warning("⚠️ Binance client yok - manuel liste kullanılıyor")
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", 
                   "XRPUSDT", "DOGEUSDT", "MATICUSDT", "LINKUSDT", "AVAXUSDT"]
        
        # Binance client (sadece public data için - API key gerekmez)
        client = Client()
        
        # 🔧 DÜZELTME: Futures için doğru API çağrısı
        # Önce futures exchange info'dan aktif sembolleri al
        exchange_info = client.futures_exchange_info()
        active_symbols = []
        
        for symbol_info in exchange_info['symbols']:
            if (symbol_info['status'] == 'TRADING' and 
                symbol_info['quoteAsset'] == 'USDT' and
                symbol_info['contractType'] == 'PERPETUAL'):
                active_symbols.append(symbol_info['symbol'])
        
        logger.info(f"📊 Futures'ta {len(active_symbols)} aktif USDT perpetual bulundu")
        
        # Şimdi 24h ticker verilerini al
        tickers_24h = client.futures_ticker()
        
        # Hacim bazında filtrele ve sırala
        valid_symbols = []
        for ticker in tickers_24h:
            symbol = ticker['symbol']
            if symbol in active_symbols:
                try:
                    # quoteVolume USDT cinsinden hacim
                    volume_usdt = float(ticker.get('quoteVolume', 0))
                    if volume_usdt >= 500_000:  # 500K USDT minimum
                        valid_symbols.append({
                            'symbol': symbol,
                            'volume': volume_usdt,
                            'change_24h': float(ticker.get('priceChangePercent', 0))
                        })
                except (ValueError, TypeError):
                    continue
        
        # Hacme göre sırala
        sorted_symbols = sorted(valid_symbols, 
                               key=lambda x: x['volume'], 
                               reverse=True)[:50]
        
        final_symbols = [s['symbol'] for s in sorted_symbols]
        
        if len(final_symbols) >= 10:
            logger.info(f"✅ Futures akıllı seçim: {len(final_symbols)} emtia")
            logger.info(f"🔝 En yüksek hacim: {sorted_symbols[0]['symbol']} (${sorted_symbols[0]['volume']:,.0f})")
            logger.info(f"📉 En düşük hacim: {sorted_symbols[-1]['symbol']} (${sorted_symbols[-1]['volume']:,.0f})")
            
            # İlk 10'u göster
            top_10 = [s['symbol'] for s in sorted_symbols[:10]]
            logger.info(f"🏆 Top 10: {', '.join(top_10)}")
            
            return final_symbols
        else:
            logger.warning(f"⚠️ Yetersiz futures emtia ({len(final_symbols)}), genişletiliyor...")
            # Hacim kriterini düşür
            relaxed_symbols = []
            for ticker in tickers_24h:
                symbol = ticker['symbol']
                if symbol in active_symbols:
                    try:
                        volume_usdt = float(ticker.get('quoteVolume', 0))
                        if volume_usdt >= 100_000:  # 100K'ya düşür
                            relaxed_symbols.append({
                                'symbol': symbol,
                                'volume': volume_usdt
                            })
                    except (ValueError, TypeError):
                        continue
            
            relaxed_sorted = sorted(relaxed_symbols, 
                                   key=lambda x: x['volume'], 
                                   reverse=True)[:50]
            
            relaxed_final = [s['symbol'] for s in relaxed_sorted]
            logger.info(f"✅ Gevşetilmiş kriterle: {len(relaxed_final)} emtia (100K+ USDT)")
            return relaxed_final
        
    except Exception as e:
        logger.error(f"❌ Futures sembol seçimi hatası: {e}")
        # Bilinen popüler futures sembolleri
        fallback_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
            "XRPUSDT", "DOGEUSDT", "MATICUSDT", "LINKUSDT", "AVAXUSDT",
            "LTCUSDT", "DOTUSDT", "ATOMUSDT", "FILUSDT", "TRXUSDT",
            "UNIUSDT", "CHZUSDT", "VETUSDT", "XLMUSDT", "EOSUSDT",
            "SANDUSDT", "MANAUSDT", "AXSUSDT", "FTMUSDT", "NEARUSDT",
            "ALGOUSDT", "ONEUSDT", "ZILUSDT", "ICPUSDT", "THETAUSDT",
            "AAVEUSDT", "COMPUSDT", "SUSHIUSDT", "YFIUSDT", "SNXUSDT",
            "MKRUSDT", "CRVUSDT", "1INCHUSDT", "ALPHAUSDT", "ENSUSDT",
            "GALAUSDT", "JASMYUSDT", "MASKUSDT", "ROSEUSDT", "DYDXUSDT",
            "GMTUSDT", "APTUSDT", "OPUSDT", "ARBUSDT", "LDOUSDT"
        ]
        logger.info(f"🔄 Fallback: {len(fallback_symbols)} bilinen futures sembol")
        return fallback_symbols
class WebSocketAnalyzer:
    """
    🔥 DÜZELTME: Real-time WebSocket Analyzer - TradingView benzeri anlık güncellemeler
    Hibrit yaklaşım: Ticker (real-time fiyat) + Kline (yapısal veri) + Sürekli indikatör hesaplama
    """
    
    def __init__(self):
        self.symbols = get_smart_50_symbols()
        self.ws = None
        self.is_running = False
        
        # Real-time data storage
        self.live_prices = {}
        self.last_analysis = {}
        self.kline_cache = {}
        self.update_count = 0
        self.price_update_count = 0
        self.indicator_update_count = 0
        self.error_count = 0
        
        # 🔥 DÜZELTME: Hibrit multi-stream yaklaşımı
        timeframe = current_settings.get('timeframe', '15m')
        self.current_timeframe = timeframe
        
        # Hem ticker (anlık fiyat) hem kline (yapısal veri) stream'leri
        ticker_streams = [f"{symbol.lower()}@ticker" for symbol in self.symbols[:25]]  # İlk 25 için ticker
        kline_streams = [f"{symbol.lower()}@kline_{timeframe}" for symbol in self.symbols[:25]]  # İlk 25 için kline
        
        self.streams = ticker_streams + kline_streams
        self.ws_url = self._get_ws_url()
        
        # Performans sayaçları
        self.last_indicator_calc = {}
        
        logger.info(f"🌐 Real-Time WebSocket Analyzer başlatıldı")
        logger.info(f"📊 Semboller: {len(self.symbols)} adet")
        logger.info(f"⚡ Ticker streams: {len(ticker_streams)} (real-time fiyat)")
        logger.info(f"📈 Kline streams: {len(kline_streams)} ({timeframe})")
        logger.info(f"🔗 Total streams: {len(self.streams)}")
        logger.info(f"🚀 TradingView seviyesi real-time indikatörler aktif")
    
    def update_timeframe(self, new_timeframe: str):
        """UI'den timeframe değiştiğinde WebSocket'i yeniden başlat"""
        if new_timeframe != self.current_timeframe:
            logger.info(f"⏰ Timeframe değişiyor: {self.current_timeframe} → {new_timeframe}")
            
            # Eski WebSocket'i kapat
            if self.ws and self.is_running:
                self.stop()
                time.sleep(2)
            
            # Yeni timeframe ile stream'leri güncelle
            self.current_timeframe = new_timeframe
            
            ticker_streams = [f"{symbol.lower()}@ticker" for symbol in self.symbols[:25]]
            kline_streams = [f"{symbol.lower()}@kline_{new_timeframe}" for symbol in self.symbols[:25]]
            self.streams = ticker_streams + kline_streams
            self.ws_url = self._get_ws_url()
            
            # Cache'i temizle
            self.kline_cache.clear()
            self.last_analysis.clear()
            self.last_indicator_calc.clear()
            
            # WebSocket'i yeniden başlat
            self.start()
            
            logger.info(f"✅ WebSocket yeni timeframe ile başlatıldı: {new_timeframe}")
    
    def _get_ws_url(self):
        """Multi-stream WebSocket URL oluştur"""
        if len(self.streams) > 1024:  # Binance limit
            self.streams = self.streams[:1024]
            logger.warning(f"⚠️ Stream limiti aşıldı, {len(self.streams)} stream kullanılıyor")
        
        streams_param = "/".join(self.streams)
        return f"wss://fstream.binance.com/stream?streams={streams_param}"
    
    def on_message(self, ws, message):
        """
        🔥 DÜZELTME: Hibrit mesaj işleyici - Hem ticker hem kline
        """
        try:
            data = json.loads(message)

            # Multi-stream response kontrolü
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                event_data = data["data"]
            else:
                event_data = data
                stream_name = "unknown"

            # Stream tipini belirle
            if "@ticker" in stream_name:
                self._handle_ticker_message(event_data)
            elif "@kline" in stream_name:
                self._handle_kline_message(event_data)
            else:
                logger.debug(f"⚠️ Bilinmeyen stream tipi: {stream_name}")

        except Exception as e:
            self.error_count += 1
            logger.debug(f"❌ WebSocket mesaj hatası: {e}")

    def _handle_ticker_message(self, ticker_data):
        """Ticker mesajlarını işle - Real-time fiyat güncellemeleri"""
        try:
            symbol = ticker_data.get("s", "").upper()
            if not symbol or symbol not in self.symbols:
                return

            current_price = float(ticker_data.get("c", 0))  # Close price
            if current_price <= 0:
                return

            # Real-time fiyat güncelle
            self.live_prices[symbol] = {
                "symbol": symbol,
                "price": current_price,
                "change_24h": float(ticker_data.get("P", 0)),  # 24h price change %
                "volume": float(ticker_data.get("v", 0)),      # Volume
                "high_24h": float(ticker_data.get("h", 0)),    # 24h high
                "low_24h": float(ticker_data.get("l", 0)),     # 24h low
                "timestamp": datetime.now(LOCAL_TZ),
                "data_source": "TICKER"
            }

            self.price_update_count += 1
            self.update_count += 1

            # 🔥 Real-time indikatör hesaplama - Her 3 fiyat güncellemesinde
            if self.price_update_count % 3 == 0:
                self._trigger_indicator_calculation(symbol)

        except Exception as e:
            logger.debug(f"❌ Ticker mesaj işleme hatası: {e}")

    def _handle_kline_message(self, kline_data):
        """Kline mesajlarını işle - Yapısal veri güncellemeleri"""
        try:
            if "k" not in kline_data:
                return

            kline = kline_data["k"]
            symbol = kline.get("s", "").upper()
            if not symbol or symbol not in self.symbols:
                return

            # Kline verilerini parse et
            open_time = pd.to_datetime(int(kline["t"]), unit="ms").tz_localize("UTC").tz_convert(LOCAL_TZ)
            close_time = pd.to_datetime(int(kline["T"]), unit="ms").tz_localize("UTC").tz_convert(LOCAL_TZ)
            open_price = float(kline["o"])
            high_price = float(kline["h"])
            low_price = float(kline["l"])
            close_price = float(kline["c"])
            volume = float(kline["v"])
            is_closed = bool(kline.get("x", False))

            # Kline cache'ini güncelle
            self._update_kline_cache(symbol, {
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "is_closed": is_closed
            })

            # Yeni kline kapandığında zorla indikatör hesapla
            if is_closed:
                self._trigger_indicator_calculation(symbol, force=True)

        except Exception as e:
            logger.debug(f"❌ Kline mesaj işleme hatası: {e}")

    def _update_kline_cache(self, symbol: str, kline_data: dict):
        """Kline cache'ini güncelle"""
        try:
            # İlk kline verisi yükleme
            if symbol not in self.kline_cache:
                df = fetch_klines(symbol, self.current_timeframe)
                if df is None or df.empty or len(df) < 30:
                    return
                self.kline_cache[symbol] = df
                logger.debug(f"🔄 {symbol}: İlk kline cache oluşturuldu ({len(df)} mum)")
                return

            df = self.kline_cache[symbol].copy()

            # Son kline'ı güncelle
            last_idx = len(df) - 1
            df.loc[last_idx, "close"] = kline_data["close"]
            df.loc[last_idx, "high"] = max(df.iloc[last_idx]["high"], kline_data["high"])
            df.loc[last_idx, "low"] = min(df.iloc[last_idx]["low"], kline_data["low"])
            df.loc[last_idx, "volume"] = kline_data["volume"]

            # Yeni kline kapandıysa yeni satır ekle
            if kline_data["is_closed"]:
                new_row = {
                    "open_time": kline_data["open_time"],
                    "open": kline_data["open"],
                    "high": kline_data["high"],
                    "low": kline_data["low"],
                    "close": kline_data["close"],
                    "volume": kline_data["volume"],
                    "close_time": kline_data["close_time"],
                }

                # Sütun kontrolü
                for col in new_row.keys():
                    if col not in df.columns:
                        df[col] = pd.NA

                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # Son 500 kline'ı sakla
                if len(df) > 500:
                    df = df.tail(500).reset_index(drop=True)

                logger.debug(f"📊 {symbol}: Yeni kline eklendi, toplam: {len(df)}")

            # Cache'i güncelle
            self.kline_cache[symbol] = df

        except Exception as e:
            logger.debug(f"❌ {symbol} kline cache güncelleme hatası: {e}")

    def _trigger_indicator_calculation(self, symbol: str, force: bool = False):
        """İndikatör hesaplamasını tetikle - Rate limiting ile"""
        try:
            current_time = time.time()
            
            # Rate limiting: Her sembol için minimum 2 saniye bekle (force edilmedikçe)
            if not force and symbol in self.last_indicator_calc:
                if current_time - self.last_indicator_calc[symbol] < 2.0:
                    return

            self.last_indicator_calc[symbol] = current_time
            self._calculate_indicators_realtime(symbol)

        except Exception as e:
            logger.debug(f"❌ {symbol} indikatör tetikleme hatası: {e}")

    def _calculate_indicators_realtime(self, symbol: str):
        """Real-time indikatör hesaplama - Core fonksiyon"""
        try:
            # Kline verisi kontrolü
            if symbol not in self.kline_cache:
                return

            df = self.kline_cache[symbol].copy()
            if df is None or df.empty or len(df) < 30:
                return

            # Real-time fiyat ile son kline'ı güncelle
            if symbol in self.live_prices:
                current_price = self.live_prices[symbol]["price"]
                df.loc[df.index[-1], "close"] = current_price

            # İndikatör hesaplama
            metrics = compute_consecutive_metrics(df)
            if not metrics or metrics.get("run_type") in (None, "", "none"):
                return

            # AI skorunu hesapla
            try:
                ai_score = float(ai_model.predict_score(metrics))
            except Exception:
                ai_score = 0.0

            # Minimum kalite kontrolü
            if (ai_score < 30 or 
                int(metrics.get("run_count", 0)) < 3 or 
                abs(float(metrics.get("run_perc", 0) or 0)) < 0.5):
                return

            # Deviso detaylı analizi
            try:
                from core.indicators import get_deviso_detailed_analysis
                deviso_details = get_deviso_detailed_analysis(df)
                trend_direction = deviso_details.get("trend_direction", "Belirsiz")
            except Exception as e:
                logger.debug(f"⚠️ {symbol}: Deviso analiz hatası: {e}")
                trend_direction = "Belirsiz"

            # Analiz sonucunu kaydet
            self.last_analysis[symbol] = {
                "symbol": symbol,
                "timeframe": self.current_timeframe,
                "last_close": float(self.live_prices.get(symbol, {}).get("price", df.iloc[-1]["close"])),
                "run_type": metrics.get("run_type"),
                "run_count": int(metrics.get("run_count", 0)),
                "run_perc": float(metrics.get("run_perc", 0) or 0),
                "gauss_run": float(metrics.get("gauss_run", 0) or 0),
                "gauss_run_perc": float(metrics.get("gauss_run_perc", 0) or 0),
                "log_volume": metrics.get("log_volume"),
                "log_volume_momentum": metrics.get("log_volume_momentum"),
                "deviso_ratio": float(metrics.get("deviso_ratio", 0) or 0),
                "c_signal_momentum": float(metrics.get("c_signal_momentum", 0) or 0),
                "ai_score": ai_score,
                "last_update": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "trend_direction": trend_direction,
                "trend_strength": abs(float(metrics.get("deviso_ratio", 0) or 0)),
                "data_source": "REALTIME_HYBRID",
            }

            self.indicator_update_count += 1

            logger.debug(
                f"⚡ {symbol}: REAL-TIME analiz - AI:{ai_score:.0f}% "
                f"Deviso:{float(metrics.get('deviso_ratio', 0) or 0):.2f} "
                f"Trend:{trend_direction} (#{self.indicator_update_count})"
            )

        except Exception as e:
            logger.debug(f"❌ {symbol} real-time indikatör hatası: {e}")
    def on_error(self, ws, error):
        """WebSocket hata işleyici"""
        self.error_count += 1
        logger.error(f"❌ WebSocket hatası: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket kapatma işleyici"""
        self.is_running = False
        logger.info(f"🔴 WebSocket bağlantısı kapandı: {close_status_code}")
    
    def on_open(self, ws):
        """WebSocket açılma işleyici"""
        self.is_running = True
        logger.info(f"🟢 WebSocket bağlantısı açıldı - {len(self.symbols)} sembol")
        logger.info(f"⚡ Real-time streams aktif: {len([s for s in self.streams if '@ticker' in s])} ticker + {len([s for s in self.streams if '@kline' in s])} kline")
    
    def start(self):
        """WebSocket'i başlat"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Ayrı thread'de çalıştır
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
            logger.info("🚀 Real-time WebSocket thread başlatıldı")
            
        except Exception as e:
            logger.error(f"❌ WebSocket başlatma hatası: {e}")
    
    def stop(self):
        """WebSocket'i durdur"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        logger.info("🛑 WebSocket durduruldu")
    
    def get_current_data(self):
        """Mevcut analiz verilerini DataFrame olarak döndür"""
        try:
            if not self.last_analysis:
                return pd.DataFrame()
            
            data_list = list(self.last_analysis.values())
            df = pd.DataFrame(data_list)
            
            # Sıralama
            df = df.sort_values(
                by=['ai_score', 'trend_strength', 'run_perc', 'gauss_run'],
                ascending=[False, False, False, False]
            )
            
            logger.debug(f"📊 Real-time WebSocket verisi: {len(df)} sinyal (Ticker: {self.price_update_count}, İndikatör: {self.indicator_update_count})")
            return df
            
        except Exception as e:
            logger.error(f"❌ WebSocket veri dönüştürme hatası: {e}")
            return pd.DataFrame()
    
    def get_performance_stats(self):
        """Performans istatistiklerini döndür"""
        return {
            "total_updates": self.update_count,
            "price_updates": self.price_update_count,
            "indicator_updates": self.indicator_update_count,
            "errors": self.error_count,
            "live_symbols": len(self.live_prices),
            "analyzed_symbols": len(self.last_analysis),
            "cached_symbols": len(self.kline_cache),
            "is_running": self.is_running
        }


def websocket_scan_worker():
    """
    🔥 DÜZELTME: Real-time WebSocket tarama işleyicisi
    TradingView benzeri sürekli güncelleme
    """
    global auto_scan_active, websocket_manager
    
    # WebSocket analyzer başlat
    websocket_manager = WebSocketAnalyzer()
    websocket_manager.start()
    
    # Bağlantının kurulmasını bekle
    time.sleep(5)
    
    loop_count = 0
    
    while auto_scan_active:
        try:
            loop_count += 1
            
            if websocket_manager.is_running:
                # WebSocket'ten gelen analiz verilerini al
                result_data = websocket_manager.get_current_data()
                
                if not result_data.empty:
                    config.current_data = result_data  # GLOBAL GÜNCELLEME
                    
                    # Her 10 döngüde bir performans logu
                    if loop_count % 10 == 0:
                        stats = websocket_manager.get_performance_stats()
                        logger.info(
                            f"⚡ Real-time tarama: {len(result_data)} sinyal | "
                            f"Ticker:{stats['price_updates']} | "
                            f"İndikatör:{stats['indicator_updates']} | "
                            f"Live:{stats['live_symbols']} | "
                            f"Analiz:{stats['analyzed_symbols']}"
                        )
                
                # 🔥 DÜZELTME: Daha hızlı güncelleme (1 saniye)
                time.sleep(1)
            else:
                logger.warning("⚠️ WebSocket bağlantısı kesildi - yeniden bağlanıyor...")
                websocket_manager.start()
                time.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ WebSocket tarama hatası: {e}")
            time.sleep(5)
    
    # WebSocket'i durdur
    if websocket_manager:
        websocket_manager.stop()


def start_auto_scan():
    """Real-time WebSocket tabanlı otomatik taramayı başlat"""
    global auto_scan_active
    
    if not auto_scan_active:
        auto_scan_active = True
        thread = threading.Thread(target=websocket_scan_worker, daemon=True)
        thread.start()
        logger.info("🚀 Real-time WebSocket otomatik tarama başlatıldı")


def stop_auto_scan():
    """Real-time WebSocket otomatik taramayı durdur"""
    global auto_scan_active, websocket_manager
    auto_scan_active = False
    
    if websocket_manager:
        websocket_manager.stop()
    
    logger.info("⛔ Real-time WebSocket otomatik tarama durduruldu")


# 🔥 DÜZELTME: TABLE_REFRESH_INTERVAL'i hızlandır
config.TABLE_REFRESH_INTERVAL = 500  # 500ms = saniyede 2 güncelleme

# Dash uygulaması oluştur
app = dash.Dash(__name__)
app.title = DASH_CONFIG['title']

# CSS stillerini ekle
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}

        <style>
            body {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            
            .Select-control, .Select-menu-outer, .Select-option {
                background-color: #2a2a2a !important;
                color: #ffffff !important;
                border-color: #404040 !important;
            }
            .Select-option:hover {
                background-color: #404040 !important;
                color: #ffffff !important;
            }
            .Select-value-label, .Select-placeholder {
                color: #ffffff !important;
            }
            .Select-arrow-zone {
                color: #ffffff !important;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #2a2a2a 0%, #363636 100%);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #404040;
            }
            
            .stat-value {
                font-size: 1.5rem;
                font-weight: bold;
                color: #22c55e;
                margin-bottom: 0.25rem;
            }
            
            .stat-label {
                font-size: 0.875rem;
                color: #cccccc;
            }
            
            .control-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                padding: 12px 24px;
                color: white;
                font-weight: 600;
                cursor: pointer;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.3s ease;
                margin: 5px;
            }
            
            .control-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .auto-button {
                background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            }
            
            .auto-button.active {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            }
            
            .live-button {
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            }
            
            .live-button.active {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            
            /* 🔥 YENİ: Real-time indicator animasyonları */
            .realtime-indicator {
                display: inline-block;
                animation: pulse 1s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .live-data-badge {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.7rem;
                font-weight: bold;
            }
            
            .rest-data-badge {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.7rem;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout'u oluştur
app.layout = create_layout()
# Callbacks
@app.callback(
    [Output('btn-auto', 'children'),
     Output('btn-auto', 'className')],
    Input('btn-auto', 'n_clicks')
)
def toggle_auto_scan(n_clicks):
    """Real-time WebSocket otomatik tarama butonunu kontrol et"""
    global auto_scan_active
    
    if n_clicks > 0:
        if auto_scan_active:
            stop_auto_scan()
            return "🔄 Real-Time Başlat", "control-button auto-button"
        else:
            start_auto_scan()
            return "⛔ Real-Time Durdur", "control-button auto-button active"
    
    return "🔄 Real-Time Başlat", "control-button auto-button"


@app.callback(
    Output('trading-status', 'children'),
    [Input('btn-start-live-trading', 'n_clicks'),
     Input('btn-stop-live-trading', 'n_clicks')]
)
def control_live_trading(start_clicks, stop_clicks):
    """Live Trading kontrolü"""
    if not LIVE_TRADER_AVAILABLE:
        return "❌ Live trader modülü bulunamadı - .env dosyasını kontrol edin"
    
    ctx = dash.callback_context
    
    if not ctx.triggered:
        status = "🔴 Live Trading Durduruldu" if not is_live_trading_active() else "🟢 Live Trading Aktif"
        return status
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'btn-start-live-trading' and start_clicks > 0:
        success = start_live_trading()
        if success:
            return "🟢 Live Trading Başarıyla Başlatıldı ✅"
        else:
            return "❌ Live Trading Başlatılamadı - API anahtarlarını kontrol edin"
    
    elif trigger_id == 'btn-stop-live-trading' and stop_clicks > 0:
        stop_live_trading()
        return "🔴 Live Trading Durduruldu - Tüm pozisyonlar kapatıldı"
    
    return "🔴 Live Trading Durduruldu" if not is_live_trading_active() else "🟢 Live Trading Aktif"


@app.callback(
    Output('api-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_api_status(n_intervals):
    """API bağlantı durumunu göster"""
    if not LIVE_TRADER_AVAILABLE:
        return html.Div("❌ Live Trader Yok", style={'color': '#ef4444'})
    
    try:
        status = get_live_trading_status()
        if status['api_connected']:
            return html.Div([
                html.Span("✅ API Bağlı", style={'color': '#22c55e'}),
                html.Br(),
                html.Small(f"Bakiye: ${status['balance']:.2f}", style={'color': '#cccccc'})
            ])
        else:
            return html.Div("❌ API Bağlantısı Yok", style={'color': '#ef4444'})
    except:
        return html.Div("⚠️ API Durumu Bilinmiyor", style={'color': '#f59e0b'})


@app.callback(
    Output('api-connection-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_api_connection_status(n_intervals):
    """API bağlantı durumu göstergesi + Real-time WebSocket status"""
    ws_status = ""
    ws_color = '#9ca3af'
    
    if websocket_manager:
        if websocket_manager.is_running:
            stats = websocket_manager.get_performance_stats()
            ws_status = f"🟢 Real-Time Aktif (T:{stats['price_updates']} I:{stats['indicator_updates']} L:{stats['live_symbols']})"
            ws_color = '#22c55e'
        else:
            ws_status = "🔴 Real-Time Bağlantısı Kesildi"
            ws_color = '#ef4444'
    else:
        ws_status = "⚪ Real-Time Başlatılmadı"
    
    if not LIVE_TRADER_AVAILABLE:
        return html.Div([
            html.Span("❌ Live Trader Modülü Yüklenmedi", style={'color': '#ef4444'}),
            html.Br(),
            html.Small(ws_status, style={'color': ws_color})
        ])
    
    try:
        status = get_live_trading_status()
        
        if status['api_connected']:
            return html.Div([
                html.Span("🟢 API Bağlantısı Aktif", style={'color': '#22c55e', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(f"Environment: {status['environment']}", style={'color': '#9ca3af'}),
                html.Br(),
                html.Small(f"Bakiye: ${status['balance']:.2f} USDT", style={'color': '#22c55e'}),
                html.Br(),
                html.Small(ws_status, style={'color': ws_color, 'fontWeight': 'bold'})
            ])
        else:
            return html.Div([
                html.Span("🔴 API Bağlantısı Yok", style={'color': '#ef4444', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(ws_status, style={'color': ws_color})
            ])
            
    except Exception as e:
        return html.Div([
            html.Span("⚠️ API Durumu Kontrol Edilemiyor", style={'color': '#f59e0b', 'fontWeight': 'bold'}),
            html.Br(),
            html.Small(ws_status, style={'color': ws_color})
        ])


@app.callback(
    Output('performance-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_metrics(n_intervals):
    """Performans metriklerini güncelle"""
    try:
        # Config'den mevcut modu al
        if config.is_live_mode():
            current_capital = config.live_capital
            open_positions = config.live_positions.copy()  
            mode_info = "🤖 Live"
        else:
            current_capital = config.paper_capital
            open_positions = config.paper_positions.copy()  
            mode_info = "📝 Paper"
        
        # Gerçekleşmemiş P&L hesapla - Real-time fiyatlarla
        total_unrealized_pnl = 0.0
        
        for symbol, position in list(open_positions.items()):
            # Önce WebSocket'ten real-time fiyat dene
            current_price = None
            if websocket_manager and symbol in websocket_manager.live_prices:
                current_price = websocket_manager.live_prices[symbol]['price']
            
            # Fallback: REST API
            if current_price is None:
                current_price = get_current_price(symbol)
                if current_price is None:
                    current_price = position['entry_price']
            
            if position['side'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            total_unrealized_pnl += unrealized_pnl
        
        # Trade geçmişi analizi
        metrics = calculate_performance_metrics()
        
        # Real-time data indicator
        rt_indicator = ""
        if websocket_manager and websocket_manager.is_running:
            stats = websocket_manager.get_performance_stats()
            rt_indicator = f" ⚡ RT:{stats['live_symbols']}"
        
        return html.Div([
            html.Div([
                html.H4(f"💰 {mode_info} Sermaye{rt_indicator}", style={'margin': '0', 'fontSize': '14px', 'color': '#22c55e'}),
                html.H2(f"${current_capital:.2f}", style={'color': '#22c55e', 'margin': '5px 0', 'fontSize': '18px'})
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #22c55e', 
                'borderRadius': '10px', 'backgroundColor': '#f8fff8'
            }),
            
            html.Div([
                html.H4("📈 Toplam Kar", style={'margin': '0', 'fontSize': '14px', 'color': '#10b981'}),
                html.H2(f"${metrics['total_gain']:.2f}", style={'color': '#10b981', 'margin': '5px 0', 'fontSize': '18px'})
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #10b981', 
                'borderRadius': '10px', 'backgroundColor': '#f0fdf4'
            }),
            
            html.Div([
                html.H4("📉 Toplam Zarar", style={'margin': '0', 'fontSize': '14px', 'color': '#ef4444'}),
                html.H2(f"${metrics['total_loss']:.2f}", style={'color': '#ef4444', 'margin': '5px 0', 'fontSize': '18px'})
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #ef4444', 
                'borderRadius': '10px', 'backgroundColor': '#fef2f2'
            }),
            
            html.Div([
                html.H4("🎯 Gerçekleşmemiş P&L", style={'margin': '0', 'fontSize': '14px', 'color': '#8b5cf6'}),
                html.H2(f"${total_unrealized_pnl:.2f}", style={
                    'color': '#22c55e' if total_unrealized_pnl >= 0 else '#ef4444', 
                    'margin': '5px 0', 'fontSize': '18px'
                })
            ], style={
                'width': '24%', 'display': 'inline-block', 'textAlign': 'center', 
                'margin': '0.5%', 'padding': '15px', 'border': '2px solid #8b5cf6', 
                'borderRadius': '10px', 'backgroundColor': '#faf5ff'
            })
        ])
        
    except Exception as e:
        logger.error(f"❌ Performans metrikleri hatası: {e}")
        return html.Div("Performans verileri yüklenemiyor...")


@app.callback(
    [Output('positions-table', 'data'),
     Output('trades-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_trading_tables(n_intervals):
    """Trading tablolarını güncelle - Real-time fiyatlarla"""
    positions_data = []
    
    # Config'den mevcut pozisyonları al
    if config.is_live_mode():
        current_positions = config.live_positions.copy()
        mode_info = "Live"
    else:
        current_positions = config.paper_positions.copy()
        mode_info = "Paper"
    
    # Açık pozisyonlar
    try:
        for symbol, position in list(current_positions.items()):
            # Real-time fiyat alma - Önce WebSocket, sonra REST
            current_price = None
            data_source_badge = ""
            
            if websocket_manager and symbol in websocket_manager.live_prices:
                live_data = websocket_manager.live_prices[symbol]
                current_price = live_data['price']
                
                # Veri yaşını kontrol et
                last_update = live_data['timestamp']
                seconds_ago = (datetime.now(LOCAL_TZ) - last_update).total_seconds()
                
                if seconds_ago < 30:
                    data_source_badge = "🟢 LIVE"
                else:
                    data_source_badge = "🟡 STALE"
            
            # Fallback: REST API'den fiyat al
            if current_price is None:
                current_price = get_current_price(symbol)
                data_source_badge = "🔴 REST"
                if current_price is None:
                    current_price = position['entry_price']
                    data_source_badge = "⚪ CACHED"
            
            current_value = position['quantity'] * current_price
            
            if position['side'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            auto_sltp_status = f"🤖 Auto ({mode_info})" if position.get('auto_sltp', False) else f"📱 Manual ({mode_info})"
            
            positions_data.append({
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'invested_amount': position['invested_amount'],
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'ai_score': position['signal_data']['ai_score'],
                'sltp_mode': auto_sltp_status,
                'data_source': data_source_badge
            })
    except Exception as e:
        logger.error(f"❌ Pozisyon tablosu güncelleme hatası: {e}")
    
    # Trade geçmişi (değişiklik yok)
    try:
        trades_df = load_trades_from_csv()
        if not trades_df.empty:
            trades_data = []
            for _, row in trades_df.sort_values('timestamp', ascending=False).head(50).iterrows():
                trades_data.append({
                    'timestamp': row.get('timestamp', ''),
                    'symbol': row.get('symbol', ''),
                    'side': row.get('side', ''),
                    'quantity': row.get('quantity', 0),
                    'entry_price': row.get('entry_price', 0),
                    'exit_price': row.get('exit_price', 0),
                    'pnl': row.get('pnl', 0),
                    'ai_score': row.get('ai_score', 0),
                    'close_reason': row.get('close_reason', ''),
                    'status': row.get('status', '')
                })
        else:
            trades_data = []
    except Exception as e:
        logger.error(f"❌ Trade tablosu güncelleme hatası: {e}")
        trades_data = []
    
    return positions_data, trades_data
@app.callback(
    [Output('signals-table', 'data'),
     Output('status-text', 'children'),
     Output('stats-panel', 'children')],
    [Input('interval-component', 'n_intervals')],  
    [State('dd-timeframe', 'value'),
     State('inp-min-streak', 'value'),
     State('inp-min-pct', 'value'),
     State('inp-min-volr', 'value'),
     State('inp-min-ai', 'value')]
)
def update_signals(n_intervals, tf, min_streak, min_pct, min_volr, min_ai_pct):  
    """
    🔥 DÜZELTME: Real-time WebSocket entegre sinyal güncelleme
    TradingView benzeri sürekli güncellenen indikatörler
    """
    global current_settings
    
    tf = tf or DEFAULT_TIMEFRAME
    min_streak = int(min_streak or DEFAULT_MIN_STREAK)
    min_pct = float(min_pct or DEFAULT_MIN_PCT)
    min_volr = float(min_volr or DEFAULT_MIN_VOLR)
    min_ai_score = float(min_ai_pct or 30) / 100
    
    # Timeframe değişimi kontrolü
    if tf != current_settings.get('timeframe'):
        if websocket_manager:
            websocket_manager.update_timeframe(tf)
    
    current_settings.update({
        'timeframe': tf,
        'min_streak': min_streak,
        'min_pct': min_pct,
        'min_volr': min_volr,
        'min_ai': min_ai_pct
    })
    
    # Real-time WebSocket'ten veri al
    if config.current_data is None or config.current_data.empty:
        ws_status = ""
        ws_color = '#9ca3af'
        
        if websocket_manager:
            if websocket_manager.is_running:
                stats = websocket_manager.get_performance_stats()
                ws_status = f" | 🟢 Real-Time aktif (T:{stats['price_updates']} I:{stats['indicator_updates']} L:{stats['live_symbols']})"
                ws_color = '#22c55e'
            else:
                ws_status = " | 🔴 Real-Time bağlantı problemi"
                ws_color = '#ef4444'
        
        status_text = f"Real-Time WebSocket tarama aktif - Veri bekleniyor...{ws_status}" if auto_scan_active else "Real-Time tarama başlatın"
        return [], status_text, []
    
    df = config.current_data.copy()
    original_count = len(df)
    
    # Filtreleme
    if min_ai_pct > 70:
        df = df[df['ai_score'] >= min_ai_pct]
        
    if min_streak > 6:
        df = df[df['run_count'] >= min_streak]
        
    if min_pct > 3.0:
        df = df[df['run_perc'] >= min_pct]

    if min_volr > 3.0:
        df = df[(df['log_volume'].isna()) | (df['log_volume'] >= min_volr)]
    
    # Sıralama
    sort_columns = ['ai_score']
    if 'trend_strength' in df.columns:
        sort_columns.append('trend_strength')
    sort_columns.extend(['run_perc', 'gauss_run', 'log_volume_momentum'])
    
    df = df.sort_values(by=sort_columns, ascending=[False] * len(sort_columns))
    
    if df.empty:
        status_text = f"Filtreler çok sıkı! {original_count} sinyal var."
        if auto_scan_active:
            status_text += f" | Real-Time aktif"
        return [], status_text, []
    
    table_data = []
    for _, row in df.iterrows():
        ai_score_unicode = create_ai_score_bar(row['ai_score'])
        
        # Live bot durumu
        live_status = "⭐"
        if LIVE_TRADER_AVAILABLE:
            live_status = get_live_bot_status_for_symbol(row['symbol'])
        
        # Real-time fiyat ve veri kaynağı göstergesi
        price_source = "🔴 REST"
        realtime_badge = ""
        
        if websocket_manager and row['symbol'] in websocket_manager.live_prices:
            live_data = websocket_manager.live_prices[row['symbol']]
            last_update = live_data['timestamp']
            seconds_ago = (datetime.now(LOCAL_TZ) - last_update).total_seconds()
            
            if seconds_ago < 10:
                price_source = "🟢 LIVE"
                realtime_badge = "⚡"
            elif seconds_ago < 30:
                price_source = "🟡 LIVE"
                realtime_badge = "⚡"
            else:
                price_source = "🔴 OLD"
        
        # Indikatör güncelleme durumu
        indicator_status = ""
        if websocket_manager and row['symbol'] in websocket_manager.last_analysis:
            analysis_data = websocket_manager.last_analysis[row['symbol']]
            if analysis_data.get('data_source') == 'REALTIME_HYBRID':
                indicator_status = "🔥"  # Real-time hesaplanmış
        
        table_data.append({
            'symbol': row['symbol'],
            'timeframe': row['timeframe'].upper(),
            'side': row['run_type'].upper(),
            'run_count': row['run_count'],
            'run_perc': row['run_perc'],
            'gauss_run': row['gauss_run'],
            'gauss_run_perc': row['gauss_run_perc'] if row['gauss_run_perc'] is not None else 0,
            'log_volume': row['log_volume'] if row['log_volume'] is not None else 0,
            'log_volume_momentum': row['log_volume_momentum'] if row['log_volume_momentum'] is not None else 0,
            'deviso_ratio': row['deviso_ratio'],
            'c_signal_momentum': row.get('c_signal_momentum', 0.0),
            'ai_score': ai_score_unicode,
            'live_status': live_status,
            'price_source': price_source,
            'realtime_indicator': f"{realtime_badge}{indicator_status}",
            'timestamp': datetime.now(LOCAL_TZ).strftime('%H:%M:%S')
        })
    
    long_count = len(df[df['run_type'] == 'long'])
    short_count = len(df[df['run_type'] == 'short'])
    
    top_3_df = df.head(3)
    top_3_info = f"(En iyi 3: {', '.join(top_3_df['symbol'].tolist())})" if len(top_3_df) > 0 else ""
    
    # Real-time WebSocket istatistikleri
    ws_symbol_count = 0
    ws_indicator_count = 0
    ws_updates = 0
    
    if websocket_manager:
        stats = websocket_manager.get_performance_stats()
        ws_symbol_count = stats['live_symbols']
        ws_indicator_count = stats['analyzed_symbols']
        ws_updates = stats['total_updates']
    
    stats_panel = html.Div([
        html.Div([
            html.Div(f"{len(df)}", className="stat-value"),
            html.Div("Sinyal", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{long_count}", className="stat-value", style={'color': '#22c55e'}), 
            html.Div("Long", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{short_count}", className="stat-value", style={'color': '#ef4444'}),  
            html.Div("Short", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{ws_symbol_count}", className="stat-value", style={'color': '#3b82f6'}),
            html.Div("Live Fiyat", className="stat-label")
        ], className="stat-card"),
        html.Div([
            html.Div(f"{ws_indicator_count}", className="stat-value", style={'color': '#8b5cf6'}),
            html.Div("RT İndikatör", className="stat-label")
        ], className="stat-card"),
    ], className="stats-grid")
    
    current_positions = config.live_positions if config.is_live_mode() else config.paper_positions
    current_mode = "Live" if config.is_live_mode() else "Paper"
    
    # Real-time WebSocket status bilgisi
    ws_info = ""
    if websocket_manager and websocket_manager.is_running:
        stats = websocket_manager.get_performance_stats()
        ws_info = f" | 🟢 RT:{ws_updates} (F:{stats['price_updates']} I:{stats['indicator_updates']} L:{ws_symbol_count})"
    elif auto_scan_active:
        ws_info = " | 🔴 RT:Bağlantı problemi"
    
    status = f"{len(df)}/{original_count} SİNYAL: (L:{long_count}, S:{short_count}) | {datetime.now(LOCAL_TZ).strftime('%H:%M:%S')}{ws_info}"
    
    if LIVE_TRADER_AVAILABLE and is_live_trading_active():
        auto_sltp_count = get_auto_sltp_count() if LIVE_TRADER_AVAILABLE else 0
        status += f" | 🤖 {current_mode} Bot aktif ({len(current_positions)}/{MAX_OPEN_POSITIONS}) [Auto SL/TP: {auto_sltp_count}]"
    elif len(current_positions) > 0:
        status += f" | 📝 {current_mode} Trading ({len(current_positions)}/{MAX_OPEN_POSITIONS})"
    
    status += f" {top_3_info}"
    
    return table_data, status, stats_panel


# 🔥 YENİ: Timeframe değişimi callback'i
@app.callback(
    Output('dummy-output', 'children'),  # Dummy output
    [Input('dd-timeframe', 'value')]
)
def handle_timeframe_change(new_timeframe):
    """Timeframe değişimini WebSocket'e bildir"""
    global current_settings
    
    if new_timeframe and new_timeframe != current_settings.get('timeframe'):
        logger.info(f"⏰ UI'den timeframe değişimi: {current_settings.get('timeframe')} → {new_timeframe}")
        
        if websocket_manager and websocket_manager.is_running:
            # WebSocket'i yeni timeframe ile güncelle
            websocket_manager.update_timeframe(new_timeframe)
            
        current_settings['timeframe'] = new_timeframe
    
    return ""  # Dummy return


# 🔥 YENİ: Real-time performans monitoring callback'i
@app.callback(
    Output('realtime-stats', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_realtime_stats(n_intervals):
    """Real-time performans istatistiklerini göster"""
    if not websocket_manager:
        return html.Div("Real-Time Monitor Yok", style={'color': '#9ca3af', 'fontSize': '0.8rem'})
    
    stats = websocket_manager.get_performance_stats()
    
    if stats['is_running']:
        return html.Div([
            html.Span(f"⚡ RT Aktif", style={'color': '#22c55e', 'fontWeight': 'bold'}),
            html.Br(),
            html.Small(f"Fiyat: {stats['price_updates']}", style={'color': '#22c55e'}),
            html.Br(),
            html.Small(f"İndikatör: {stats['indicator_updates']}", style={'color': '#3b82f6'}),
            html.Br(),
            html.Small(f"Canlı: {stats['live_symbols']}/{stats['cached_symbols']}", style={'color': '#8b5cf6'}),
            html.Br(),
            html.Small(f"Hata: {stats['errors']}", style={'color': '#ef4444' if stats['errors'] > 0 else '#9ca3af'})
        ], style={'fontSize': '0.8rem', 'textAlign': 'center'})
    else:
        return html.Div([
            html.Span("🔴 RT Durduruldu", style={'color': '#ef4444', 'fontWeight': 'bold'}),
            html.Br(),
            html.Small(f"Toplam Hata: {stats['errors']}", style={'color': '#9ca3af'})
        ], style={'fontSize': '0.8rem', 'textAlign': 'center'})
# Ana çalıştırma
if __name__ == "__main__":
    setup_csv_files()
    logger.info("🚀 Real-Time AI Crypto Analytics + WebSocket Live Data başlatılıyor...")
    
    # WebSocket gerekli paketler kontrolü
    try:
        import websocket
        logger.info("✅ WebSocket paketi hazır")
    except ImportError:
        logger.error("❌ WebSocket paketi bulunamadı: pip install websocket-client")
        exit(1)
    
    # Binance client kontrolü
    if BINANCE_AVAILABLE:
        logger.info("✅ Binance client hazır - akıllı sembol seçimi aktif")
    else:
        logger.warning("⚠️ python-binance yok - manuel sembol listesi kullanılacak")
    
    # Live trader kontrolü
    if LIVE_TRADER_AVAILABLE:
        logger.info("✅ Live trading modülü hazır")
        logger.info("🔑 .env dosyasındaki API anahtarları ile otomatik bağlantı yapılacak")
        logger.info("🤖 Otomatik SL/TP emirleri Binance sunucu tarafında yönetilecek")
    else:
        logger.warning("⚠️ Live trading modülü bulunamadı - sadece analiz modu")
        logger.warning("⚠️ Live trading için: pip install python-binance")
        logger.warning("⚠️ .env dosyasına BINANCE_API_KEY ve BINANCE_SECRET_KEY ekleyin")
    
    # 🔥 Real-time sistem bilgileri
    logger.info("⚡ Real-time WebSocket data streaming aktif")
    logger.info("📊 500K+ USDT hacimli top 50 emtia otomatik seçilecek")
    logger.info("🌐 Hibrit stream: Ticker (real-time fiyat) + Kline (yapısal veri)")
    logger.info("🔥 TradingView benzeri sürekli indikatör güncellemesi")
    logger.info("⏱️ UI güncelleme: 500ms (saniyede 2 kez)")
    logger.info("📈 İndikatör hesaplama: Her 3 fiyat güncellemesinde")
    logger.info("🎯 Timeframe değişimi: Otomatik WebSocket yeniden bağlantı")
    
    # Performans ayarları loglama
    logger.info(f"🔧 Performans ayarları:")
    logger.info(f"   TABLE_REFRESH_INTERVAL: {config.TABLE_REFRESH_INTERVAL}ms")
    logger.info(f"   MAX_SYMBOLS: {len(get_smart_50_symbols()[:25])} (hibrit stream)")
    logger.info(f"   RATE_LIMITING: 2 saniye/sembol")
    logger.info(f"   CACHE_SIZE: 500 kline/sembol")
    
    # Güvenlik uyarıları
    logger.info("🛡️ Güvenlik uyarıları:")
    logger.info("   📝 Bu eğitim amaçlıdır, yatırım tavsiyesi değildir")
    logger.info("   💰 Real trading yalnızca test fonlarıyla yapılmalıdır")
    logger.info("   🔑 API anahtarlarınızı kimseyle paylaşmayın")
    logger.info("   ⚡ Real-time veriler internet bağlantınıza bağlıdır")
    
    # Son kontroller
    logger.info("🔍 Son kontroller:")
    
    # Sembol listesi kontrolü
    test_symbols = get_smart_50_symbols()
    if len(test_symbols) < 10:
        logger.error("❌ Yetersiz sembol sayısı - minimum 10 gerekli")
        exit(1)
    else:
        logger.info(f"✅ {len(test_symbols)} sembol hazır")
    
    # Config kontrolü
    if not hasattr(config, 'TABLE_REFRESH_INTERVAL'):
        config.TABLE_REFRESH_INTERVAL = 500
        logger.warning("⚠️ TABLE_REFRESH_INTERVAL varsayılan değere ayarlandı")
    
    # Memory management
    import gc
    gc.collect()
    logger.info("🧹 Memory temizlendi")
    
    # Başlatma mesajı
    print("\n" + "="*80)
    print("🚀 REAL-TIME AI CRYPTO ANALYTICS SISTEMI")
    print("="*80)
    print("🌐 WebSocket Real-Time Data Streaming")
    print("📊 TradingView Benzeri Sürekli İndikatör Güncelleme")
    print("🤖 AI-Powered Signal Analysis")
    print("💰 Live Trading Bot (Binance Futures)")
    print("⚡ Milisaniye Seviyesi Fiyat Güncellemeleri")
    print("="*80)
    print(f"🔗 Web Arayüzü: http://{DASH_CONFIG['host']}:{DASH_CONFIG['port']}")
    print("🛑 Durdurmak için: Ctrl+C")
    print("="*80)
    
    # Dash uygulamasını başlat
    try:
        app.run(
            debug=DASH_CONFIG['debug'],
            host=DASH_CONFIG['host'],
            port=DASH_CONFIG['port'],
            threaded=True  # Real-time için threading desteği
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Kullanıcı tarafından durduruldu")
        
        # Temizlik işlemleri
        if websocket_manager and websocket_manager.is_running:
            websocket_manager.stop()
            logger.info("🧹 WebSocket bağlantısı kapatıldı")
        
        if LIVE_TRADER_AVAILABLE and is_live_trading_active():
            logger.info("🤖 Live trading durduruluyor...")
            stop_live_trading()
        
        logger.info("✅ Sistem güvenli bir şekilde kapatıldı")
        
    except Exception as e:
        logger.error(f"❌ Kritik sistem hatası: {e}")
        
        # Acil temizlik
        try:
            if websocket_manager:
                websocket_manager.stop()
            if LIVE_TRADER_AVAILABLE:
                stop_live_trading()
        except:
            pass
        
        logger.error("🚨 Sistem acil kapatma yapıldı")
        exit(1)


# 🔥 YENİ: Sistem bilgi fonksiyonları
def get_system_info():
    """Sistem bilgilerini döndür"""
    info = {
        "websocket_available": True,
        "binance_available": BINANCE_AVAILABLE,
        "live_trader_available": LIVE_TRADER_AVAILABLE,
        "symbols_count": len(get_smart_50_symbols()),
        "refresh_interval": getattr(config, 'TABLE_REFRESH_INTERVAL', 1000),
        "realtime_enabled": True
    }
    
    if websocket_manager:
        info.update(websocket_manager.get_performance_stats())
    
    return info


def validate_system_requirements():
    """Sistem gereksinimlerini kontrol et"""
    requirements = {
        "websocket": False,
        "binance": BINANCE_AVAILABLE,
        "live_trader": LIVE_TRADER_AVAILABLE,
        "symbols": False,
        "config": False
    }
    
    # WebSocket kontrolü
    try:
        import websocket
        requirements["websocket"] = True
    except ImportError:
        pass
    
    # Sembol listesi kontrolü
    symbols = get_smart_50_symbols()
    requirements["symbols"] = len(symbols) >= 10
    
    # Config kontrolü
    requirements["config"] = hasattr(config, 'LOCAL_TZ') and hasattr(config, 'DASH_CONFIG')
    
    return requirements


# 🔥 YENİ: Debugging yardımcı fonksiyonları
def debug_websocket_status():
    """WebSocket durumunu debug et"""
    if not websocket_manager:
        return {"status": "not_initialized"}
    
    stats = websocket_manager.get_performance_stats()
    live_symbols = list(websocket_manager.live_prices.keys())[:10]  # İlk 10 sembol
    analyzed_symbols = list(websocket_manager.last_analysis.keys())[:10]
    
    return {
        "status": "running" if stats['is_running'] else "stopped",
        "stats": stats,
        "sample_live_symbols": live_symbols,
        "sample_analyzed_symbols": analyzed_symbols,
        "timeframe": websocket_manager.current_timeframe if websocket_manager else None
    }


def debug_signal_quality():
    """Sinyal kalitesini debug et"""
    if not config.current_data or config.current_data.empty:
        return {"status": "no_data"}
    
    df = config.current_data
    
    return {
        "total_signals": len(df),
        "avg_ai_score": df['ai_score'].mean(),
        "high_quality_count": len(df[df['ai_score'] >= 70]),
        "realtime_count": len(df[df.get('data_source', '') == 'REALTIME_HYBRID']),
        "top_5_symbols": df.head(5)['symbol'].tolist(),
        "data_sources": df.get('data_source', pd.Series()).value_counts().to_dict()
    }


# Export fonksiyonları (isteğe bağlı debug için)
__all__ = [
    'app', 'websocket_manager', 'get_system_info', 
    'validate_system_requirements', 'debug_websocket_status', 
    'debug_signal_quality'
]