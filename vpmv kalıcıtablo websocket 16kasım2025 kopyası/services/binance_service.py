"""
Binance API servisleri
Binance vadeli işlemler API'si ile veri çekme ve sembol yönetimi
✅ YENİ: Multi-stream format desteği düzeltildi
✅ YENİ: WebSocket heartbeat/ping-pong
✅ YENİ: Graceful shutdown mekanizması
✅ YENİ: WebSocket instance yönetimi
"""

import requests
import pandas as pd
import logging
import threading
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class BinanceService:
    """Binance API işlemleri için service sınıfı"""
    
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    # Timeframe limitleri - SUPERTREND İÇİN ARTTIRILDI
    TIMEFRAME_LIMITS = {
        '1m': 500, '5m': 500, '15m': 500, '30m': 500,
        '1h': 500, '2h': 500, '4h': 500, '1d': 500
    }
    
    # ✅ YENİ: Aktif WebSocket instance'larını takip et
    _ws_instances = []
    _ws_instances_lock = threading.Lock()
    
    @classmethod
    def fetch_symbols(cls) -> List[str]:
        """
        Binance USDT vadeli işlem sembollerini çek
        
        Returns:
            List[str]: Aktif USDT vadeli işlem sembolleri listesi
        """
        try:
            url = f"{cls.BASE_URL}/exchangeInfo"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info.get('quoteAsset') == 'USDT' and 
                    symbol_info.get('status') == 'TRADING' and
                    symbol_info.get('contractType') == 'PERPETUAL'):
                    symbols.append(symbol_info['symbol'])
            
            logger.info(f"Binance: {len(symbols)} USDT vadeli sembolü bulundu")
            return sorted(symbols)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API bağlantı hatası: {e}")
            raise
        except Exception as e:
            logger.error(f"Binance sembol listesi çekme hatası: {e}")
            raise
    
    @classmethod
    def fetch_klines_data(cls, symbol: str, timeframe: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Belirli sembol ve timeframe için OHLCV verilerini çek
        
        Args:
            symbol (str): Sembol adı (ör: BTCUSDT)
            timeframe (str): Zaman dilimi (ör: 4h)
            limit (Optional[int]): Veri limiti, None ise otomatik belirlenir
            
        Returns:
            Optional[pd.DataFrame]: OHLCV verileri içeren DataFrame
        """
        try:
            if limit is None:
                limit = cls.TIMEFRAME_LIMITS.get(timeframe, 500)
            
            # Maksimum 1000'e sınırla (Binance limiti)
            limit = min(limit, 1000)
            
            url = f"{cls.BASE_URL}/klines"
            
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"Boş veri: {symbol} {timeframe}")
                return None
            
            # DataFrame oluştur
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Timestamp'i datetime'a çevir (UTC -> Istanbul)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Istanbul')
            
            # Numerik kolonları float'a çevir
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Tarih sırasına göre sırala
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            
            logger.debug(f"Veri çekildi: {symbol} {timeframe} - {len(df)} mum (limit: {limit})")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API bağlantı hatası {symbol}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Binance veri çekme hatası {symbol}: {e}")
            return None
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        Sembolün geçerli olup olmadığını kontrol et
        
        Args:
            symbol (str): Kontrol edilecek sembol
            
        Returns:
            bool: Sembol geçerli ise True
        """
        try:
            all_symbols = cls.fetch_symbols()
            return symbol in all_symbols
        except Exception:
            return False
    
    @classmethod
    def get_current_price(cls, symbol: str) -> Optional[float]:
        """
        Sembolün güncel fiyatını getir
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Optional[float]: Güncel fiyat
        """
        try:
            url = f"{cls.BASE_URL}/ticker/price"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return float(data['price'])
            
        except Exception as e:
            logger.debug(f"Fiyat çekme hatası {symbol}: {e}")
            return None
    
    @classmethod
    def get_market_info(cls, symbol: str) -> Dict[str, Any]:
        """
        Sembol hakkında pazar bilgilerini getir
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Dict[str, Any]: Pazar bilgileri
        """
        try:
            url = f"{cls.BASE_URL}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price_change': float(data['priceChange']),
                'price_change_percent': float(data['priceChangePercent']),
                'last_price': float(data['lastPrice']),
                'volume': float(data['volume']),
                'high_price': float(data['highPrice']),
                'low_price': float(data['lowPrice'])
            }
            
        except Exception as e:
            logger.debug(f"Pazar bilgisi çekme hatası {symbol}: {e}")
            return {}
    
    @classmethod
    def get_bulk_prices(cls, symbols: List[str]) -> Dict[str, float]:
        """
        Birden fazla sembolün güncel fiyatlarını toplu olarak getir
        
        Args:
            symbols (List[str]): Sembol listesi
            
        Returns:
            Dict[str, float]: Sembol -> fiyat mapping
        """
        try:
            url = f"{cls.BASE_URL}/ticker/price"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Sadece istenen sembollerin fiyatlarını döndür
            price_dict = {}
            for item in data:
                symbol = item['symbol']
                if symbol in symbols:
                    price_dict[symbol] = float(item['price'])
            
            logger.debug(f"Toplu fiyat çekimi: {len(price_dict)} sembol")
            return price_dict
            
        except Exception as e:
            logger.debug(f"Toplu fiyat çekme hatası: {e}")
            return {}
    
    @classmethod
    def get_symbol_precision(cls, symbol: str) -> Dict[str, int]:
        """
        Sembolün fiyat ve miktar hassasiyetini getir
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Dict[str, int]: Hassasiyet bilgileri
        """
        try:
            url = f"{cls.BASE_URL}/exchangeInfo"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == symbol:
                    price_precision = symbol_info.get('pricePrecision', 2)
                    quantity_precision = symbol_info.get('quantityPrecision', 3)
                    
                    return {
                        'price_precision': price_precision,
                        'quantity_precision': quantity_precision
                    }
            
            # Varsayılan değerler
            return {
                'price_precision': 2,
                'quantity_precision': 3
            }
            
        except Exception as e:
            logger.debug(f"Hassasiyet bilgisi çekme hatası {symbol}: {e}")
            return {
                'price_precision': 2,
                'quantity_precision': 3
            }
        
    # ==========================
    #    WEBSOCKET MODULE
    # ==========================
    @classmethod
    def create_ws_stream_url(cls, symbols: List[str], interval: str = "1m") -> str:
        """
        Çoklu sembol WebSocket stream URL'si oluşturur.
        Ör: wss://fstream.binance.com/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
        
        Args:
            symbols: Sembol listesi
            interval: Zaman dilimi (1m, 5m, 15m, vs.)
            
        Returns:
            WebSocket URL
        """
        from config import Config
        streams = "/".join([f"{symbol.lower()}@kline_{interval}" for symbol in symbols])
        return f"{Config.BINANCE_FUTURES_WS_MULTI_STREAM}?streams={streams}"
    
    @classmethod
    def start_websocket(cls, symbols: List[str], interval: str = "1m", on_message_callback=None):
        """
        WebSocket bağlantısını başlatır.
        Kalıcı tablodaki semboller için canlı fiyat güncellemesi sağlar.
        
        ✅ YENİ: Multi-stream format düzeltmesi
        ✅ YENİ: Heartbeat/ping-pong desteği
        ✅ YENİ: Stop mekanizması
        
        Args:
            symbols: İzlenecek sembol listesi
            interval: Zaman dilimi
            on_message_callback: Mesaj geldiğinde çağrılacak fonksiyon
            
        Returns:
            Thread ve stop_flag tuple
        """
        import websocket
        import json
        import threading

        ws_url = cls.create_ws_stream_url(symbols, interval)
        
        # ✅ YENİ: Stop flag
        stop_flag = threading.Event()
        ws_app = None  # WebSocketApp instance referansı

        # ✅ DÜZELTME: Multi-stream format kontrolü
        def on_message(ws, message):
            try:
                # Stop edilmişse işleme alma
                if stop_flag.is_set():
                    return
                
                data = json.loads(message)
                
                # ✅ YENİ: Multi-stream ve single-stream formatlarını destekle
                kline_data = None
                
                if "stream" in data and "data" in data:
                    # Multi-stream format: {"stream":"btcusdt@kline_1m","data":{...}}
                    kline_data = data["data"]
                elif "e" in data and data.get("e") == "kline":
                    # Single-stream format: {"e":"kline","s":"BTCUSDT",...}
                    kline_data = data
                else:
                    # Bilinmeyen format
                    logger.debug(f"Bilinmeyen WS mesaj formatı: {list(data.keys())}")
                    return
                
                # Kline verisi var mı kontrol et
                kline = kline_data.get("k")
                if not kline:
                    logger.debug(f"Kline verisi bulunamadı: {kline_data}")
                    return
                
                # Veriyi parse et
                symbol = kline["s"]
                close_price = float(kline["c"])
                is_kline_closed = kline["x"]
                
                # Callback'i çağır
                if on_message_callback:
                    on_message_callback(symbol, close_price, is_kline_closed)
                    
            except json.JSONDecodeError as e:
                logger.error(f"WebSocket JSON parse hatası: {e}")
            except KeyError as e:
                logger.error(f"WebSocket veri format hatası: {e}, data: {message[:200]}")
            except Exception as e:
                logger.error(f"WebSocket mesaj işleme hatası: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket hatası: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket bağlantısı kapandı (code: {close_status_code}, msg: {close_msg})")
            stop_flag.set()

        # ✅ YENİ: Ping/Pong handler'ları
        def on_ping(ws, message):
            logger.debug("WebSocket PING alındı")

        def on_pong(ws, message):
            logger.debug("WebSocket PONG alındı")

        def on_open(ws):
            logger.info(f"✅ WebSocket bağlantısı açıldı: {len(symbols)} sembol ({interval})")

        def run_ws():
            nonlocal ws_app
            
            try:
                ws_app = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_ping=on_ping,
                    on_pong=on_pong,
                    on_open=on_open,
                )
                
                # ✅ YENİ: Heartbeat ile çalıştır (20 saniyede bir ping)
                ws_app.run_forever(
                    ping_interval=20,  # 20 saniyede bir ping gönder
                    ping_timeout=10    # 10 saniye içinde pong gelmezse timeout
                )
                
            except Exception as e:
                logger.error(f"WebSocket run_forever hatası: {e}")
            finally:
                stop_flag.set()
                logger.info("WebSocket thread sonlandı")

        ws_thread = threading.Thread(target=run_ws, daemon=True, name=f"WS-{symbols[0] if symbols else 'unknown'}")
        ws_thread.start()

        # ✅ YENİ: Instance'ı kaydet
        with cls._ws_instances_lock:
            cls._ws_instances.append({
                'thread': ws_thread,
                'stop_flag': stop_flag,
                'symbols': symbols,
                'interval': interval,
                'ws_app': ws_app,
                'url': ws_url
            })

        logger.info(f"🚀 WebSocket thread başlatıldı: {len(symbols)} sembol ({interval})")
        
        # Thread ve stop_flag'i döndür
        return ws_thread, stop_flag

    # ✅ YENİ: WebSocket yönetim fonksiyonları
    @classmethod
    def stop_websocket(cls, ws_thread: threading.Thread, stop_flag: threading.Event) -> bool:
        """
        Belirli bir WebSocket bağlantısını durdur
        
        Args:
            ws_thread: WebSocket thread'i
            stop_flag: Stop flag
            
        Returns:
            Başarılı ise True
        """
        try:
            logger.info("🛑 WebSocket durduruluyor...")
            
            # Stop flag'i set et
            stop_flag.set()
            
            # Thread'in bitmesini bekle (max 5 saniye)
            if ws_thread and ws_thread.is_alive():
                ws_thread.join(timeout=5)
                
                if ws_thread.is_alive():
                    logger.warning("⚠️ WebSocket thread 5 saniyede bitmedi (daemon olarak devam edecek)")
                    return False
            
            logger.info("✅ WebSocket başarıyla durduruldu")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket durdurma hatası: {e}")
            return False

    @classmethod
    def stop_all_websockets(cls):
        """
        Tüm aktif WebSocket bağlantılarını durdur
        """
        with cls._ws_instances_lock:
            instance_count = len(cls._ws_instances)
            
            if instance_count == 0:
                logger.info("Durduralacak aktif WebSocket yok")
                return
            
            logger.info(f"🛑 {instance_count} WebSocket bağlantısı durduruluyor...")
            
            for instance in cls._ws_instances:
                try:
                    # Stop flag'i set et
                    instance['stop_flag'].set()
                    
                    # WebSocketApp'i kapat
                    if instance.get('ws_app'):
                        instance['ws_app'].close()
                    
                except Exception as e:
                    logger.error(f"WebSocket kapatma hatası: {e}")
            
            # Listeyi temizle
            cls._ws_instances.clear()
            
            logger.info(f"✅ {instance_count} WebSocket bağlantısı durduruldu")

    @classmethod
    def get_websocket_status(cls) -> Dict[str, Any]:
        """
        Aktif WebSocket bağlantılarının durumunu döndür
        
        Returns:
            Durum bilgileri
        """
        with cls._ws_instances_lock:
            active_count = 0
            total_symbols = 0
            
            for instance in cls._ws_instances:
                if instance['thread'].is_alive() and not instance['stop_flag'].is_set():
                    active_count += 1
                    total_symbols += len(instance['symbols'])
            
            return {
                'total_connections': len(cls._ws_instances),
                'active_connections': active_count,
                'total_symbols': total_symbols,
                'instances': [
                    {
                        'symbols_count': len(inst['symbols']),
                        'interval': inst['interval'],
                        'is_alive': inst['thread'].is_alive(),
                        'is_stopped': inst['stop_flag'].is_set(),
                        'symbols_preview': inst['symbols'][:3]  # İlk 3 sembol
                    }
                    for inst in cls._ws_instances
                ]
            }

    @classmethod
    def cleanup_dead_websockets(cls):
        """
        Ölü WebSocket instance'larını temizle
        """
        with cls._ws_instances_lock:
            initial_count = len(cls._ws_instances)
            
            # Sadece canlı olanları tut
            cls._ws_instances = [
                inst for inst in cls._ws_instances
                if inst['thread'].is_alive() and not inst['stop_flag'].is_set()
            ]
            
            cleaned_count = initial_count - len(cls._ws_instances)
            
            if cleaned_count > 0:
                logger.info(f"🧹 {cleaned_count} ölü WebSocket instance temizlendi")
            
            return cleaned_count