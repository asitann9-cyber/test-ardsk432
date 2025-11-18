"""
Binance API servisleri
Binance vadeli işlemler API'si ile veri çekme ve sembol yönetimi
✅ YENİ: DETAYLI DEBUG LOGLARI - Her mesaj görünür
✅ YENİ: Multi-stream format desteği düzeltildi
✅ YENİ: WebSocket heartbeat/ping-pong
✅ YENİ: Graceful shutdown mekanizması
✅ YENİ: WebSocket instance yönetimi
"""

import requests
import pandas as pd
import logging
import threading
import json
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
    
    # ✅ YENİ: Mesaj sayacı (debug için)
    _message_counter = 0
    _message_counter_lock = threading.Lock()
    
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
        WebSocket stream URL'si oluşturur.
        
        🔥 YENİ: Tek sembol = SINGLE-STREAM (her saniye güncelleme)
                 Çoklu sembol = MULTI-STREAM
        
        Args:
            symbols: Sembol listesi
            interval: Zaman dilimi (1m, 5m, 15m, vs.)
            
        Returns:
            WebSocket URL
        """
        from config import Config
        
        # 🔥 DETAYLI LOG
        logger.info(f"=" * 80)
        logger.info(f"🔗 WebSocket URL OLUŞTURULUYOR:")
        logger.info(f"   📍 Sembol Sayısı: {len(symbols)}")
        logger.info(f"   📍 Semboller: {symbols}")
        logger.info(f"   ⏰ Interval: {interval}")
        
        # 🔥 YENİ: TEK SEMBOL = SINGLE-STREAM (DAHA SIK GÜNCELLEME!)
        if len(symbols) == 1:
            symbol = symbols[0].lower()
            url = f"{Config.BINANCE_FUTURES_WS_SINGLE_STREAM}/{symbol}@kline_{interval}"
            logger.info(f"   🔷 Format: SINGLE-STREAM (daha sık güncelleme)")
            logger.info(f"   🌐 Final URL: {url}")
        else:
            # Çoklu sembol = multi-stream
            streams = "/".join([f"{symbol.lower()}@kline_{interval}" for symbol in symbols])
            url = f"{Config.BINANCE_FUTURES_WS_MULTI_STREAM}?streams={streams}"
            logger.info(f"   🔷 Format: MULTI-STREAM")
            logger.info(f"   🌐 Final URL: {url}")
        
        logger.info(f"=" * 80)
        
        return url

    
    @classmethod
    def start_websocket(cls, symbols: List[str], interval: str = "1m", on_message_callback=None):
        """
        WebSocket bağlantısını başlatır.
        
        ✅ YENİ: DETAYLI DEBUG LOGLARI
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

        ws_url = cls.create_ws_stream_url(symbols, interval)
        
        # ✅ YENİ: Stop flag
        stop_flag = threading.Event()
        ws_app = None  # WebSocketApp instance referansı

        # 🔥 DETAYLI on_message callback
        def on_message(ws, message):
            try:
                # Stop edilmişse işleme alma
                if stop_flag.is_set():
                    return
                
                # 🔥 Mesaj sayacını artır
                with cls._message_counter_lock:
                    cls._message_counter += 1
                    msg_num = cls._message_counter
                
                # JSON parse
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON Parse HATASI: {str(e)}")
                    return
                
                # Format tespiti
                kline_data = None
                
                if "stream" in data and "data" in data:
                    # Multi-stream format
                    kline_data = data["data"]
                elif "e" in data and data.get("e") == "kline":
                    # Single-stream format
                    kline_data = data
                else:
                    logger.warning(f"Bilinmeyen format! Keys: {list(data.keys())}")
                    return
                
                # Kline verisi kontrol
                kline = kline_data.get("k") if kline_data else None
                if not kline:
                    logger.error("Kline verisi bulunamadı!")
                    return
                
                # Parse edilmiş değerler
                symbol = kline["s"]
                close_price = float(kline["c"])
                is_kline_closed = kline["x"]  # ⚡ ÖNEMLİ: True = kapanan, False = açık
                
                # 🔥 DETAYLI LOG
                logger.info(f"╔{'=' * 78}╗")
                logger.info(f"║ 🔵 BINANCE WS MSG #{msg_num:<40} ║")
                logger.info(f"║   Symbol: {symbol:<65}║")
                logger.info(f"║   Price: {close_price:<66}║")
                logger.info(f"║   Is Closed: {'KAPANAN MUM 🟦' if is_kline_closed else 'AÇIK MUM ⚡':<59}║")
                logger.info(f"╚{'=' * 78}╝")
                
                # Callback çağır - 3 PARAMETRE İLE
                if on_message_callback:
                    try:
                        on_message_callback(symbol, close_price, is_kline_closed)
                        logger.info(f"✅ Callback başarılı")
                    except Exception as cb_error:
                        logger.error(f"❌ Callback HATASI: {str(cb_error)}")
                        import traceback
                        logger.error(traceback.format_exc())
                    
            except Exception as e:
                logger.error(f"❌ WEBSOCKET MESAJ IŞLEME HATASI: {e}")
                import traceback
                logger.error(traceback.format_exc())


        def on_error(ws, error):
            logger.error(f"╔{'=' * 78}╗")
            logger.error(f"║ ❌ WEBSOCKET ERROR{' ' * 54}║")
            logger.error(f"║ {str(error):<76}║")
            logger.error(f"╚{'=' * 78}╝")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"╔{'=' * 78}╗")
            logger.warning(f"║ 🔴 WEBSOCKET KAPANDI{' ' * 52}║")
            logger.warning(f"║ Code: {close_status_code:<69}║")
            logger.warning(f"║ Message: {str(close_msg):<66}║")
            logger.warning(f"╚{'=' * 78}╝")
            stop_flag.set()

        def on_ping(ws, message):
            logger.debug(f"🏓 WebSocket PING alındı")

        def on_pong(ws, message):
            logger.debug(f"🏓 WebSocket PONG alındı")

        def on_open(ws):
            logger.info(f"╔{'=' * 78}╗")
            logger.info(f"║ 🟢 WEBSOCKET BAĞLANTISI AÇILDI!{' ' * 41}║")
            logger.info(f"║ Sembol Sayısı: {len(symbols):<59}║")
            logger.info(f"║ Interval     : {interval:<59}║")
            logger.info(f"║ URL          : {ws_url[:60]:<59}║")
            logger.info(f"╚{'=' * 78}╝")

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
                
                logger.info("🚀 WebSocket run_forever() başlatılıyor...")
                
                # ✅ YENİ: Heartbeat ile çalıştır
                ws_app.run_forever(
                    ping_interval=20,  # 20 saniyede bir ping
                    ping_timeout=10    # 10 saniye pong timeout
                )
                
            except Exception as e:
                logger.error(f"❌ WebSocket run_forever hatası: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                stop_flag.set()
                logger.info("🛑 WebSocket thread sonlandı")

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

        logger.info(f"✅ WebSocket thread başlatıldı: {len(symbols)} sembol ({interval})")
        
        return ws_thread, stop_flag

    # ✅ Diğer WebSocket yönetim fonksiyonları (değişiklik yok)
    @classmethod
    def stop_websocket(cls, ws_thread: threading.Thread, stop_flag: threading.Event) -> bool:
        """Belirli bir WebSocket bağlantısını durdur"""
        try:
            logger.info("🛑 WebSocket durduruluyor...")
            stop_flag.set()
            
            if ws_thread and ws_thread.is_alive():
                ws_thread.join(timeout=5)
                
                if ws_thread.is_alive():
                    logger.warning("⚠️ WebSocket thread 5 saniyede bitmedi")
                    return False
            
            logger.info("✅ WebSocket başarıyla durduruldu")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket durdurma hatası: {e}")
            return False

    @classmethod
    def stop_all_websockets(cls):
        """Tüm aktif WebSocket bağlantılarını durdur"""
        with cls._ws_instances_lock:
            instance_count = len(cls._ws_instances)
            
            if instance_count == 0:
                logger.info("Durduralacak aktif WebSocket yok")
                return
            
            logger.info(f"🛑 {instance_count} WebSocket bağlantısı durduruluyor...")
            
            for instance in cls._ws_instances:
                try:
                    instance['stop_flag'].set()
                    if instance.get('ws_app'):
                        instance['ws_app'].close()
                except Exception as e:
                    logger.error(f"WebSocket kapatma hatası: {e}")
            
            cls._ws_instances.clear()
            logger.info(f"✅ {instance_count} WebSocket bağlantısı durduruldu")

    @classmethod
    def get_websocket_status(cls) -> Dict[str, Any]:
        """Aktif WebSocket bağlantılarının durumunu döndür"""
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
                'total_messages_received': cls._message_counter,
                'instances': [
                    {
                        'symbols_count': len(inst['symbols']),
                        'interval': inst['interval'],
                        'is_alive': inst['thread'].is_alive(),
                        'is_stopped': inst['stop_flag'].is_set(),
                        'symbols_preview': inst['symbols'][:3]
                    }
                    for inst in cls._ws_instances
                ]
            }

    @classmethod
    def cleanup_dead_websockets(cls):
        """Ölü WebSocket instance'larını temizle"""
        with cls._ws_instances_lock:
            initial_count = len(cls._ws_instances)
            
            cls._ws_instances = [
                inst for inst in cls._ws_instances
                if inst['thread'].is_alive() and not inst['stop_flag'].is_set()
            ]
            
            cleaned_count = initial_count - len(cls._ws_instances)
            
            if cleaned_count > 0:
                logger.info(f"🧹 {cleaned_count} ölü WebSocket instance temizlendi")
            
            return cleaned_count
