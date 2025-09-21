"""
🔗 WebSocket Manager - Binance Real-time Bağlantısı
AI Top 10 sinyalleri + açık pozisyonlar için real-time fiyat takibi
🚀 ThreadedWebSocketManager ile performanslı çoklu stream yönetimi
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

try:
    from binance import ThreadedWebSocketManager
    from binance.exceptions import BinanceAPIException
    BINANCE_WS_AVAILABLE = True
except ImportError:
    BINANCE_WS_AVAILABLE = False
    print("⚠️ python-binance WebSocket desteği bulunamadı")

import config
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, ENVIRONMENT, LOCAL_TZ

logger = logging.getLogger("crypto-analytics")

# Global WebSocket durumu
websocket_manager: Optional["WebSocketManager"] = None
websocket_active: bool = False
realtime_prices: Dict[str, Dict] = {}  # {symbol: {price, change, volume, time}}
websocket_symbols: Set[str] = set()  # İzlenen semboller
last_update_time: Optional[datetime] = None


class WebSocketManager:
    """🔗 Binance WebSocket Manager - Real-time veri akışı"""
    
    def __init__(self):
        self.twm: Optional[ThreadedWebSocketManager] = None
        self.is_running: bool = False
        self.streams_active: List[str] = []
        self.connection_attempts: int = 0
        self.max_connection_attempts: int = 5
        self.reconnect_delay: int = 5
        self.last_heartbeat: Optional[datetime] = None
        
        # Callback fonksiyonları
        self.price_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # İstatistikler
        self.message_count: int = 0
        self.error_count: int = 0
        self.start_time: Optional[datetime] = None

    def add_price_callback(self, callback: Callable):
        """Fiyat güncellemesi için callback ekle"""
        self.price_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """Hata durumu için callback ekle"""
        self.error_callbacks.append(callback)

    def _handle_ticker_message(self, msg: Dict):
        """Ticker mesajını işle - real-time fiyat güncellemesi"""
        try:
            symbol = msg.get('s', '')  # BTCUSDT
            if not symbol:
                return
                
            # Fiyat verilerini parse et
            current_price = float(msg.get('c', 0))  # Current price
            price_change = float(msg.get('P', 0))   # Price change percent
            volume = float(msg.get('v', 0))         # Volume
            count = int(msg.get('n', 0))           # Trade count
            
            # Global real-time data güncelle
            global realtime_prices, last_update_time
            
            realtime_prices[symbol] = {
                'price': current_price,
                'change_percent': price_change,
                'volume_24h': volume,
                'trade_count': count,
                'timestamp': datetime.now(LOCAL_TZ),
                'raw_data': msg
            }
            
            last_update_time = datetime.now(LOCAL_TZ)
            self.message_count += 1
            self.last_heartbeat = datetime.now(LOCAL_TZ)
            
            # Callback'leri çağır
            for callback in self.price_callbacks:
                try:
                    callback(symbol, realtime_prices[symbol])
                except Exception as e:
                    logger.debug(f"Price callback hatası: {e}")
            
            # Debug log (her 100 mesajda bir)
            if self.message_count % 100 == 0:
                logger.debug(f"📡 WS Message #{self.message_count}: {symbol} = ${current_price:.6f} ({price_change:+.2f}%)")
                
        except Exception as e:
            logger.error(f"❌ Ticker mesaj işleme hatası: {e}")
            self.error_count += 1
            
            for callback in self.error_callbacks:
                try:
                    callback(f"Ticker parse error: {e}")
                except:
                    pass

    def _handle_error_message(self, msg: Dict):
        """Hata mesajını işle"""
        try:
            error_msg = f"WebSocket error: {msg}"
            logger.error(f"❌ {error_msg}")
            self.error_count += 1
            
            for callback in self.error_callbacks:
                try:
                    callback(error_msg)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Error mesaj işleme hatası: {e}")

    def start_streams(self, symbols: List[str]) -> bool:
        """WebSocket stream'lerini başlat"""
        global websocket_active, websocket_symbols
        
        if not BINANCE_WS_AVAILABLE:
            logger.error("❌ python-binance WebSocket desteği yok")
            return False
        
        if not symbols:
            logger.warning("⚠️ WebSocket için sembol listesi boş")
            return False
            
        if self.is_running:
            logger.warning("⚠️ WebSocket zaten çalışıyor")
            return True
            
        try:
            logger.info(f"🔗 WebSocket başlatılıyor: {len(symbols)} sembol")
            logger.info(f"🎯 İzlenecek semboller: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
            
            # ThreadedWebSocketManager oluştur
            if ENVIRONMENT == "testnet":
                # Testnet için special handling gerekebilir
                self.twm = ThreadedWebSocketManager(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=True
                )
                logger.info("🧪 Testnet WebSocket bağlantısı")
            else:
                self.twm = ThreadedWebSocketManager(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY
                )
                logger.info("🚀 Mainnet WebSocket bağlantısı")
            
            # WebSocket manager'ı başlat
            self.twm.start()
            
            # Multi-symbol ticker stream başlat
            streams = []
            for symbol in symbols:
                stream_name = f"{symbol.lower()}@ticker"
                streams.append(stream_name)
                
            # Birleşik ticker stream
            combined_stream = '/'.join(streams)
            
            # Stream'i başlat
            self.twm.start_multiplex_socket(
                callback=self._handle_ticker_message,
                streams=streams
            )
            
            self.streams_active = streams
            self.is_running = True
            self.start_time = datetime.now(LOCAL_TZ)
            self.connection_attempts += 1
            
            # Global state güncelle
            websocket_active = True
            websocket_symbols = set(symbols)
            
            logger.info(f"✅ WebSocket başarıyla başlatıldı: {len(streams)} stream aktif")
            logger.info(f"📊 Stream listesi: {streams[:3]}{'...' if len(streams) > 3 else ''}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket başlatma hatası: {e}")
            self.error_count += 1
            self.is_running = False
            
            if self.twm:
                try:
                    self.twm.stop()
                except:
                    pass
                self.twm = None
                
            return False

    def stop_streams(self):
        """WebSocket stream'lerini durdur"""
        global websocket_active, websocket_symbols, realtime_prices
        
        if not self.is_running:
            logger.info("💤 WebSocket zaten durdurulmuş")
            return
            
        try:
            logger.info("🛑 WebSocket durduruluyor...")
            
            if self.twm:
                self.twm.stop()
                self.twm = None
                
            # Global state temizle
            self.is_running = False
            self.streams_active = []
            websocket_active = False
            websocket_symbols.clear()
            
            # İstatistikleri logla
            if self.start_time:
                duration = (datetime.now(LOCAL_TZ) - self.start_time).total_seconds()
                logger.info(f"📊 WebSocket istatistikleri:")
                logger.info(f"   ⏱️ Çalışma süresi: {duration:.0f} saniye")
                logger.info(f"   📡 Toplam mesaj: {self.message_count}")
                logger.info(f"   ❌ Hata sayısı: {self.error_count}")
                logger.info(f"   📈 Mesaj/saniye: {self.message_count/duration:.1f}" if duration > 0 else "")
            
            logger.info("✅ WebSocket başarıyla durduruldu")
            
        except Exception as e:
            logger.error(f"❌ WebSocket durdurma hatası: {e}")

    def restart_streams(self, symbols: List[str]) -> bool:
        """WebSocket stream'lerini yeniden başlat"""
        logger.info("🔄 WebSocket yeniden başlatılıyor...")
        self.stop_streams()
        time.sleep(2)  # Kısa bekleme
        return self.start_streams(symbols)

    def is_healthy(self) -> bool:
        """WebSocket sağlık kontrolü"""
        if not self.is_running:
            return False
            
        # Son heartbeat kontrolü (60 saniye)
        if self.last_heartbeat:
            seconds_since_heartbeat = (datetime.now(LOCAL_TZ) - self.last_heartbeat).total_seconds()
            if seconds_since_heartbeat > 60:
                logger.warning(f"⚠️ WebSocket heartbeat gecikme: {seconds_since_heartbeat:.0f}s")
                return False
                
        return True

    def get_statistics(self) -> Dict:
        """WebSocket istatistiklerini al"""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(LOCAL_TZ) - self.start_time).total_seconds()
            
        return {
            'is_running': self.is_running,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'uptime_seconds': uptime,
            'symbols_count': len(websocket_symbols),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'connection_attempts': self.connection_attempts,
            'streams_active': len(self.streams_active)
        }


# Global fonksiyonlar
def start_websocket_streams(symbols: List[str]) -> bool:
    """WebSocket stream'lerini başlat (global interface)"""
    global websocket_manager
    
    if not websocket_manager:
        websocket_manager = WebSocketManager()
        
    return websocket_manager.start_streams(symbols)


def stop_websocket_streams():
    """WebSocket stream'lerini durdur (global interface)"""
    global websocket_manager
    
    if websocket_manager:
        websocket_manager.stop_streams()


def is_websocket_active() -> bool:
    """WebSocket aktif mi?"""
    return websocket_active


def get_websocket_status() -> Dict:
    """WebSocket durumunu al"""
    global websocket_manager, realtime_prices, last_update_time
    
    base_status = {
        'is_active': websocket_active,
        'symbols_count': len(websocket_symbols),
        'symbols': list(websocket_symbols),
        'prices_count': len(realtime_prices),
        'last_update': last_update_time.isoformat() if last_update_time else None
    }
    
    if websocket_manager:
        base_status.update(websocket_manager.get_statistics())
        base_status['is_healthy'] = websocket_manager.is_healthy()
    else:
        base_status['is_healthy'] = False
        
    return base_status


def get_realtime_price(symbol: str) -> Optional[Dict]:
    """Belirtilen sembol için real-time fiyat al"""
    global realtime_prices
    return realtime_prices.get(symbol)


def get_all_realtime_prices() -> Dict[str, Dict]:
    """Tüm real-time fiyatları al"""
    global realtime_prices
    return realtime_prices.copy()


def update_monitored_symbols(new_symbols: List[str]) -> bool:
    """İzlenen sembolleri güncelle"""
    global websocket_manager, websocket_symbols
    
    current_symbols = set(websocket_symbols)
    new_symbols_set = set(new_symbols)
    
    # Değişiklik var mı kontrol et
    if current_symbols == new_symbols_set:
        logger.debug("🔄 WebSocket sembolleri değişmedi")
        return True
        
    logger.info(f"🔄 WebSocket sembolleri güncelleniyor: {len(current_symbols)} → {len(new_symbols_set)}")
    
    # Eklenen/çıkarılan sembolleri logla
    added = new_symbols_set - current_symbols
    removed = current_symbols - new_symbols_set
    
    if added:
        logger.info(f"➕ Eklenen: {', '.join(added)}")
    if removed:
        logger.info(f"➖ Çıkarılan: {', '.join(removed)}")
    
    # WebSocket'i yeniden başlat
    if websocket_manager and websocket_active:
        return websocket_manager.restart_streams(new_symbols)
    else:
        return start_websocket_streams(new_symbols)


# Auto-reconnect thread
def _websocket_health_monitor():
    """WebSocket sağlık izleme thread'i"""
    global websocket_manager
    
    while True:
        try:
            time.sleep(30)  # 30 saniyede bir kontrol
            
            if websocket_active and websocket_manager:
                if not websocket_manager.is_healthy():
                    logger.warning("⚠️ WebSocket sağlıksız - yeniden bağlanmaya çalışılıyor...")
                    symbols_to_reconnect = list(websocket_symbols)
                    if symbols_to_reconnect:
                        websocket_manager.restart_streams(symbols_to_reconnect)
                        
        except Exception as e:
            logger.error(f"❌ WebSocket health monitor hatası: {e}")
            time.sleep(60)  # Hata durumunda 1 dakika bekle


# Health monitor thread'ini başlat
_health_thread = threading.Thread(target=_websocket_health_monitor, daemon=True)
_health_thread.start()
logger.debug("💓 WebSocket health monitor başlatıldı")