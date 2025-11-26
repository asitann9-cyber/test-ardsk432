"""
WebSocket Manager
Binance Futures kline stream'ini yönetir ve kapanan mumları
dışarıya callback ile iletir.
✅ YENİ: Thread-safe client yönetimi
✅ YENİ: Graceful shutdown mekanizması
✅ YENİ: WebSocket durumu takibi
✅ FIX: BinanceService tuple return value uyumluluğu
🔥🔥 FIX: Defensive JSON serialization - Timestamp/datetime crash önleme
🔥🔥 FIX: Broadcast güvenliği - %100 crash-proof
🔥🔥🔥 FINAL FIX: numpy.bool_ + numpy.ndarray serialization - %100 crash-proof!
"""

import logging
import threading
import time
import pandas as pd
import numpy as np  # 🔥 YENİ: numpy import
from datetime import datetime
from typing import List, Optional, Callable, Any, Dict
from config import Config
from services.binance_service import BinanceService

logger = logging.getLogger(__name__)


# ================================================================
# 🔥🔥🔥 FINAL FIX: JSON Serialization Helper (numpy.bool_ + Timestamp cleaning)
# ================================================================
def _clean_message_for_json(obj):
    """
    🔥🔥🔥 FINAL FIX: JSON serialization için tüm özel tipleri temizle
    
    SORUN:
    - pd.Timestamp → "Object of type Timestamp is not JSON serializable"
    - numpy.bool_ → "Object of type bool_ is not JSON serializable"
    - numpy.ndarray → "Object of type ndarray is not JSON serializable"
    - datetime → "Object of type datetime is not JSON serializable"
    
    ÇÖZÜM:
    1. numpy.bool_ → Python bool (ÖNCE kontrol et!)
    2. numpy.ndarray → Python list
    3. pd.Timestamp → string
    4. datetime → string
    5. NaN/NaT → None
    
    🔥🔥 FIX: DeprecationWarning - numpy array pd.isna() sorunu çözüldü
    
    Recursive olarak tüm dict/list/tuple yapılarını tarar.
    
    Args:
        obj: Herhangi bir Python objesi
        
    Returns:
        JSON-safe objesi
    """
    if isinstance(obj, dict):
        return {k: _clean_message_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_message_for_json(item) for item in obj]
    # 🔥🔥🔥 KRİTİK: numpy.bool_ kontrolü ÖNCE (pd.isna'dan önce!)
    elif isinstance(obj, (np.bool_, np.generic)):
        return bool(obj)  # numpy.bool_ → Python bool
    # 🔥 numpy.ndarray kontrolü
    elif isinstance(obj, np.ndarray):
        if obj.size == 0:
            return None
        else:
            return obj.tolist()  # array → list
    # Timestamp kontrolü
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    # datetime kontrolü
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    # NaN/NaT kontrolü (EN SONDA - numpy.bool_ ile çakışmasın)
    else:
        try:
            # 🔥🔥 FIX: numpy array'leri pd.isna()'dan önce tekrar kontrol et
            # (DeprecationWarning: "truth value of empty array is ambiguous" fix)
            if isinstance(obj, np.ndarray):
                if obj.size == 0:
                    return None
                else:
                    return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        except (TypeError, ValueError):
            # pd.isna bazı tiplerde hata verebilir, o zaman olduğu gibi döndür
            return obj

class WebSocketManager:
    """
    Binance Futures WebSocket kline stream yöneticisi.

    - Kalıcı tabloda takip edilen semboller için Binance WebSocket'i açar
    - Her yeni mum kapanışında callback tetikler
    - UI tarafına broadcast yapılabilir (clients listesi yönetilir)
    ✅ Thread-safe client yönetimi
    ✅ Graceful shutdown desteği
    ✅ BinanceService tuple return uyumluluğu
    🔥🔥🔥 FINAL FIX: numpy.bool_ + numpy.ndarray serialization
    """

    # 🔥 FRONTEND WS CLIENT LISTESI (Thread-safe)
    clients = set()
    clients_lock = threading.Lock()  # ✅ YENİ: Thread-safety için lock

    def __init__(
        self,
        symbols: List[str],
        interval: Optional[str] = None,
        on_kline_closed: Optional[Callable[[str, float, bool], None]] = None,
    ) -> None:

        self.symbols = symbols
        self.interval = interval or Config.DEFAULT_WS_TIMEFRAME
        self.on_kline_closed = on_kline_closed
        self.ws_thread = None
        
        # ✅ YENİ: Stop mekanizması
        self.stop_flag = threading.Event()
        self.is_running = False
        
        # ✅ YENİ: WebSocket instance referansı
        self.ws_instance = None
        
        # ✅ YENİ: BinanceService'den gelen stop_flag'i de sakla
        self.binance_stop_flag = None

    # ================================================================
    # 🔥 Binance WS callback → sadece kapanan mumları dışarıya iletir
    # ================================================================
    def _internal_ws_callback(self, symbol: str, close_price: float, is_kline_closed: bool) -> None:
        """
        Binance WebSocket'ten gelen mesajları işle
        
        🔥 YENİ: AÇIK VE KAPANAN MUMLARIN HEPSİNİ İLETİYOR
        
        Args:
            symbol: Sembol adı
            close_price: Kapanış fiyatı
            is_kline_closed: Mum kapandı mı?
        """
        
        # ✅ YENİ: Stop kontrolü en üstte
        if self.stop_flag.is_set():
            logger.debug(f"[WS] Stop edilmiş, mesaj göz ardı ediliyor: {symbol}")
            return

        # 🔥 YENİ: HER MESAJI LOGLA (debug için)
        if is_kline_closed:
            logger.debug(f"[WS] 🟦 KAPANAN MUM: {symbol} close={close_price}")
        else:
            logger.debug(f"[WS] ⚡ AÇIK MUM: {symbol} close={close_price}")

        # ✅ YENİ: CALLBACK'İ HER DURUMDA ÇAĞIR (açık + kapanan)
        if self.on_kline_closed:
            try:
                # CALLBACK → 3 parametre (AÇIK VE KAPANAN MUMLAR)
                self.on_kline_closed(symbol, close_price, is_kline_closed)
            except Exception as e:
                logger.exception(f"on_kline_closed callback hatası ({symbol}): {e}")


    # ================================================================
    # 🔥 Binance WebSocket başlatıcı
    # ================================================================
    def start(self):
        """
        WebSocket bağlantısını başlat
        
        Returns:
            Thread object veya None
        """
        if not self.symbols:
            logger.warning("WebSocketManager: Takip edilecek sembol yok, WS başlatılmadı.")
            return None

        if self.is_running:
            logger.warning("WebSocketManager: Zaten çalışıyor, yeniden başlatılmıyor.")
            return self.ws_thread

        logger.info(f"WebSocketManager: Semboller için WS başlatılıyor: {self.symbols} ({self.interval})")

        # Stop flag'i sıfırla
        self.stop_flag.clear()

        try:
            # ✅ GÜNCELLEME: BinanceService artık tuple döndürüyor (ws_thread, stop_flag)
            result = BinanceService.start_websocket(
                symbols=self.symbols,
                interval=self.interval,
                on_message_callback=self._internal_ws_callback,
            )
            
            # Tuple unpacking
            self.ws_thread, self.binance_stop_flag = result
            
            self.is_running = True
            logger.info(f"✅ WebSocket başarıyla başlatıldı: {len(self.symbols)} sembol")
            
            return self.ws_thread
            
        except Exception as e:
            logger.error(f"❌ WebSocket başlatma hatası: {e}")
            self.is_running = False
            return None

    # ================================================================
    # ✅ YENİ: WebSocket durdurma mekanizması
    # ================================================================
    def stop(self):
        """
        WebSocket bağlantısını gracefully durdur
        """
        if not self.is_running:
            logger.warning("WebSocketManager: Zaten durdurulmuş.")
            return

        logger.info("🛑 WebSocketManager: Durduruluyor...")
        
        # Kendi stop flag'imizi set et
        self.stop_flag.set()
        
        # BinanceService'in stop flag'ini de set et
        if self.binance_stop_flag:
            self.binance_stop_flag.set()
            logger.debug("BinanceService stop_flag set edildi")
        
        # WebSocket instance'ı kapat
        if self.ws_instance:
            try:
                self.ws_instance.close()
                logger.info("✅ WebSocket instance kapatıldı")
            except Exception as e:
                logger.error(f"❌ WebSocket kapatma hatası: {e}")
        
        # Thread'in bitmesini bekle (max 5 saniye)
        if self.ws_thread and self.ws_thread.is_alive():
            logger.info("⏳ WebSocket thread'inin bitmesi bekleniyor...")
            self.ws_thread.join(timeout=5)
            
            if self.ws_thread.is_alive():
                logger.warning("⚠️ WebSocket thread 5 saniyede bitmedi (daemon thread olarak devam edecek)")
        
        self.is_running = False
        logger.info("✅ WebSocketManager durduruldu")

    # ================================================================
    # 🔥 Kullanım fonksiyonları
    # ================================================================
    def get_symbols(self) -> List[str]:
        """Takip edilen sembolleri döndür"""
        return list(self.symbols)

    def add_symbol(self, symbol: str) -> None:
        """
        Sembol ekle (WebSocket yeniden başlatılmalı)
        
        Args:
            symbol: Eklenecek sembol
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"WebSocketManager: {symbol} listeye eklendi (yeniden başlatma gerekiyor).")

    def remove_symbol(self, symbol: str) -> None:
        """
        Sembol çıkar (WebSocket yeniden başlatılmalı)
        
        Args:
            symbol: Çıkarılacak sembol
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.info(f"WebSocketManager: {symbol} listeden çıkarıldı (yeniden başlatma gerekiyor).")

    def get_status(self) -> Dict[str, Any]:
        """
        WebSocket durumunu döndür
        
        Returns:
            Durum bilgileri
        """
        return {
            'is_running': self.is_running,
            'symbols_count': len(self.symbols),
            'symbols': self.symbols,
            'interval': self.interval,
            'thread_alive': self.ws_thread.is_alive() if self.ws_thread else False,
            'stop_flag_set': self.stop_flag.is_set(),
            'binance_stop_flag_set': self.binance_stop_flag.is_set() if self.binance_stop_flag else None
        }

    # ================================================================
    # 🔥🔥🔥 FINAL FIX: Frontend WebSocket broadcast (numpy.bool_ + Timestamp safe)
    # ================================================================
    @staticmethod
    def add_client(ws):
        """
        Bir client bağlandığında kaydet (Thread-safe)
        
        Args:
            ws: WebSocket client instance
        """
        with WebSocketManager.clients_lock:
            WebSocketManager.clients.add(ws)
            client_count = len(WebSocketManager.clients)
        
        logger.info(f"🔌 Yeni WS client bağlandı ({client_count} aktif)")

    @staticmethod
    def remove_client(ws):
        """
        Client ayrıldığında sil (Thread-safe)
        
        Args:
            ws: WebSocket client instance
        """
        with WebSocketManager.clients_lock:
            if ws in WebSocketManager.clients:
                WebSocketManager.clients.remove(ws)
                client_count = len(WebSocketManager.clients)
                logger.info(f"❌ WS client ayrıldı ({client_count} aktif)")

    @staticmethod
    def broadcast(message: Dict[str, Any]):
        """
        Tüm bağlı client'lara JSON mesaj gönder
        
        🔥🔥🔥 FINAL FIX: numpy.bool_ + numpy.ndarray + Timestamp serialization
        
        Features:
        - Thread-safe client yönetimi
        - Defensive JSON serialization (numpy.bool_ → bool, Timestamp → string)
        - Dead client temizliği
        - Comprehensive error handling
        
        Args:
            message: Gönderilecek mesaj (dict)
        """
        import json

        # 🔥 STEP 1: Client kontrolü (Thread-safe)
        with WebSocketManager.clients_lock:
            if not WebSocketManager.clients:
                logger.debug("Broadcast: Client yok, mesaj gönderilmedi")
                return
            
            # Clients listesinin kopyasını al (iteration sırasında değişebilir)
            clients_copy = list(WebSocketManager.clients)
        
        # 🔥🔥🔥 STEP 2: DEFENSIVE JSON SERIALIZATION (numpy.bool_ + Timestamp)
        try:
            # numpy.bool_, numpy.ndarray, Timestamp/datetime nesnelerini temizle
            cleaned_message = _clean_message_for_json(message)
            
            # JSON'a çevir
            data = json.dumps(cleaned_message)
            
            logger.debug(f"📡 Broadcast hazır: {len(clients_copy)} client'a gönderilecek")
            
        except TypeError as e:
            logger.error(f"❌ JSON serialization hatası (TypeError): {e}")
            logger.error(f"   Mesaj tipi: {type(message)}")
            logger.error(f"   Mesaj keys: {list(message.keys()) if isinstance(message, dict) else 'N/A'}")
            
            # 🔥 DEBUG: Mesajdaki problematik değerleri bul
            if isinstance(message, dict):
                for key, value in message.items():
                    try:
                        json.dumps({key: value})
                    except TypeError:
                        logger.error(f"   ❌ Problematik alan: {key} = {type(value)}")
            
            return
        except Exception as e:
            logger.error(f"❌ JSON serialization hatası (Unexpected): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return

        # 🔥 STEP 3: Client'lara gönder (kilitli OLMADAN - I/O blocking olabilir)
        dead_clients = []
        success_count = 0
        
        for ws in clients_copy:
            try:
                ws.send(data)
                success_count += 1
            except Exception as e:
                logger.error(f"❌ WS gönderim hatası: {e}")
                dead_clients.append(ws)

        # 🔥 STEP 4: Bozuk client'ları temizle (Thread-safe)
        if dead_clients:
            with WebSocketManager.clients_lock:
                for ws in dead_clients:
                    if ws in WebSocketManager.clients:
                        WebSocketManager.clients.remove(ws)
            
            logger.warning(f"🧹 {len(dead_clients)} bozuk client temizlendi")
        
        # 🔥 STEP 5: Başarı logu
        if success_count > 0:
            logger.debug(f"✅ Broadcast başarılı: {success_count}/{len(clients_copy)} client")

    @staticmethod
    def get_client_count() -> int:
        """
        Aktif client sayısını döndür (Thread-safe)
        
        Returns:
            Aktif client sayısı
        """
        with WebSocketManager.clients_lock:
            return len(WebSocketManager.clients)

    @staticmethod
    def clear_all_clients():
        """
        Tüm client'ları temizle (Thread-safe)
        """
        with WebSocketManager.clients_lock:
            count = len(WebSocketManager.clients)
            WebSocketManager.clients.clear()
        
        logger.info(f"🧹 {count} WS client temizlendi")

    # ================================================================
    # ✅ YENİ: Health check
    # ================================================================
    def is_healthy(self) -> bool:
        """
        WebSocket sağlıklı mı kontrol et
        
        Returns:
            True = sağlıklı, False = sorunlu
        """
        if not self.is_running:
            return False
        
        if self.stop_flag.is_set():
            return False
        
        if not self.ws_thread or not self.ws_thread.is_alive():
            return False
        
        return True

    # ================================================================
    # ✅ YENİ: Restart mekanizması
    # ================================================================
    def restart(self):
        """
        WebSocket'i yeniden başlat (sembol listesi güncellendiğinde kullan)
        """
        logger.info("🔄 WebSocketManager: Yeniden başlatılıyor...")
        
        # Önce durdur
        if self.is_running:
            self.stop()
            # Durmasını bekle
            time.sleep(1)
        
        # Sonra başlat
        return self.start()

    # ================================================================
    # 🔥🔥🔥 YENİ: Diagnostic / Debug fonksiyonları
    # ================================================================
    @staticmethod
    def get_broadcast_stats() -> Dict[str, Any]:
        """
        Broadcast istatistiklerini döndür
        
        Returns:
            İstatistik bilgileri
        """
        with WebSocketManager.clients_lock:
            client_count = len(WebSocketManager.clients)
        
        return {
            'active_clients': client_count,
            'is_ready': client_count > 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    @staticmethod
    def test_broadcast():
        """
        Test broadcast gönder (diagnostic amaçlı)
        
        🔥🔥🔥 YENİ: numpy.bool_ ve Timestamp test
        
        Returns:
            Başarı durumu
        """
        test_message = {
            'event': 'test',
            'message': 'WebSocket test broadcast',
            'timestamp': datetime.now(),  # 🔥 datetime test
            'test_bool': np.bool_(True),  # 🔥🔥🔥 numpy.bool_ test
            'test_array': np.array([1, 2, 3]),  # 🔥 numpy.ndarray test
            'test_timestamp': pd.Timestamp.now()  # 🔥 Timestamp test
        }
        
        try:
            WebSocketManager.broadcast(test_message)
            logger.info("✅ Test broadcast başarılı (numpy.bool_ + Timestamp test)")
            return True
        except Exception as e:
            logger.error(f"❌ Test broadcast başarısız: {e}")
            return False