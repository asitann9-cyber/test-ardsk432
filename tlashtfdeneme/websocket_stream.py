"""
📡 Binance WebSocket Real-Time Stream
TradingView benzeri anlık veri akışı - VPMV Sistemi için
🔥 SADECE 4 BİLEŞEN: Volume-Price-Momentum-Volatility
"""

import json
import threading
import logging
from typing import Callable, List, Dict, Optional
import time

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("⚠️ websocket-client kurulu değil: pip install websocket-client")

logger = logging.getLogger("crypto-analytics")


class BinanceWebSocketStream:
    """Binance Futures WebSocket real-time stream"""
    
    def __init__(self, symbols: List[str], interval: str = '15m'):
        """
        Args:
            symbols: ['BTCUSDT', 'ETHUSDT', ...]
            interval: '1m', '5m', '15m', '1h', '4h'
        """
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websocket-client kütüphanesi gerekli: pip install websocket-client")
        
        self.symbols = symbols
        self.interval = interval
        self.ws_apps = []
        self.on_kline_callback = None
        self.running = False
        self.threads = []
        
        # Binance limit: 200 stream/connection
        self.max_streams_per_connection = 200
        
    def subscribe(self, on_kline: Callable):
        """
        WebSocket'e abone ol
        
        Args:
            on_kline: Callback function(symbol, kline_data)
        """
        self.on_kline_callback = on_kline
        
        # Sembolleri 200'lük gruplara böl
        symbol_chunks = [
            self.symbols[i:i + self.max_streams_per_connection]
            for i in range(0, len(self.symbols), self.max_streams_per_connection)
        ]
        
        logger.info(f"📡 WebSocket başlatılıyor: {len(self.symbols)} sembol, {len(symbol_chunks)} bağlantı")
        
        # Her grup için ayrı WebSocket bağlantısı
        for chunk_idx, chunk in enumerate(symbol_chunks):
            thread = threading.Thread(
                target=self._start_websocket,
                args=(chunk, chunk_idx),
                daemon=True,
                name=f"WebSocket-{chunk_idx}"
            )
            thread.start()
            self.threads.append(thread)
            time.sleep(0.5)  # Bağlantılar arası kısa bekleme
            
        self.running = True
        logger.info(f"✅ WebSocket başlatıldı: {len(self.symbols)} sembol, {len(symbol_chunks)} bağlantı")
    
    def _start_websocket(self, symbols_chunk: List[str], chunk_idx: int):
        """WebSocket bağlantısını başlat"""
        
        # Stream URL'i oluştur
        streams = [f"{s.lower()}@kline_{self.interval}" for s in symbols_chunk]
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        logger.info(f"🔗 Bağlantı #{chunk_idx}: {len(symbols_chunk)} stream")
        
        ws_app = websocket.WebSocketApp(
            stream_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=lambda ws: self._on_open(ws, chunk_idx)
        )
        
        self.ws_apps.append(ws_app)
        
        # Sonsuz döngü - otomatik reconnect
        while self.running:
            try:
                ws_app.run_forever()
            except Exception as e:
                logger.error(f"WebSocket #{chunk_idx} hatası: {e}")
                if self.running:
                    logger.info(f"🔄 WebSocket #{chunk_idx} 5 saniye içinde yeniden bağlanacak...")
                    time.sleep(5)
    
    def _on_open(self, ws, chunk_idx: int):
        """Bağlantı açıldı"""
        logger.info(f"✅ WebSocket #{chunk_idx} bağlantısı açıldı")
    
    def _on_message(self, ws, message):
        """Mesaj geldi (yeni candle verisi)"""
        try:
            data = json.loads(message)
            
            if 'data' not in data:
                return
            
            kline_data = data['data']
            
            # stream field kontrolü
            if 'stream' not in kline_data:
                return
                
            symbol = kline_data.get('s')  # BTCUSDT
            kline = kline_data.get('k')   # Candle verisi
            
            if not symbol or not kline:
                return
            
            # Sadece kapanan candle'ları işle
            if kline.get('x'):  # x = is_closed
                if self.on_kline_callback:
                    self.on_kline_callback(symbol, kline)
                    
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse hatası: {e}")
        except Exception as e:
            logger.debug(f"Mesaj işleme hatası: {e}")
    
    def _on_error(self, ws, error):
        """Hata oluştu"""
        logger.error(f"❌ WebSocket hatası: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Bağlantı kapandı"""
        if self.running:
            logger.warning(f"⚠️ WebSocket kapandı: {close_status_code} - {close_msg}")
            logger.info("🔄 Yeniden bağlanılıyor...")
    
    def stop(self):
        """WebSocket'i durdur"""
        logger.info("🛑 WebSocket durduruluyor...")
        self.running = False
        
        # Tüm WebSocket bağlantılarını kapat
        for ws_app in self.ws_apps:
            try:
                ws_app.close()
            except:
                pass
        
        # Thread'lerin bitmesini bekle (max 5 saniye)
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.ws_apps.clear()
        self.threads.clear()
        
        logger.info("⛔ WebSocket durduruldu")
    
    def is_running(self) -> bool:
        """WebSocket aktif mi?"""
        return self.running
    
    def get_status(self) -> Dict:
        """WebSocket durumu"""
        return {
            'running': self.running,
            'symbols_count': len(self.symbols),
            'connections': len(self.ws_apps),
            'threads_alive': sum(1 for t in self.threads if t.is_alive())
        }


def convert_ws_kline_to_dict(kline: dict) -> Dict:
    """
    WebSocket kline verisini standart formata çevir
    
    Args:
        kline: WebSocket'ten gelen kline dict
        
    Returns:
        Standart format dict
    """
    try:
        return {
            'open_time': int(kline['t']),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'close_time': int(kline['T']),
            'quote_volume': float(kline['q']),
            'trades': int(kline['n']),
            'taker_buy_base': float(kline['V']),
            'taker_buy_quote': float(kline['Q'])
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Kline dönüştürme hatası: {e}")
        return {}


# Global WebSocket instance
_ws_stream: Optional[BinanceWebSocketStream] = None


def get_websocket_instance() -> Optional[BinanceWebSocketStream]:
    """Global WebSocket instance'ını al"""
    global _ws_stream
    return _ws_stream


def set_websocket_instance(ws_stream: Optional[BinanceWebSocketStream]):
    """Global WebSocket instance'ını set et"""
    global _ws_stream
    _ws_stream = ws_stream


def is_websocket_active() -> bool:
    """WebSocket aktif mi?"""
    global _ws_stream
    return _ws_stream is not None and _ws_stream.is_running()


def get_websocket_status() -> Dict:
    """WebSocket durumu"""
    global _ws_stream
    if _ws_stream is None:
        return {
            'running': False,
            'symbols_count': 0,
            'connections': 0,
            'threads_alive': 0
        }
    return _ws_stream.get_status()