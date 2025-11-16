"""
WebSocket Manager
Binance Futures kline stream'ini yönetir ve kapanan mumları
dışarıya callback ile iletir.
"""

import logging
from typing import List, Optional, Callable, Any, Dict
from config import Config
from services.binance_service import BinanceService

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Binance Futures WebSocket kline stream yöneticisi.

    - Kalıcı tabloda takip edilen semboller için Binance WebSocket'i açar
    - Her yeni mum kapanışında callback tetikler
    - UI tarafına broadcast yapılabilir (clients listesi yönetilir)
    """

    # 🔥 FRONTEND WS CLIENT LISTESI
    clients = set()

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

    # ================================================================
    # 🔥 Binance WS callback → sadece kapanan mumları dışarıya iletir
    # ================================================================
    def _internal_ws_callback(self, symbol: str, close_price: float, is_kline_closed: bool) -> None:

        # sadece kapanan mum gelsin
        if not is_kline_closed:
            return

        logger.debug(f"[WS] Kapanan mum: {symbol} close={close_price}")

        if self.on_kline_closed:
            try:
                # CALLBACK → 3 parametre
                self.on_kline_closed(symbol, close_price, is_kline_closed)
            except Exception as e:
                logger.exception(f"on_kline_closed callback hatası ({symbol}): {e}")

    # ================================================================
    # 🔥 Binance WebSocket başlatıcı
    # ================================================================
    def start(self):
        if not self.symbols:
            logger.warning("WebSocketManager: Takip edilecek sembol yok, WS başlatılmadı.")
            return None

        logger.info(f"WebSocketManager: Semboller için WS başlatılıyor: {self.symbols} ({self.interval})")

        self.ws_thread = BinanceService.start_websocket(
            symbols=self.symbols,
            interval=self.interval,
            on_message_callback=self._internal_ws_callback,
        )
        return self.ws_thread

    # ================================================================
    # 🔥 Kullanım fonksiyonları
    # ================================================================
    def get_symbols(self) -> List[str]:
        return list(self.symbols)

    def add_symbol(self, symbol: str) -> None:
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"WebSocketManager: {symbol} listeye eklendi (yeniden başlatma gerekebilir).")

    # ================================================================
    # 🔥 Frontend WebSocket broadcast sistemi
    # ================================================================
    @staticmethod
    def add_client(ws):
        """Bir client bağlandığında kaydet."""
        WebSocketManager.clients.add(ws)
        logger.info(f"🔌 Yeni WS client bağlandı ({len(WebSocketManager.clients)} aktif)")

    @staticmethod
    def remove_client(ws):
        """Client ayrıldığında sil."""
        if ws in WebSocketManager.clients:
            WebSocketManager.clients.remove(ws)
            logger.info(f"❌ WS client ayrıldı ({len(WebSocketManager.clients)} aktif)")

    @staticmethod
    def broadcast(message: Dict[str, Any]):
        """Tüm bağlı client'lara JSON mesaj gönderir."""
        import json

        if not WebSocketManager.clients:
            return

        data = json.dumps(message)
        dead_clients = []

        for ws in WebSocketManager.clients:
            try:
                ws.send(data)
            except Exception as e:
                logger.error(f"❌ WS gönderim hatası: {e}")
                dead_clients.append(ws)

        # Bozuk client'ları sil
        for ws in dead_clients:
            WebSocketManager.remove_client(ws)
