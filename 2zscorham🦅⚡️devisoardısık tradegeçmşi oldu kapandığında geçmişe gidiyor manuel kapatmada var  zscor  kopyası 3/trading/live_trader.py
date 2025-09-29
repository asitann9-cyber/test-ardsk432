"""
🤖 Live Trading Bot - Gerçek Binance API ile Trading
AI sinyallerini gerçek paraya çeviren bot sistemi (testnet/mainnet uyumlu)
🔥 YENİ: WebSocket ile ANLIK SL/TP takibi - Gerçek zamanlı pozisyon yönetimi
🔧 DÜZELTME: Config senkronizasyonu ile dashboard entegrasyonu
"""

import time
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Set
from decimal import Decimal, ROUND_DOWN
import math
import json

try:
    from binance.client import Client
    from binance.enums import *
    from binance.streams import ThreadedWebsocketManager  # 🔥 YENİ: WebSocket
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ python-binance kurulu değil: pip install python-binance")

import config
from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, ENVIRONMENT, LOCAL_TZ,
    INITIAL_CAPITAL, MAX_OPEN_POSITIONS, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    SCAN_INTERVAL, MIN_ORDER_SIZE, MAX_POSITION_SIZE_PCT,
    current_settings
)
from data.fetch_data import get_current_price
from data.database import log_trade_to_csv, log_capital_to_csv

logger = logging.getLogger("crypto-analytics")

# Global durum
binance_client: Optional["Client"] = None
live_trading_active: bool = False
live_trading_thread: Optional[threading.Thread] = None

# 🔥 YENİ: WebSocket Global Değişkenleri
websocket_manager: Optional["ThreadedWebsocketManager"] = None
websocket_active_symbols: Set[str] = set()
listen_key: Optional[str] = None
listen_key_refresh_thread: Optional[threading.Thread] = None


class WebSocketManager:
    """🔥 YENİ: WebSocket Yönetim Sınıfı - Anlık pozisyon takibi"""
    
    def __init__(self, client: Client):
        self.client = client
        self.twm: Optional[ThreadedWebsocketManager] = None
        self.listen_key: Optional[str] = None
        self.is_active: bool = False
        self.refresh_thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """WebSocket'i başlat - DÜZELTME: Tüm hatalar giderildi"""
        try:
            # 🔥 DÜZELTME 1: listen_key direkt string döner
            self.listen_key = self.client.futures_stream_get_listen_key()
            
            if not self.listen_key or not isinstance(self.listen_key, str):
                logger.error("❌ Listen key alınamadı veya geçersiz format")
                return False
            
            logger.info(f"🔑 Listen key alındı: {self.listen_key[:10]}...")
            
            # WebSocket Manager başlat
            self.twm = ThreadedWebsocketManager(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY,
                testnet=(ENVIRONMENT == "testnet")
            )
            self.twm.start()
            logger.info("🚀 WebSocket Manager başlatıldı")
            
            # 🔥 DÜZELTME 2: start_futures_user_socket kullan (listen_key parametresi YOK)
            self.twm.start_futures_user_socket(
                callback=self._handle_user_data
            )
            logger.info("📡 Futures User Data Stream'e bağlandı")
            
            self.is_active = True
            
            # Listen key yenileme thread'i başlat
            self.refresh_thread = threading.Thread(
                target=self._refresh_listen_key_loop,
                daemon=True
            )
            self.refresh_thread.start()
            logger.info("🔄 Listen key otomatik yenileme başlatıldı")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket başlatma hatası: {e}")
            import traceback
            logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
            return False
    
    def stop(self) -> None:
        """WebSocket'i durdur"""
        try:
            self.is_active = False
            
            if self.twm:
                self.twm.stop()
                logger.info("🛑 WebSocket Manager durduruldu")
            
            if self.listen_key:
                try:
                    self.client.futures_stream_close(listenKey=self.listen_key)
                    logger.info("🔒 Listen key kapatıldı")
                except Exception as e:
                    logger.debug(f"Listen key kapatma hatası (ignore): {e}")
            
            self.twm = None
            self.listen_key = None
            
        except Exception as e:
            logger.error(f"❌ WebSocket durdurma hatası: {e}")
    
    def _refresh_listen_key_loop(self) -> None:
        """Listen key'i otomatik yenile (her 30 dakikada)"""
        while self.is_active:
            try:
                time.sleep(30 * 60)  # 30 dakika bekle
                
                if not self.is_active:
                    break
                
                if self.listen_key:
                    self.client.futures_stream_keepalive(listenKey=self.listen_key)
                    logger.info("🔄 Listen key yenilendi")
                    
            except Exception as e:
                logger.error(f"❌ Listen key yenileme hatası: {e}")
                time.sleep(60)
    
    def _handle_user_data(self, msg: dict) -> None:
        """🔥 核心: WebSocket mesajlarını işle"""
        try:
            event_type = msg.get('e')
            
            if event_type == 'ORDER_TRADE_UPDATE':
                self._handle_order_update(msg)
            elif event_type == 'ACCOUNT_UPDATE':
                self._handle_account_update(msg)
            elif event_type == 'listenKeyExpired':
                logger.warning("⚠️ Listen key süresi doldu - yeniden başlatılıyor...")
                self.stop()
                time.sleep(1)
                self.start()
                
        except Exception as e:
            logger.error(f"❌ WebSocket mesaj işleme hatası: {e}")
    
    def _handle_order_update(self, msg: dict) -> None:
        """🎯 Emir güncellemelerini işle (SL/TP tetikleme)"""
        try:
            order_data = msg.get('o', {})
            symbol = order_data.get('s')
            order_status = order_data.get('X')  # Order status
            order_type = order_data.get('o')    # Order type
            order_side = order_data.get('S')    # Side
            
            # Sadece FILLED emirlerle ilgilen
            if order_status != 'FILLED':
                return
            
            # Config'deki pozisyonu kontrol et
            if symbol not in config.live_positions:
                return
            
            position = config.live_positions[symbol]
            
            # SL veya TP tetiklendi mi kontrol et
            if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                avg_price = float(order_data.get('ap', 0))
                executed_qty = float(order_data.get('z', 0))
                
                close_reason = "Stop Loss - Auto WebSocket" if order_type == 'STOP_MARKET' else "Take Profit - Auto WebSocket"
                
                logger.info(f"🔔 WEBSOCKET: {symbol} {close_reason} tetiklendi!")
                logger.info(f"   💰 Çıkış fiyatı: ${avg_price:.6f}")
                logger.info(f"   📊 Miktar: {executed_qty}")
                
                # Pozisyonu kapat
                self._close_position_from_websocket(
                    symbol=symbol,
                    position=position,
                    exit_price=avg_price,
                    close_reason=close_reason
                )
                
        except Exception as e:
            logger.error(f"❌ Order update işleme hatası: {e}")
    
    def _handle_account_update(self, msg: dict) -> None:
        """💰 Hesap güncellemelerini işle"""
        try:
            update_data = msg.get('a', {})
            positions = update_data.get('P', [])
            
            for pos in positions:
                symbol = pos.get('s')
                position_amt = float(pos.get('pa', 0))
                
                # Pozisyon kapandı mı kontrol et
                if abs(position_amt) == 0 and symbol in config.live_positions:
                    logger.info(f"🔔 WEBSOCKET: {symbol} pozisyonu kapandı (ACCOUNT_UPDATE)")
                    
                    # Eğer henüz işlenmediyse (ORDER_TRADE_UPDATE gelmemişse)
                    # backfill ile kapat
                    if symbol in config.live_positions:
                        old_pos = config.live_positions[symbol]
                        live_bot._backfill_closed_from_exchange(old_pos)
                        
                        if symbol in config.live_positions:
                            del config.live_positions[symbol]
                            sync_to_config()
                
        except Exception as e:
            logger.error(f"❌ Account update işleme hatası: {e}")
    
    def _close_position_from_websocket(self, symbol: str, position: dict, 
                                      exit_price: float, close_reason: str) -> None:
        """WebSocket'ten gelen kapanış sinyalini işle"""
        try:
            # P&L hesapla
            if position['side'] == 'LONG':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            logger.info(f"✅ WEBSOCKET KAPANIŞ: {symbol} {position['side']}")
            logger.info(f"   💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f}")
            logger.info(f"   💰 P&L: ${pnl:.4f}")
            
            # Trade kaydı oluştur
            trade_data = position.copy()
            trade_data.update({
                "exit_price": exit_price,
                "current_value": position["quantity"] * exit_price,
                "pnl": pnl,
                "close_reason": close_reason,
                "close_time": datetime.now(LOCAL_TZ),
            })
            
            # CSV'ye kaydet
            live_bot._log_trade_to_csv(trade_data, "CLOSED")
            
            # Açık emirleri temizle
            live_bot.cleanup_symbol_orders(symbol)
            
            # Config'den sil
            if symbol in config.live_positions:
                del config.live_positions[symbol]
                sync_to_config()
                logger.info(f"🧹 {symbol} config'ten silindi (WebSocket)")
            
        except Exception as e:
            logger.error(f"❌ WebSocket kapanış işleme hatası: {e}")


def stop_websocket():
    """WebSocket bağlantısını kapat"""
    global websocket_manager, websocket_active_symbols
    
    if websocket_manager:
        websocket_manager.stop()
        websocket_manager = None
    
    websocket_active_symbols.clear()
    logger.info("🛑 WebSocket kapatıldı")


def _sync_server_time(client: "Client", retries: int = 3, sleep_s: float = 0.2) -> None:
    """Binance Futures sunucu saatine göre timestamp offset ayarla."""
    import time as _t
    last_offset = 0
    for i in range(retries):
        try:
            srv = client.futures_time()["serverTime"]
            loc = int(_t.time() * 1000)
            last_offset = int(srv) - loc
            client.timestamp_offset = last_offset
            logger.info(f"⏱️ Time sync: offset={last_offset} ms (try {i+1})")
            _t.sleep(sleep_s)
        except Exception as e:
            logger.warning(f"⚠️ Time sync attempt {i+1} failed: {e}")
            _t.sleep(sleep_s)
def sync_to_config() -> None:
    """🔥 Config senkronizasyonu + backfill"""
    try:
        config.switch_to_live_mode()
        if getattr(live_bot, "is_connected", False):
            try:
                balance = live_bot.get_account_balance()
                config.update_live_capital(balance)
            except Exception:
                pass

        prev_positions = dict(config.live_positions)
        prev_symbols = set(prev_positions.keys())

        fresh_positions = get_current_live_positions() or {}
        new_symbols = set(fresh_positions.keys())

        disappeared = prev_symbols - new_symbols
        for sym in disappeared:
            old = prev_positions.get(sym, {}) or {}
            if not old:
                continue
            if live_bot._backfill_closed_from_exchange(old):
                logger.info(f"🔒 Backfill ile {sym} kapatma CSV'ye yazıldı")

        config.update_live_positions(fresh_positions)
        config.live_trading_active = live_trading_active
        logger.debug("🔄 Config senkronizasyonu tamamlandı")

    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")


def get_current_live_positions() -> Dict:
    """🔥 Binance'den açık pozisyonları al"""
    try:
        if not live_bot.client:
            return {}
            
        old_positions = config.live_positions.copy()
        binance_positions = live_bot.client.futures_position_information()
        
        live_positions = {}
        
        for pos in binance_positions:
            symbol = pos['symbol']
            position_amt = float(pos['positionAmt'])
            
            if abs(position_amt) > 0:
                entry_price = float(pos['entryPrice'])
                mark_price = float(pos['markPrice'])
                unrealized_pnl = float(pos['unRealizedProfit'])
                
                side = 'LONG' if position_amt > 0 else 'SHORT'
                quantity = abs(position_amt)
                
                if side == 'LONG':
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)
                
                signal_data = {'ai_score': 50}
                
                if config.current_data is not None and not config.current_data.empty:
                    symbol_rows = config.current_data[config.current_data['symbol'] == symbol]
                    if not symbol_rows.empty:
                        latest_signal = symbol_rows.iloc[0]
                        signal_data = {
                            'ai_score': latest_signal['ai_score'],
                            'run_type': latest_signal['run_type'],
                            'run_count': latest_signal['run_count'],
                            'run_perc': latest_signal['run_perc'],
                            'gauss_run': latest_signal['gauss_run'],
                            'vol_ratio': latest_signal.get('vol_ratio', 0),
                            'deviso_ratio': latest_signal.get('deviso_ratio', 0)
                        }
                
                old_pos = old_positions.get(symbol, {})
                sl_order_id = old_pos.get('sl_order_id')
                tp_order_id = old_pos.get('tp_order_id')
                main_order_id = old_pos.get('main_order_id')
                entry_time = old_pos.get('entry_time', datetime.now(LOCAL_TZ))
                
                position_data = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': mark_price,
                    'invested_amount': quantity * entry_price,
                    'current_value': quantity * mark_price,
                    'unrealized_pnl': unrealized_pnl,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': entry_time,
                    'auto_sltp': True,
                    'signal_data': signal_data,
                    'sl_order_id': sl_order_id,
                    'tp_order_id': tp_order_id,
                    'main_order_id': main_order_id,
                }
                
                live_positions[symbol] = position_data
        
        logger.debug(f"📊 Binance'den {len(live_positions)} açık pozisyon alındı")
        return live_positions
        
    except Exception as e:
        logger.error(f"❌ Binance pozisyon alma hatası: {e}")
        return {}


class LiveTradingBot:
    """🤖 Gerçek Binance Trading Bot Sınıfı - WebSocket destekli"""

    def __init__(self):
        self.client: Optional["Client"] = None
        self.is_connected: bool = False
        self.account_balance: float = 0.0
        self.tradable_cache: Set[str] = set()
        self.symbol_info_cache: Dict[str, Dict] = {}

    def _refresh_tradable_cache(self) -> None:
        """Testnet/Mainnet TRADABLE sembolleri keşfet"""
        try:
            tickers = self.client.futures_symbol_ticker()
            cache = {t["symbol"] for t in tickers if t.get("symbol", "").endswith("USDT")}

            try:
                info = self.client.futures_exchange_info()
                perp_usdt = {
                    s["symbol"] for s in info.get("symbols", [])
                    if s.get("quoteAsset") == "USDT"
                    and s.get("status") == "TRADING"
                    and s.get("contractType") == "PERPETUAL"
                }
                if perp_usdt:
                    cache = cache & perp_usdt
            except Exception as e2:
                logger.debug(f"exchangeInfo kesişimi atlandı: {e2}")

            self.tradable_cache = cache
            logger.info(f"🧭 TRADABLE semboller: {len(cache)} adet")
            if cache:
                logger.info("🧭 Örnek: " + ", ".join(list(cache)[:10]))
        except Exception as e:
            logger.warning(f"⚠️ Tradable sembol keşfi başarısız: {e}")

    def connect_to_binance(self) -> bool:
        """🔑 Binance API'ye bağlan + WebSocket başlat"""
        global binance_client, websocket_manager

        if not BINANCE_AVAILABLE:
            logger.error("❌ python-binance kütüphanesi yüklü değil")
            return False

        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.error("❌ API anahtarları .env dosyasında bulunamadı")
            return False

        try:
            if ENVIRONMENT == "testnet":
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=True,
                )
                self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
                self.client.FUTURES_DATA_URL = "https://testnet.binancefuture.com/futures/data"
                if hasattr(self.client, "futures_api_url"):
                    self.client.futures_api_url = self.client.FUTURES_URL
                if hasattr(self.client, "futures_data_api_url"):
                    self.client.futures_data_api_url = self.client.FUTURES_DATA_URL
                logger.info("🧪 Binance Futures Testnet için URL'ler override edildi")
            else:
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                )
                logger.info("🚀 Binance Mainnet'e bağlanılıyor")

            _sync_server_time(self.client)
            self.client.futures_ping()

            account_info = self.client.futures_account(recvWindow=60000)
            self.account_balance = float(account_info["totalWalletBalance"])
            logger.info(f"✅ API bağlantısı başarılı - Bakiye: ${self.account_balance:.2f}")

            self._refresh_tradable_cache()

            binance_client = self.client
            self.is_connected = True
            
            # 🔥 YENİ: WebSocket başlat
            websocket_manager = WebSocketManager(self.client)
            if websocket_manager.start():
                logger.info("✅ WebSocket başarıyla başlatıldı - ANLIK takip aktif")
            else:
                logger.warning("⚠️ WebSocket başlatılamadı - REST API ile devam edilecek")
            
            sync_to_config()
            
            return True

        except Exception as e:
            logger.error(f"❌ Binance bağlantı hatası: {e}")
            self.is_connected = False
            return False

    def get_account_balance(self) -> float:
        """💰 Hesap bakiyesini al"""
        try:
            if not self.client:
                return 0.0
            account_info = self.client.futures_account(recvWindow=60000)
            balance = float(account_info["totalWalletBalance"])
            self.account_balance = balance
            
            config.update_live_capital(balance)
            
            return balance
        except Exception as e:
            logger.error(f"❌ Bakiye alma hatası: {e}")
            return 0.0
        
    def _backfill_closed_from_exchange(self, old_pos: Dict) -> bool:
        """Kapanmış pozisyonu Binance fills'ten yakala ve CSV'ye yaz"""
        try:
            from datetime import datetime
            if not self.client:
                return False

            symbol = old_pos.get("symbol")
            if not symbol:
                return False

            side = (old_pos.get("side") or "").upper()
            qty = float(old_pos.get("quantity") or 0.0)
            entry_price = float(old_pos.get("entry_price") or 0.0)
            entry_time = old_pos.get("entry_time") or datetime.now(LOCAL_TZ)

            if qty <= 0.0 or entry_price <= 0.0:
                return False

            start_ms = int(entry_time.timestamp() * 1000)
            fills = self.client.futures_account_trades(
                symbol=symbol, startTime=start_ms, recvWindow=60000
            ) or []

            exit_side = "SELL" if side in ("LONG", "BUY") else "BUY"

            used_qty = 0.0
            wsum_px_qty = 0.0
            for f in fills:
                if (f.get("side") or "").upper() != exit_side:
                    continue
                fq = float(f.get("qty") or 0.0)
                fp = float(f.get("price") or 0.0)
                if fq <= 0.0 or fp <= 0.0:
                    continue

                take = min(fq, max(0.0, qty - used_qty))
                if take <= 0.0:
                    continue

                used_qty += take
                wsum_px_qty += take * fp

                if used_qty + 1e-12 >= qty:
                    break

            if used_qty <= 0.0:
                return False

            exit_price = wsum_px_qty / used_qty

            if side in ("LONG", "BUY"):
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty

            trade_data = dict(old_pos)
            trade_data.setdefault("signal_data", old_pos.get("signal_data") or {})
            trade_data.update({
                "exit_price": exit_price,
                "current_value": qty * exit_price,
                "pnl": pnl,
                "close_reason": trade_data.get("close_reason", "Auto - Backfill"),
                "close_time": datetime.now(LOCAL_TZ),
            })

            self._log_trade_to_csv(trade_data, "CLOSED")
            self.cleanup_symbol_orders(symbol)

            logger.info(f"🧾 Backfill CLOSED: {symbol} @ {exit_price:.6f}")
            return True

        except Exception as e:
            logger.error(f"❌ Backfill hata: {e}")
            return False
    def get_symbol_info(self, symbol: str) -> Dict:
        """📊 Sembol bilgilerini al (cache'li)"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        try:
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info["symbols"]:
                if s["symbol"] == symbol:
                    info = {
                        "symbol": symbol,
                        "status": s["status"],
                        "quantity_precision": s["quantityPrecision"],
                        "price_precision": s["pricePrecision"],
                        "lot_size": None,
                        "min_qty": None,
                        "min_notional": None,
                    }
                    
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            info["lot_size"] = float(f["stepSize"])
                            info["min_qty"] = float(f["minQty"])
                        elif f["filterType"] == "MIN_NOTIONAL":
                            info["min_notional"] = float(f.get("notional", 0.0))
                    
                    self.symbol_info_cache[symbol] = info
                    return info
            return {}
        except Exception as e:
            logger.error(f"❌ Sembol bilgisi alma hatası {symbol}: {e}")
            return {}

    def _is_tradable_symbol(self, symbol: str) -> bool:
        """🔍 Sembolün trade edilebilir olduğunu doğrula"""
        try:
            if self.tradable_cache and symbol not in self.tradable_cache:
                logger.debug(f"🔍 {symbol} tradable_cache'de yok")
                return False
            _ = self.client.futures_symbol_ticker(symbol=symbol, recvWindow=60000)
            return True
        except Exception as e:
            logger.debug(f"⛔ {symbol} tradable değil: {e}")
            return False

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """📏 Pozisyon büyüklüğünü hesapla"""
        try:
            max_position_value = self.account_balance * (MAX_POSITION_SIZE_PCT / 100)

            if max_position_value < MIN_ORDER_SIZE:
                logger.warning(f"⚠️ Yetersiz bakiye - Min: ${MIN_ORDER_SIZE}")
                return 0.0

            if ENVIRONMENT == "testnet":
                max_position_value *= 0.1
                logger.debug(f"🧪 Testnet: Pozisyon boyutu küçültüldü: ${max_position_value:.2f}")

            raw_qty = max_position_value / price

            info = self.get_symbol_info(symbol)
            lot_size = float(info.get("lot_size") or 0.0)
            min_qty = float(info.get("min_qty") or 0.0)
            min_notional = float(info.get("min_notional") or 0.0)
            quantity_precision = int(info.get("quantity_precision", 8))

            qty = round(raw_qty, quantity_precision)

            if lot_size > 0:
                qty_steps = qty / lot_size
                qty = math.floor(qty_steps) * lot_size
                qty = round(qty, quantity_precision)

            if min_qty > 0 and qty < min_qty:
                logger.warning(f"⚠️ {symbol} minimum quantity altında")
                return 0.0

            if min_notional > 0:
                notional = qty * price
                if notional < min_notional:
                    required_qty = min_notional / price
                    required_qty = round(required_qty, quantity_precision)
                    
                    if lot_size > 0:
                        qty_steps = required_qty / lot_size
                        required_qty = math.ceil(qty_steps) * lot_size
                        required_qty = round(required_qty, quantity_precision)
                    
                    if required_qty * price < min_notional:
                        logger.warning(f"⚠️ {symbol} notional karşılanamıyor")
                        return 0.0
                        
                    qty = required_qty

            logger.debug(f"🎯 {symbol} Final qty: {qty}")
            return max(qty, 0.0)

        except Exception as e:
            logger.error(f"❌ Pozisyon hesaplama hatası: {e}")
            return 0.0

    def _format_price(self, symbol: str, price: float) -> float:
        """🎯 Fiyatı sembol precision'ına göre formatla"""
        try:
            info = self.get_symbol_info(symbol)
            price_precision = int(info.get("price_precision", 8))
            return round(price, price_precision)
        except:
            return round(price, 8)

    def open_position(self, signal: Dict) -> bool:
        """🚀 Pozisyon aç + WebSocket ile anlık takip"""
        try:
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()

            current_positions = config.live_positions
            
            if symbol in current_positions:
                logger.warning(f"⚠️ {symbol} için zaten açık pozisyon var")
                return False

            if len(current_positions) >= MAX_OPEN_POSITIONS:
                logger.warning(f"⚠️ Maksimum pozisyon sayısına ulaşıldı")
                return False

            current_price = get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ {symbol} için fiyat alınamadı")
                return False

            quantity = self.calculate_position_size(symbol, current_price)
            if quantity <= 0:
                logger.error(f"❌ {symbol} için geçersiz pozisyon büyüklüğü")
                return False

            order_side = SIDE_BUY if side_txt == "LONG" else SIDE_SELL

            main_order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                recvWindow=60000,
            )

            time.sleep(3)
            
            order_check = self.client.futures_get_order(
                symbol=symbol, 
                orderId=main_order["orderId"], 
                recvWindow=60000
            )
            
            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0))
            
            logger.info(f"📋 {symbol} Emir Status: {order_status} | Executed: {executed_qty}")
            
            if order_status == "FILLED" and executed_qty > 0:
                avg_price = float(order_check.get("avgPrice") or current_price)
                
                if executed_qty != quantity:
                    logger.warning(f"⚠️ {symbol} Kısmi dolum: İstenen={quantity}, Gerçekleşen={executed_qty}")
                
                quantity = executed_qty

                if side_txt == "LONG":
                    stop_loss = self._format_price(symbol, avg_price * (1 - STOP_LOSS_PCT))
                    take_profit = self._format_price(symbol, avg_price * (1 + TAKE_PROFIT_PCT))
                    close_side = SIDE_SELL
                else:
                    stop_loss = self._format_price(symbol, avg_price * (1 + STOP_LOSS_PCT))
                    take_profit = self._format_price(symbol, avg_price * (1 - TAKE_PROFIT_PCT))
                    close_side = SIDE_BUY

                logger.info(f"✅ LIVE POZİSYON AÇILDI: {symbol} {side_txt} {executed_qty} @ ${avg_price:.6f}")
                logger.info(f"💰 Yatırılan: ${executed_qty * avg_price:.2f}")

                sl_order_id = None
                try:
                    sl_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=close_side,
                        type="STOP_MARKET",
                        quantity=executed_qty,
                        stopPrice=stop_loss,
                        timeInForce="GTC",
                        recvWindow=60000,
                    )
                    sl_order_id = sl_order["orderId"]
                    logger.info(f"🛑 Stop Loss emri: ${stop_loss:.6f} (ID: {sl_order_id})")
                except Exception as e:
                    logger.error(f"❌ Stop Loss emri hatası: {e}")

                tp_order_id = None
                try:
                    tp_order = self.client.futures_create_order(
                        symbol=symbol,
                        side=close_side,
                        type="TAKE_PROFIT_MARKET",
                        quantity=executed_qty,
                        stopPrice=take_profit,
                        timeInForce="GTC",
                        recvWindow=60000,
                    )
                    tp_order_id = tp_order["orderId"]
                    logger.info(f"🎯 Take Profit emri: ${take_profit:.6f} (ID: {tp_order_id})")
                except Exception as e:
                    logger.error(f"❌ Take Profit emri hatası: {e}")

                position_data = {
                    "symbol": symbol,
                    "side": side_txt,
                    "quantity": executed_qty,
                    "entry_price": avg_price,
                    "invested_amount": executed_qty * avg_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now(LOCAL_TZ),
                    "signal_data": signal,
                    "main_order_id": main_order["orderId"],
                    "sl_order_id": sl_order_id,
                    "tp_order_id": tp_order_id,
                    "auto_sltp": True,
                }

                config.live_positions[symbol] = position_data
                sync_to_config()
                
                # 🔥 YENİ: WebSocket pozisyonu takibe al
                global websocket_active_symbols
                websocket_active_symbols.add(symbol)
                logger.info(f"📡 {symbol} WebSocket ile ANLIK takibe alındı")
                
                if sl_order_id and tp_order_id:
                    logger.info(f"🤖 {symbol} otomatik SL/TP + WebSocket aktif")
                else:
                    logger.warning(f"⚠️ {symbol} SL/TP verilemedi")
                
                return True
            else:
                logger.error(f"❌ {symbol} market emir beklemede kaldı")
                return False

        except Exception as e:
            logger.error(f"❌ Pozisyon açma hatası {symbol}: {e}")
            return False

    def close_position(self, symbol: str, close_reason: str) -> bool:
        """🔒 Pozisyon kapat + WebSocket temizliği"""
        try:
            current_positions = config.live_positions

            if symbol not in current_positions:
                logger.warning(f"⚠️ {symbol} için açık pozisyon bulunamadı")
                return False

            position = current_positions[symbol]
            close_side = SIDE_SELL if position["side"] == "LONG" else SIDE_BUY

            close_order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=position["quantity"],
                recvWindow=60000,
            )

            logger.info(f"📋 {symbol} kapatma emri verildi - ID: {close_order['orderId']}")
            time.sleep(3)

            order_check = self.client.futures_get_order(
                symbol=symbol,
                orderId=close_order["orderId"],
                recvWindow=60000,
            )

            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0.0))

            logger.info(f"📋 {symbol} Kapatma Status: {order_status} | Executed: {executed_qty}")

            if order_status == "FILLED" and executed_qty > 0:
                avg_price_str = order_check.get("avgPrice")
                if avg_price_str in (None, "", "0"):
                    try:
                        exit_price = float(self.client.futures_mark_price(symbol=symbol).get("markPrice"))
                    except Exception:
                        exit_price = float(position["entry_price"])
                else:
                    exit_price = float(avg_price_str)

                if position["side"] == "LONG":
                    pnl = (exit_price - position["entry_price"]) * position["quantity"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["quantity"]

                logger.info(f"✅ LIVE POZİSYON KAPANDI: {symbol} {position['side']} | Sebep: {close_reason}")
                logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f} | P&L: ${pnl:.4f}")

                self.cleanup_symbol_orders(symbol)

                trade_data = position.copy()
                trade_data.update({
                    "exit_price": exit_price,
                    "current_value": position["quantity"] * exit_price,
                    "pnl": pnl,
                    "close_reason": close_reason,
                    "close_time": datetime.now(LOCAL_TZ),
                })

                self._log_trade_to_csv(trade_data, "CLOSED")

                # 🔥 YENİ: WebSocket'ten kaldır
                global websocket_active_symbols
                if symbol in websocket_active_symbols:
                    websocket_active_symbols.remove(symbol)
                    logger.info(f"📡 {symbol} WebSocket'ten kaldırıldı")

                del config.live_positions[symbol]
                sync_to_config()

                return True
            else:
                logger.error(f"❌ {symbol} kapatma emri beklemede kaldı")
                return False

        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma hatası {symbol}: {e}")
            return False

    def monitor_positions(self) -> None:
        """👀 Pozisyonları izle - WebSocket varsa minimal kontrol"""
        try:
            current_positions = config.live_positions
            
            if not current_positions:
                return

            # WebSocket aktifse sadece bilgi logla
            global websocket_manager
            if websocket_manager and websocket_manager.is_active:
                logger.debug(f"📡 WebSocket aktif - {len(current_positions)} pozisyon ANLIK takipte")
                return

            # WebSocket yoksa REST API ile kontrol
            logger.debug(f"👀 REST API ile {len(current_positions)} pozisyon izleniyor...")
            
            for symbol, position in list(current_positions.items()):
                if position.get("auto_sltp", False):
                    current_price = get_current_price(symbol)
                    if current_price:
                        entry_price = position["entry_price"]
                        if position["side"] == "LONG":
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        
                        logger.debug(f"🤖 {symbol}: {current_price:.6f} | PnL: {pnl_pct:+.2f}%")

        except Exception as e:
            logger.error(f"❌ Pozisyon izleme hatası: {e}")

    def check_filled_orders(self) -> None:
        """🔍 Emir kontrolü - WebSocket varsa atla"""
        try:
            # WebSocket aktifse bu fonksiyon gereksiz
            global websocket_manager
            if websocket_manager and websocket_manager.is_active:
                logger.debug("📡 WebSocket aktif - REST API emir kontrolü atlandı")
                return

            # WebSocket yoksa eski yöntemle kontrol et
            current_positions = config.live_positions
            symbols_to_remove = set()

            for symbol, position in list(current_positions.items()):
                if not position.get("auto_sltp", False):
                    continue

                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")

                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(
                            symbol=symbol, orderId=sl_order_id, recvWindow=60000
                        )
                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss tetiklendi (REST)")
                            self._handle_auto_close(symbol, "Stop Loss - Auto REST", sl_order)
                            symbols_to_remove.add(symbol)
                            continue
                    except Exception as e:
                        logger.error(f"❌ SL check error {symbol}: {e}")

                if tp_order_id:
                    try:
                        tp_order = self.client.futures_get_order(
                            symbol=symbol, orderId=tp_order_id, recvWindow=60000
                        )
                        if tp_order["status"] == "FILLED":
                            logger.info(f"🎯 {symbol} Take Profit tetiklendi (REST)")
                            self._handle_auto_close(symbol, "Take Profit - Auto REST", tp_order)
                            symbols_to_remove.add(symbol)
                            continue
                    except Exception as e:
                        logger.error(f"❌ TP check error {symbol}: {e}")

            for symbol in symbols_to_remove:
                if symbol in config.live_positions:
                    del config.live_positions[symbol]
                
                global websocket_active_symbols
                if symbol in websocket_active_symbols:
                    websocket_active_symbols.remove(symbol)

            if symbols_to_remove:
                sync_to_config()

        except Exception as e:
            logger.error(f"❌ Emir kontrolü hatası: {e}")
    def cleanup_symbol_orders(self, symbol: str) -> None:
        """🧹 Sembol için tüm açık emirleri temizle + WebSocket cleanup"""
        try:
            if not self.client:
                return

            open_orders = self.client.futures_get_open_orders(symbol=symbol, recvWindow=60000)

            if not open_orders:
                logger.debug(f"🧹 {symbol} için açık emir yok")
                return

            logger.info(f"🧹 {symbol} için {len(open_orders)} açık emir temizleniyor...")

            cancelled_count = 0
            failed_count = 0

            for order in open_orders:
                order_id = order["orderId"]
                order_type = order["type"]
                order_side = order["side"]

                try:
                    self.client.futures_cancel_order(
                        symbol=symbol,
                        orderId=order_id,
                        recvWindow=60000,
                    )
                    cancelled_count += 1
                    logger.info(f"🚫 {symbol} emir iptal: {order_type} {order_side} (ID: {order_id})")

                except Exception as e:
                    failed_count += 1
                    logger.warning(f"⚠️ {symbol} emir iptal hatası (ID: {order_id}): {e}")

            if cancelled_count > 0:
                logger.info(f"✅ {symbol} temizlik: {cancelled_count} iptal, {failed_count} başarısız")
            
            # 🔥 YENİ: WebSocket'ten kaldır
            global websocket_active_symbols
            if symbol in websocket_active_symbols:
                websocket_active_symbols.remove(symbol)
                logger.debug(f"📡 {symbol} WebSocket'ten kaldırıldı (cleanup)")

        except Exception as e:
            logger.error(f"❌ {symbol} emir temizlik hatası: {e}")

    def _handle_auto_close(self, symbol: str, close_reason: str, filled_order: dict) -> None:
        """🔄 Otomatik kapatma işle + WebSocket cleanup"""
        try:
            current_positions = config.live_positions
            if symbol not in current_positions:
                return

            position = current_positions[symbol]
            exit_price = float(filled_order.get("avgPrice", position["entry_price"]))

            pnl = (
                (exit_price - position["entry_price"]) * position["quantity"]
                if position["side"] == "LONG"
                else (position["entry_price"] - exit_price) * position["quantity"]
            )

            logger.info(f"✅ OTOMATIK KAPANIŞ: {symbol} {position['side']} | Sebep: {close_reason}")
            logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f} | P&L: ${pnl:.4f}")

            self.cleanup_symbol_orders(symbol)

            trade_data = position.copy()
            trade_data.update({
                "exit_price": exit_price,
                "current_value": position["quantity"] * exit_price,
                "pnl": pnl,
                "close_reason": close_reason,
                "close_time": datetime.now(LOCAL_TZ),
            })

            self._log_trade_to_csv(trade_data, "CLOSED")

            if symbol in config.live_positions:
                del config.live_positions[symbol]
                sync_to_config()
                logger.info(f"🧹 {symbol} config'ten silindi")

        except Exception as e:
            logger.error(f"❌ Otomatik kapatma işleme hatası {symbol}: {e}")

    def cancel_pending_orders(self) -> None:
        """🚫 Bekleyen emirleri temizle"""
        try:
            current_positions = config.live_positions
            for symbol, position in list(current_positions.items()):
                if position.get("auto_sltp", False):
                    continue
            logger.debug("🧹 Bekleyen emirler kontrol edildi")
        except Exception as e:
            logger.error(f"❌ Bekleyen emir temizleme hatası: {e}")

    def fill_empty_positions(self) -> None:
        """🎯 UI'deki filtrelenmiş en iyi sinyalleri al"""
        try:
            logger.info("🔄 fill_empty_positions başlatıldı")
        
            if not live_trading_active:
                logger.info("❌ Live trading aktif değil - çıkılıyor")
                return

            current_positions = config.live_positions
            current_position_count = len(current_positions)
            logger.info(f"📊 Mevcut pozisyon: {current_position_count}/{MAX_OPEN_POSITIONS}")
        
            if current_position_count >= MAX_OPEN_POSITIONS:
                logger.info("✅ Tüm pozisyon slotları dolu")
                return

            needed_slots = MAX_OPEN_POSITIONS - current_position_count
            logger.info(f"🎯 Gereken slot sayısı: {needed_slots}")

            if config.current_data is None or config.current_data.empty:
                logger.warning("❌ config.current_data boş - UI'den veri bekleniyor")
                return

            logger.info(f"✅ UI'den veri alındı: {len(config.current_data)} satır")

            df = config.current_data.copy()
            exclude_symbols = set(current_positions.keys())
            if exclude_symbols:
                before_exclude = len(df)
                df = df[~df["symbol"].isin(exclude_symbols)]
                logger.info(f"🚫 Açık pozisyonlar hariç: {len(df)}/{before_exclude} sinyal kaldı")

            if df.empty:
                logger.info("ℹ️ Uygun yeni sembol yok")
                return

            logger.info("📊 UI'deki sıralama korunuyor")
            top_signals = df.head(needed_slots)
        
            logger.info(f"🏆 UI'nin gösterdiği ilk {len(top_signals)} sinyal seçildi:")
            for i, (_, signal) in enumerate(top_signals.iterrows(), 1):
                logger.info(f"   🥇 #{i}: {signal['symbol']} | AI={signal['ai_score']:.0f}%")

            opened = 0
            for idx, (_, signal) in enumerate(top_signals.iterrows(), 1):
                if opened >= needed_slots:
                    break
                
                symbol = signal["symbol"]
                if not self._is_tradable_symbol(symbol):
                    logger.warning(f"⛔ {symbol} tradable değil - atlanıyor")
                    continue
            
                success = self.open_position(signal.to_dict())
                if success:
                    opened += 1
                    logger.info(f"🚀 {symbol} pozisyonu açıldı! (UI sırası: #{idx})")
                    time.sleep(1)
                else:
                    logger.error(f"❌ {symbol} pozisyonu açılamadı!")

            if opened > 0:
                logger.info(f"🎊 BAŞARILI: {opened} yeni pozisyon açıldı")
                logger.info(f"📊 Yeni durum: {len(current_positions) + opened}/{MAX_OPEN_POSITIONS} pozisyon")
            else:
                logger.warning("😔 Hiçbir pozisyon açılamadı")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")
            import traceback
            logger.error(f"📋 Detaylı hata: {traceback.format_exc()}")

    def _log_trade_to_csv(self, trade_data: Dict, status: str) -> None:
        """📝 Trade'i CSV'ye kaydet"""
        try:
            csv_data = {
                "timestamp": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": trade_data["symbol"],
                "side": trade_data["side"],
                "quantity": trade_data["quantity"],
                "entry_price": trade_data["entry_price"],
                "exit_price": trade_data.get("exit_price", 0),
                "invested_amount": trade_data["invested_amount"],
                "current_value": trade_data.get("current_value", 0),
                "pnl": trade_data.get("pnl", 0),
                "commission": 0.0,
                "ai_score": trade_data["signal_data"]["ai_score"],
                "run_type": trade_data["signal_data"]["run_type"],
                "run_count": trade_data["signal_data"]["run_count"],
                "run_perc": trade_data["signal_data"]["run_perc"],
                "gauss_run": trade_data["signal_data"]["gauss_run"],
                "vol_ratio": trade_data["signal_data"].get("vol_ratio", 0),
                "deviso_ratio": trade_data["signal_data"].get("deviso_ratio", 0),
                "stop_loss": trade_data["stop_loss"],
                "take_profit": trade_data["take_profit"],
                "close_reason": trade_data.get("close_reason", ""),
                "status": status,
            }
            log_trade_to_csv(csv_data)
        except Exception as e:
            logger.error(f"❌ CSV log hatası: {e}")


# Global bot instance
live_bot = LiveTradingBot()


def live_trading_loop() -> None:
    """🔄 Ana live trading döngüsü - WebSocket ile optimize edilmiş"""
    global live_trading_active

    logger.info("🤖 Live Trading döngüsü başlatıldı")
    loop_count = 0

    while live_trading_active:
        try:
            loop_count += 1
            loop_start = time.time()

            current_positions = config.live_positions
            logger.info(f"🔄 Live döngü #{loop_count} - Pozisyon: {len(current_positions)}/{MAX_OPEN_POSITIONS}")

            balance = live_bot.get_account_balance()
            logger.info(f"💰 Mevcut bakiye: ${balance:.2f}")

            sync_to_config()
            live_bot.cancel_pending_orders()
            
            # WebSocket kontrolü
            global websocket_manager
            if websocket_manager and websocket_manager.is_active:
                logger.info("📡 WebSocket AKTIF - Pozisyonlar anlık takipte")
            else:
                logger.warning("⚠️ WebSocket pasif - REST API ile kontrol")
                live_bot.check_filled_orders()

            live_bot.fill_empty_positions()
            live_bot.monitor_positions()
            log_capital_to_csv()

            loop_time = time.time() - loop_start
            logger.info(f"⏱️ Döngü #{loop_count}: {loop_time:.2f}s tamamlandı")

            if current_positions:
                positions_summary = ", ".join(current_positions.keys())
                auto_count = sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
                logger.info(f"🔥 Açık pozisyonlar: {positions_summary} (Auto SL/TP: {auto_count}/{len(current_positions)})")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Live trading döngüsü hatası: {e}")
            time.sleep(30)

    logger.info("ℹ️ Live trading döngüsü sonlandırıldı")


def start_live_trading() -> bool:
    """🚀 Live trading'i başlat + WebSocket"""
    global live_trading_active, live_trading_thread

    if live_trading_thread is not None and live_trading_thread.is_alive():
        logger.warning("⚠️ Live trading zaten aktif")
        return False
    if live_trading_active:
        logger.warning("⚠️ Live trading zaten aktif")
        return False

    if not live_bot.connect_to_binance():
        logger.error("❌ Binance API bağlantısı başarısız")
        return False

    logger.info("🚀 Live Trading başlatılıyor...")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"💰 Başlangıç bakiyesi: ${live_bot.account_balance:.2f}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info(f"🛑 Stop Loss: %{STOP_LOSS_PCT * 100}")
    logger.info(f"🎯 Take Profit: %{TAKE_PROFIT_PCT * 100}")
    logger.info(f"📡 WebSocket: ANLIK SL/TP takibi aktif")

    config.switch_to_live_mode()
    live_trading_active = True

    live_trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
    live_trading_thread.start()

    logger.info("✅ Live Trading başlatıldı")
    return True


def stop_live_trading() -> None:
    """🛑 Live trading'i durdur + WebSocket kapat"""
    global live_trading_active

    if not live_trading_active:
        logger.info("💤 Live trading zaten durdurulmuş")
        return

    logger.info("🛑 Live Trading durduruluyor...")
    live_trading_active = False

    # WebSocket'i kapat
    stop_websocket()

    current_positions = config.live_positions.copy()
    
    if current_positions:
        logger.info(f"📚 {len(current_positions)} açık pozisyon toplu kapatılıyor...")
        successful_closes = 0
        failed_closes = 0
        
        for symbol in current_positions.keys():
            try:
                logger.info(f"🔒 {symbol} pozisyonu kapatılıyor...")
                success = live_bot.close_position(symbol, "Trading Stopped")
                if success:
                    successful_closes += 1
                    logger.info(f"✅ {symbol} başarıyla kapatıldı")
                else:
                    failed_closes += 1
                    logger.error(f"❌ {symbol} kapatılamadı")
                time.sleep(1)
            except Exception as e:
                failed_closes += 1
                logger.error(f"❌ {symbol} kapatma hatası: {e}")
        
        logger.info(f"📊 Kapatma özeti: ✅{successful_closes} başarılı, ❌{failed_closes} başarısız")
    
    try:
        config.reset_live_trading()
        logger.info("🔄 Live trading config sıfırlandı")
    except Exception as e:
        logger.error(f"❌ Config sıfırlama hatası: {e}")
    
    logger.info("✅ Live Trading durduruldu")


def is_live_trading_active() -> bool:
    """📊 Live trading aktif mi?"""
    return live_trading_active


def get_live_trading_status() -> Dict:
    """📊 Live trading durum bilgilerini al"""
    current_positions = config.live_positions
    auto_sltp_count = sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    
    # WebSocket durumu
    global websocket_manager
    websocket_status = websocket_manager.is_active if websocket_manager else False
    
    return {
        "is_active": live_trading_active,
        "api_connected": live_bot.is_connected,
        "balance": live_bot.account_balance,
        "environment": ENVIRONMENT,
        "open_positions": len(current_positions),
        "max_positions": MAX_OPEN_POSITIONS,
        "auto_sltp_positions": auto_sltp_count,
        "auto_sltp_enabled": True,
        "websocket_active": websocket_status,  # 🔥 YENİ
    }


def get_live_bot_status_for_symbol(symbol: str) -> str:
    """App.py callback'i için sembol durumu al"""
    try:
        current_positions = config.live_positions
        
        if symbol in current_positions:
            pos = current_positions[symbol]
            
            # WebSocket aktif mi kontrol et
            global websocket_active_symbols
            if symbol in websocket_active_symbols:
                return "✅📡"  # Açık pozisyon + WebSocket aktif
            elif pos.get('auto_sltp', False):
                return "✅🤖"  # Açık pozisyon + otomatik SL/TP
            else:
                return "✅📱"  # Açık pozisyon + manuel
        else:
            return "⭐"  # Beklemede
    except:
        return "❓"


def get_auto_sltp_count() -> int:
    """App.py callback'i için otomatik SL/TP sayısı"""
    try:
        current_positions = config.live_positions
        return sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    except:
        return 0