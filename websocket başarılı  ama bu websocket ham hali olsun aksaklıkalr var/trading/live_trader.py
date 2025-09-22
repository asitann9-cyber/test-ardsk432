"""
🤖 Live Trading Bot - Hibrit WebSocket/REST Sistemi
Sadece açık pozisyonlar için WebSocket, diğerleri için REST API
🔧 Event loop çökme sorunu çözüldü - Stabil sistem
"""

import time
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Set
import math

try:
    from binance.client import Client
    from binance import ThreadedWebsocketManager
    from binance.enums import *
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ python-binance kurulu değil: pip install python-binance")

import config
from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, ENVIRONMENT, LOCAL_TZ,
    MAX_OPEN_POSITIONS, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    SCAN_INTERVAL, MIN_ORDER_SIZE, MAX_POSITION_SIZE_PCT
)
from data.fetch_data import get_current_price
from data.database import log_trade_to_csv, log_capital_to_csv

logger = logging.getLogger("crypto-analytics")

# Global değişkenler
binance_client: Optional["Client"] = None
live_trading_active: bool = False
live_trading_thread: Optional[threading.Thread] = None

# WebSocket değişkenleri - SADECE AÇIK POZİSYONLAR İÇİN
websocket_manager: Optional[ThreadedWebsocketManager] = None
websocket_active_symbols: Set[str] = set()  # Sadece live pozisyonlar

# Thread güvenliği
_cleanup_lock = threading.Lock()


def _sync_server_time(client: "Client", retries: int = 3) -> bool:
    """Binance sunucu saati senkronizasyonu"""
    import time as _t
    for i in range(retries):
        try:
            srv = client.futures_time()["serverTime"]
            loc = int(_t.time() * 1000)
            offset = int(srv) - loc
            client.timestamp_offset = offset
            logger.info(f"⏱️ Time sync: offset={offset}ms")
            _t.sleep(0.2)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Time sync hatası: {e}")
            _t.sleep(0.5)
    return False


def sync_to_config():
    """Config senkronizasyonu"""
    try:
        config.switch_to_live_mode()
        
        if live_bot.is_connected:
            balance = live_bot.get_account_balance()
            config.update_live_capital(balance)
        
        live_positions = get_current_live_positions()
        config.update_live_positions(live_positions)
        config.live_trading_active = live_trading_active
        
        logger.debug("🔄 Config senkronizasyonu tamamlandı")
        
    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")


def get_current_live_positions() -> Dict:
    """Binance'den mevcut pozisyonları al"""
    try:
        if not live_bot.client:
            return {}
            
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
                
                # AI skorunu tablodan al
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
                    'entry_time': datetime.now(LOCAL_TZ),
                    'auto_sltp': True,
                    'signal_data': signal_data
                }
                
                live_positions[symbol] = position_data
        
        logger.debug(f"📊 Binance'den {len(live_positions)} pozisyon alındı")
        return live_positions
        
    except Exception as e:
        logger.error(f"❌ Binance pozisyon alma hatası: {e}")
        return {}


def handle_websocket_message(msg):
    """WebSocket mesaj handler - SADECE AÇIK POZİSYONLAR İÇİN"""
    try:
        msg_type = msg.get('e', 'unknown')
        
        if msg_type == 'ORDER_TRADE_UPDATE':
            order_data = msg.get('o', {})
            symbol = order_data.get('s')
            order_type = order_data.get('o')
            order_status = order_data.get('X')
            order_id = order_data.get('i')
            
            logger.debug(f"📨 WebSocket: {symbol} {order_type} {order_status}")
            
            # Sadece SL/TP tetiklemelerini işle ve sadece açık pozisyonlar için
            if (order_status == 'FILLED' and 
                order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and
                symbol in websocket_active_symbols and
                symbol in config.live_positions):
                
                logger.info(f"🎯 SL/TP TETİKLENDİ (WebSocket): {symbol} {order_type}")
                
                close_reason = "Stop Loss - WebSocket" if order_type == 'STOP_MARKET' else "Take Profit - WebSocket"
                
                order_details = {
                    'orderId': order_id,
                    'avgPrice': order_data.get('ap', order_data.get('L', 0)),
                    'executedQty': order_data.get('z', 0),
                    'status': 'FILLED'
                }
                
                live_bot._handle_auto_close_websocket(symbol, close_reason, order_details)
            
        elif msg_type == 'ACCOUNT_UPDATE':
            logger.debug("📊 WebSocket hesap güncellendi")
            
    except Exception as e:
        logger.error(f"❌ WebSocket mesaj işleme hatası: {e}")


def setup_smart_websocket():
    """AKILLI WebSocket: Sadece açık pozisyonlar için"""
    global websocket_manager, websocket_active_symbols
    
    try:
        # Mevcut açık pozisyonları al
        current_live_positions = set(config.live_positions.keys())
        
        # WebSocket'e ihtiyaç var mı?
        if not current_live_positions:
            # Açık pozisyon yoksa WebSocket'i durdur
            if websocket_manager:
                stop_websocket()
            logger.debug("📱 Açık pozisyon yok - WebSocket kapalı, sadece REST aktif")
            return False
        
        # Semboller değişti mi?
        if current_live_positions == websocket_active_symbols and websocket_manager:
            logger.debug("🔄 WebSocket sembolleri değişmedi")
            return True
        
        # Eski WebSocket'i durdur
        if websocket_manager:
            stop_websocket()
            time.sleep(1)
        
        # Yeni WebSocket başlat
        websocket_manager = ThreadedWebsocketManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY,
            testnet=(ENVIRONMENT == "testnet")
        )
        
        websocket_manager.start()
        stream_name = websocket_manager.start_futures_user_socket(callback=handle_websocket_message)
        
        websocket_active_symbols = current_live_positions
        
        logger.info(f"✅ WebSocket aktif - {len(websocket_active_symbols)} açık pozisyon izleniyor")
        logger.debug(f"🎯 WebSocket izlenen: {', '.join(websocket_active_symbols)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket kurulum hatası: {e}")
        return False


def stop_websocket():
    """WebSocket güvenli kapatma"""
    global websocket_manager, websocket_active_symbols
    
    with _cleanup_lock:
        try:
            if websocket_manager:
                websocket_manager.stop()
                time.sleep(1)
                websocket_manager = None
            websocket_active_symbols.clear()
            logger.info("🛑 WebSocket durduruldu")
        except Exception as e:
            logger.error(f"❌ WebSocket durdurma hatası: {e}")
            websocket_manager = None
            websocket_active_symbols.clear()


class LiveTradingBot:
    """Live Trading Bot Sınıfı"""

    def __init__(self):
        self.client: Optional["Client"] = None
        self.is_connected: bool = False
        self.account_balance: float = 0.0
        self.symbol_info_cache: Dict[str, Dict] = {}

    def connect_to_binance(self) -> bool:
        """Binance API bağlantısı"""
        global binance_client

        if not BINANCE_AVAILABLE:
            logger.error("❌ python-binance kütüphanesi yüklü değil")
            return False

        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.error("❌ API anahtarları .env dosyasında bulunamadı")
            return False

        try:
            if self.client:
                self.client = None
            
            if ENVIRONMENT == "testnet":
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=True,
                )
                self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
                self.client.FUTURES_DATA_URL = "https://testnet.binancefuture.com/futures/data"
                logger.info("🧪 Testnet bağlantısı")
            else:
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                )
                logger.info("🚀 Mainnet bağlantısı")

            # Timestamp sync
            if not _sync_server_time(self.client):
                logger.warning("⚠️ Timestamp sync başarısız")

            self.client.futures_ping()

            account_info = self.client.futures_account(recvWindow=60000)
            self.account_balance = float(account_info["totalWalletBalance"])
            logger.info(f"✅ API bağlantısı başarılı - Bakiye: ${self.account_balance:.2f}")

            binance_client = self.client
            self.is_connected = True
            sync_to_config()
            
            return True

        except Exception as e:
            logger.error(f"❌ Binance bağlantı hatası: {e}")
            self.is_connected = False
            self.client = None
            return False

    def get_account_balance(self) -> float:
        """Hesap bakiyesi"""
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

    def get_symbol_info(self, symbol: str) -> Dict:
        """Sembol bilgileri (cache'li)"""
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
            logger.error(f"❌ Sembol bilgisi hatası {symbol}: {e}")
            return {}

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Pozisyon büyüklüğü hesaplama"""
        try:
            max_position_value = self.account_balance * (MAX_POSITION_SIZE_PCT / 100)

            if max_position_value < MIN_ORDER_SIZE:
                logger.warning(f"⚠️ Yetersiz bakiye")
                return 0.0

            if ENVIRONMENT == "testnet":
                max_position_value *= 0.1

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
                        logger.warning(f"⚠️ {symbol} notional gereksinimi karşılanamıyor")
                        return 0.0
                        
                    qty = required_qty

            return max(qty, 0.0)

        except Exception as e:
            logger.error(f"❌ Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0

    def _format_price(self, symbol: str, price: float) -> float:
        """Fiyat formatlama"""
        try:
            info = self.get_symbol_info(symbol)
            price_precision = int(info.get("price_precision", 8))
            return round(price, price_precision)
        except:
            return round(price, 8)

    def open_position(self, signal: Dict) -> bool:
        """Pozisyon açma"""
        try:
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()

            current_positions = config.live_positions
            
            if symbol in current_positions:
                logger.warning(f"⚠️ {symbol} için zaten açık pozisyon var")
                return False

            if len(current_positions) >= MAX_OPEN_POSITIONS:
                logger.warning(f"⚠️ Maksimum pozisyon sayısı: {MAX_OPEN_POSITIONS}")
                return False

            current_price = get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ {symbol} fiyat alınamadı")
                return False

            quantity = self.calculate_position_size(symbol, current_price)
            if quantity <= 0:
                logger.error(f"❌ {symbol} geçersiz pozisyon büyüklüğü")
                return False

            order_side = SIDE_BUY if side_txt == "LONG" else SIDE_SELL

            # Market emir
            main_order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                recvWindow=60000,
            )

            time.sleep(3)  # Emir kontrolü için bekle
            
            order_check = self.client.futures_get_order(
                symbol=symbol, 
                orderId=main_order["orderId"], 
                recvWindow=60000
            )
            
            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0))
            
            if order_status == "FILLED" and executed_qty > 0:
                avg_price = float(order_check.get("avgPrice") or current_price)
                quantity = executed_qty

                # SL/TP fiyatları
                if side_txt == "LONG":
                    stop_loss = self._format_price(symbol, avg_price * (1 - STOP_LOSS_PCT))
                    take_profit = self._format_price(symbol, avg_price * (1 + TAKE_PROFIT_PCT))
                    close_side = SIDE_SELL
                else:
                    stop_loss = self._format_price(symbol, avg_price * (1 + STOP_LOSS_PCT))
                    take_profit = self._format_price(symbol, avg_price * (1 - TAKE_PROFIT_PCT))
                    close_side = SIDE_BUY

                logger.info(f"✅ POZİSYON AÇILDI: {symbol} {side_txt} {executed_qty} @ ${avg_price:.6f}")

                # Stop Loss emri
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
                    logger.info(f"🛑 Stop Loss: ${stop_loss:.6f}")
                except Exception as e:
                    logger.error(f"❌ Stop Loss emri hatası: {e}")

                # Take Profit emri
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
                    logger.info(f"🎯 Take Profit: ${take_profit:.6f}")
                except Exception as e:
                    logger.error(f"❌ Take Profit emri hatası: {e}")

                # Pozisyon verisini kaydet
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
                
                return True
            else:
                logger.error(f"❌ {symbol} market emir başarısız: {order_status}")
                return False

        except Exception as e:
            logger.error(f"❌ Pozisyon açma hatası {symbol}: {e}")
            return False

    def close_position(self, symbol: str, close_reason: str) -> bool:
        """Pozisyon kapatma"""
        try:
            current_positions = config.live_positions
            
            if symbol not in current_positions:
                logger.warning(f"⚠️ {symbol} açık pozisyon bulunamadı")
                return False

            position = current_positions[symbol]

            # SL/TP emirlerini iptal et
            if position.get("auto_sltp", False):
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                for order_id, order_name in [(sl_order_id, "SL"), (tp_order_id, "TP")]:
                    if order_id:
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=order_id, recvWindow=60000)
                        except Exception as e:
                            logger.debug(f"⚠️ {symbol} {order_name} iptal hatası: {e}")

            # Pozisyonu kapat
            close_side = SIDE_SELL if position["side"] == "LONG" else SIDE_BUY

            close_order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=position["quantity"],
                recvWindow=60000,
            )

            time.sleep(3)  # Emir kontrolü için bekle
            
            order_check = self.client.futures_get_order(
                symbol=symbol, 
                orderId=close_order["orderId"], 
                recvWindow=60000
            )
            
            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0))

            if order_status == "FILLED" and executed_qty > 0:
                exit_price = float(order_check.get("avgPrice"))
                
                if position["side"] == "LONG":
                    pnl = (exit_price - position["entry_price"]) * position["quantity"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["quantity"]

                logger.info(f"✅ POZİSYON KAPANDI: {symbol} | Sebep: {close_reason} | P&L: ${pnl:.4f}")

                trade_data = position.copy()
                trade_data.update({
                    "exit_price": exit_price,
                    "current_value": position["quantity"] * exit_price,
                    "pnl": pnl,
                    "close_reason": close_reason,
                    "close_time": datetime.now(LOCAL_TZ),
                })

                self._log_trade_to_csv(trade_data, "CLOSED")
                del config.live_positions[symbol]
                sync_to_config()
                
                return True
            else:
                logger.error(f"❌ {symbol} kapatma emri başarısız: {order_status}")
                return False

        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma hatası {symbol}: {e}")
            return False

    def check_filled_orders_rest(self) -> None:
        """REST API ile SL/TP kontrol - WebSocket olmayan pozisyonlar için"""
        try:
            current_positions = config.live_positions
            symbols_to_remove = []
            
            for symbol, position in list(current_positions.items()):
                if not position.get("auto_sltp", False):
                    continue
                
                # WebSocket ile izlenen pozisyonları atla
                if symbol in websocket_active_symbols:
                    continue
                    
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                if not sl_order_id and not tp_order_id:
                    continue
                
                # SL kontrolü
                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(
                            symbol=symbol, orderId=sl_order_id, recvWindow=60000
                        )
                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss tetiklendi (REST)")
                            self._handle_auto_close(symbol, "Stop Loss - REST", sl_order)
                            symbols_to_remove.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"SL kontrol hatası {symbol}: {e}")
                
                # TP kontrolü
                if tp_order_id:
                    try:
                        tp_order = self.client.futures_get_order(
                            symbol=symbol, orderId=tp_order_id, recvWindow=60000
                        )
                        if tp_order["status"] == "FILLED":
                            logger.info(f"🎯 {symbol} Take Profit tetiklendi (REST)")
                            self._handle_auto_close(symbol, "Take Profit - REST", tp_order)
                            symbols_to_remove.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"TP kontrol hatası {symbol}: {e}")
            
            # Kapatılan pozisyonları temizle
            for symbol in symbols_to_remove:
                if symbol in config.live_positions:
                    del config.live_positions[symbol]
                    sync_to_config()

        except Exception as e:
            logger.error(f"❌ REST emir kontrolü hatası: {e}")

    def _handle_auto_close(self, symbol: str, close_reason: str, filled_order: Dict) -> None:
        """REST API otomatik kapatma işleyicisi"""
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

            logger.info(f"✅ OTOMATIK KAPANIŞ: {symbol} | Sebep: {close_reason} | P&L: ${pnl:.4f}")

            # Diğer emri iptal et
            other_order_id = None
            if "Stop Loss" in close_reason:
                other_order_id = position.get("tp_order_id")
            elif "Take Profit" in close_reason:
                other_order_id = position.get("sl_order_id")
                
            if other_order_id:
                try:
                    self.client.futures_cancel_order(
                        symbol=symbol, orderId=other_order_id, recvWindow=60000
                    )
                except Exception as e:
                    logger.debug(f"⚠️ Diğer emir iptal hatası: {e}")

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

        except Exception as e:
            logger.error(f"❌ Otomatik kapatma işleme hatası {symbol}: {e}")

    def _handle_auto_close_websocket(self, symbol: str, close_reason: str, order_details: Dict) -> None:
        """WebSocket otomatik kapatma işleyicisi"""
        try:
            current_positions = config.live_positions
            if symbol not in current_positions:
                logger.warning(f"⚠️ WebSocket {symbol} pozisyonu bulunamadı")
                return
                
            position = current_positions[symbol]
            exit_price = float(order_details.get("avgPrice", position["entry_price"]))
            
            pnl = (
                (exit_price - position["entry_price"]) * position["quantity"]
                if position["side"] == "LONG"
                else (position["entry_price"] - exit_price) * position["quantity"]
            )

            logger.info(f"✅ WEBSOCKET OTOMATIK KAPANIŞ: {symbol} | Sebep: {close_reason} | P&L: ${pnl:.4f}")

            # Diğer emri iptal et
            other_order_id = None
            if "Stop Loss" in close_reason:
                other_order_id = position.get("tp_order_id")
            elif "Take Profit" in close_reason:
                other_order_id = position.get("sl_order_id")
                
            if other_order_id:
                try:
                    self.client.futures_cancel_order(
                        symbol=symbol, orderId=other_order_id, recvWindow=60000
                    )
                except Exception as e:
                    logger.debug(f"⚠️ Diğer emir iptal hatası: {e}")

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

        except Exception as e:
            logger.error(f"❌ WebSocket otomatik kapatma hatası {symbol}: {e}")

    def fill_empty_positions(self) -> None:
        """Boş pozisyon slotlarını doldur"""
        try:
            if not live_trading_active:
                return

            current_positions = config.live_positions
            current_count = len(current_positions)
            
            if current_count >= MAX_OPEN_POSITIONS:
                return

            needed_slots = MAX_OPEN_POSITIONS - current_count

            if config.current_data is None or config.current_data.empty:
                return

            df = config.current_data.copy()
            exclude_symbols = set(current_positions.keys())
            
            if exclude_symbols:
                df = df[~df["symbol"].isin(exclude_symbols)]

            if df.empty:
                return

            top_signals = df.head(needed_slots)
            
            logger.info(f"🎯 {len(top_signals)} yeni pozisyon açılacak")

            opened = 0
            for _, signal in top_signals.iterrows():
                if opened >= needed_slots:
                    break
                
                symbol = signal["symbol"]
                success = self.open_position(signal.to_dict())
                
                if success:
                    opened += 1
                    logger.info(f"🚀 {symbol} pozisyonu açıldı!")
                    time.sleep(1)

            if opened > 0:
                logger.info(f"🎊 {opened} yeni pozisyon açıldı")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")

    def _log_trade_to_csv(self, trade_data: Dict, status: str) -> None:
        """Trade'i CSV'ye kaydet"""
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
    """Ana live trading döngüsü - Hibrit WebSocket/REST"""
    global live_trading_active

    logger.info("🤖 Live Trading döngüsü başlatıldı")
    loop_count = 0

    while live_trading_active:
        try:
            loop_count += 1
            loop_start = time.time()

            current_positions = config.live_positions
            logger.info(f"🔄 Döngü #{loop_count} - Pozisyon: {len(current_positions)}/{MAX_OPEN_POSITIONS}")

            # Bakiye güncelle
            balance = live_bot.get_account_balance()
            logger.debug(f"💰 Bakiye: ${balance:.2f}")

            # Config senkronize et
            sync_to_config()

            # AKILLI WebSocket kurulumu (sadece açık pozisyonlar için)
            setup_smart_websocket()

            # REST API ile SL/TP kontrol (WebSocket olmayanlar için)
            live_bot.check_filled_orders_rest()

            # Boş slotları doldur
            live_bot.fill_empty_positions()

            # Capital log
            log_capital_to_csv()

            loop_time = time.time() - loop_start
            logger.debug(f"⏱️ Döngü tamamlandı: {loop_time:.2f}s")

            if current_positions:
                ws_count = len([s for s in current_positions.keys() if s in websocket_active_symbols])
                rest_count = len(current_positions) - ws_count
                logger.info(f"📊 Açık pozisyonlar: {', '.join(current_positions.keys())}")
                logger.debug(f"🤖 WebSocket: {ws_count}, 📱 REST: {rest_count}")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Live trading döngüsü hatası: {e}")
            time.sleep(30)

    logger.info("ℹ️ Live trading döngüsü sonlandırıldı")


def start_live_trading() -> bool:
    """Live trading başlat"""
    global live_trading_active, live_trading_thread

    with _cleanup_lock:
        if live_trading_thread is not None and live_trading_thread.is_alive():
            logger.warning("⚠️ Live trading zaten aktif")
            return False
            
        if live_trading_active:
            logger.warning("⚠️ Live trading flag aktif")
            return False

        if live_trading_thread is not None:
            try:
                live_trading_thread.join(timeout=1)
            except:
                pass
            live_trading_thread = None

        if not live_bot.connect_to_binance():
            logger.error("❌ Binance bağlantısı başarısız")
            return False

        logger.info("🚀 Live Trading başlatılıyor...")
        logger.info(f"🔑 Environment: {ENVIRONMENT}")
        logger.info(f"💰 Bakiye: ${live_bot.account_balance:.2f}")
        logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
        logger.info(f"🤖 Hibrit sistem: Açık pozisyonlar WebSocket, diğerleri REST")

        config.switch_to_live_mode()
        live_trading_active = True

        live_trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
        live_trading_thread.start()

        logger.info("✅ Live Trading başlatıldı")
        return True


def stop_live_trading() -> None:
    """Live trading durdur"""
    global live_trading_active, live_trading_thread

    with _cleanup_lock:
        if not live_trading_active:
            logger.info("💤 Live trading zaten durdurulmuş")
            return

        logger.info("🛑 Live Trading durduruluyor...")
        live_trading_active = False

        # WebSocket durdur
        stop_websocket()

        # Pozisyonları kapat
        current_positions = config.live_positions.copy()
        
        if current_positions:
            logger.info(f"🔒 {len(current_positions)} pozisyon kapatılıyor...")
            
            for symbol in current_positions.keys():
                try:
                    success = live_bot.close_position(symbol, "Trading Stopped")
                    if success:
                        logger.info(f"✅ {symbol} kapatıldı")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"❌ {symbol} kapatma hatası: {e}")

        # Thread durdur
        if live_trading_thread is not None:
            try:
                live_trading_thread.join(timeout=5)
            except Exception as e:
                logger.error(f"❌ Thread durdurma hatası: {e}")
            finally:
                live_trading_thread = None
        
        # Config temizle
        try:
            config.reset_live_trading()
        except Exception as e:
            logger.error(f"❌ Config reset hatası: {e}")
        
        # Client temizle
        try:
            if live_bot.client:
                live_bot.is_connected = False
                live_bot.client = None
        except Exception as e:
            logger.debug(f"Client cleanup hatası: {e}")
        
        logger.info("✅ Live Trading durduruldu")


def is_live_trading_active() -> bool:
    """Live trading aktif mi?"""
    return live_trading_active


def get_live_trading_status() -> Dict:
    """Live trading durum bilgileri"""
    current_positions = config.live_positions
    auto_sltp_count = sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    
    return {
        "is_active": live_trading_active,
        "api_connected": live_bot.is_connected,
        "balance": live_bot.account_balance,
        "environment": ENVIRONMENT,
        "open_positions": len(current_positions),
        "max_positions": MAX_OPEN_POSITIONS,
        "auto_sltp_positions": auto_sltp_count,
        "auto_sltp_enabled": True,
        "websocket_symbols": len(websocket_active_symbols),
    }


def get_live_bot_status_for_symbol(symbol: str) -> str:
    """Sembol için bot durumu"""
    try:
        current_positions = config.live_positions
        
        if symbol in current_positions:
            if symbol in websocket_active_symbols:
                return "✅🤖"  # Açık pozisyon + WebSocket
            else:
                return "✅📱"  # Açık pozisyon + REST
        else:
            return "⭐📱"  # Pozisyon yok, REST ile kontrol
    except:
        return "❓"


def get_auto_sltp_count() -> int:
    """Otomatik SL/TP pozisyon sayısı"""
    try:
        current_positions = config.live_positions
        return sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    except:
        return 0