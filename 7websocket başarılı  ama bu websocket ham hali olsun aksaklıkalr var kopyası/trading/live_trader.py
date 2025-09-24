"""
🤖 Live Trading Bot - Hata Düzeltilmiş Versiyon
DÜZELTME: TP/SL tetiklenme CSV kayıt sorunu + PyLance hataları çözüldü
"""

import os
import time
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Set, Any
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

# Global değişkenler - Tip tanımlamaları eklendi
binance_client: Optional[Client] = None
live_trading_active: bool = False
live_trading_thread: Optional[threading.Thread] = None

# WebSocket değişkenleri
websocket_manager: Optional[ThreadedWebsocketManager] = None
websocket_active_symbols: Set[str] = set()

# Thread güvenliği
_cleanup_lock = threading.Lock()


def sync_to_config() -> None:
    """Config senkronizasyonu"""
    try:
        config.switch_to_live_mode()
        
        if hasattr(live_bot, 'is_connected') and live_bot.is_connected:
            balance = live_bot.get_account_balance()
            config.update_live_capital(balance)
        
        live_positions = get_current_live_positions()
        config.update_live_positions(live_positions)
        config.live_trading_active = live_trading_active
        
        logger.debug("🔄 Config senkronize edildi")
        
    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")


def get_current_live_positions() -> Dict[str, Any]:
    """Pozisyon senkronizasyonu"""
    try:
        if not hasattr(live_bot, 'client') or not live_bot.client:
            return {}
            
        binance_positions = live_bot.client.futures_position_information()
        live_positions: Dict[str, Any] = {}
        
        for pos in binance_positions:
            position_amt = float(pos['positionAmt'])
            
            if abs(position_amt) > 0:
                symbol = pos['symbol']
                entry_price = float(pos['entryPrice'])
                side = 'LONG' if position_amt > 0 else 'SHORT'
                quantity = abs(position_amt)
                
                # SL/TP hesaplama
                if side == 'LONG':
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)
                
                # AI score
                signal_data = {'ai_score': 50}
                if hasattr(config, 'current_data') and config.current_data is not None and not config.current_data.empty:
                    symbol_rows = config.current_data[config.current_data['symbol'] == symbol]
                    if not symbol_rows.empty:
                        signal_data = symbol_rows.iloc[0].to_dict()
                
                live_positions[symbol] = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'invested_amount': quantity * entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now(LOCAL_TZ),
                    'auto_sltp': True,
                    'signal_data': signal_data
                }
        
        logger.debug(f"📊 {len(live_positions)} pozisyon senkronize edildi")
        return live_positions
        
    except Exception as e:
        logger.error(f"❌ Pozisyon senkronizasyon hatası: {e}")
        return {}


def handle_websocket_message(msg: Dict[str, Any]) -> None:
    """🔧 KRİTİK DÜZELTME: WebSocket mesaj handler"""
    try:
        msg_type = msg.get('e', 'unknown')
        
        if msg_type == 'ORDER_TRADE_UPDATE':
            order_data = msg.get('o', {})
            symbol = order_data.get('s')
            order_type = order_data.get('o')
            order_status = order_data.get('X')
            avg_price = order_data.get('ap', '0')
            
            logger.info(f"📨 WebSocket: {symbol} {order_type} {order_status}")
            
            # TP/SL tetiklenme kontrolü
            if (order_status == 'FILLED' and 
                order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and
                symbol and symbol in config.live_positions):
                
                logger.info(f"🎯 SL/TP TETİKLENDİ: {symbol} {order_type}")
                
                try:
                    position = config.live_positions[symbol]
                    entry_price = float(position['entry_price'])
                    exit_price = float(avg_price) if avg_price != '0' else entry_price
                    quantity = float(position['quantity'])
                    
                    # P&L hesaplama
                    if position['side'] == 'LONG':
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - exit_price) * quantity
                    
                    close_reason = "Stop Loss - WebSocket" if order_type == 'STOP_MARKET' else "Take Profit - WebSocket"
                    
                    # 🔥 KRİTİK: CSV ÖNCE kaydet, başarılıysa temizle
                    csv_success = live_bot._save_closed_trade(symbol, exit_price, pnl, close_reason)
                    
                    if csv_success:
                        logger.info(f"✅ {symbol} CSV kayıt başarılı - pozisyon temizleniyor")
                        
                        # Pozisyonu config'den sil
                        if symbol in config.live_positions:
                            del config.live_positions[symbol]
                            sync_to_config()
                            
                        # WebSocket'ten çıkar
                        if symbol in websocket_active_symbols:
                            websocket_active_symbols.discard(symbol)
                    else:
                        logger.error(f"❌ {symbol} CSV kayıt başarısız - pozisyon korunuyor")
                
                except Exception as sltp_err:
                    logger.error(f"❌ {symbol} SL/TP işleme hatası: {sltp_err}")
            
        elif msg_type == 'ACCOUNT_UPDATE':
            logger.debug("📊 WebSocket hesap güncellendi")
            sync_to_config()
        
    except Exception as e:
        logger.error(f"❌ WebSocket mesaj hatası: {e}")


def setup_smart_websocket() -> bool:
    """WebSocket kurulumu"""
    global websocket_manager, websocket_active_symbols
    
    try:
        current_positions = set(config.live_positions.keys()) if hasattr(config, 'live_positions') else set()
        
        # Pozisyon yoksa WebSocket'i kapat
        if not current_positions:
            if websocket_manager:
                stop_websocket()
            logger.debug("📱 Pozisyon yok - WebSocket kapalı")
            return False
        
        # Değişiklik yoksa restart yapma
        if current_positions == websocket_active_symbols and websocket_manager:
            logger.debug("🔄 WebSocket değişiklik yok")
            return True
        
        # WebSocket restart
        logger.info(f"🔧 WebSocket restart: {len(current_positions)} pozisyon")
        
        if websocket_manager:
            stop_websocket()
            time.sleep(1)
        
        # Yeni WebSocket başlat
        if BINANCE_API_KEY and BINANCE_SECRET_KEY:
            websocket_manager = ThreadedWebsocketManager(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY,
                testnet=(ENVIRONMENT == "testnet")
            )
            
            websocket_manager.start()
            websocket_manager.start_futures_user_socket(callback=handle_websocket_message)
            
            websocket_active_symbols = current_positions
            
            logger.info(f"✅ WebSocket aktif - {len(websocket_active_symbols)} sembol")
            return True
        else:
            logger.error("❌ API anahtarları bulunamadı")
            return False
        
    except Exception as e:
        logger.error(f"❌ WebSocket kurulum hatası: {e}")
        return False


def stop_websocket() -> None:
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

    def __init__(self) -> None:
        self.client: Optional[Client] = None
        self.is_connected: bool = False
        self.account_balance: float = 0.0
        self.symbol_info_cache: Dict[str, Dict[str, Any]] = {}

    def connect_to_binance(self) -> bool:
        """Binance API bağlantısı"""
        global binance_client

        if not BINANCE_AVAILABLE:
            logger.error("❌ python-binance yüklü değil")
            return False

        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.error("❌ API anahtarları bulunamadı")
            return False

        try:
            self.client = Client(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY,
                testnet=(ENVIRONMENT == "testnet"),
            )

            # Timestamp sync
            server_time = self.client.futures_time()["serverTime"]
            local_time = int(time.time() * 1000)
            offset = int(server_time) - local_time
            self.client.timestamp_offset = offset
            
            # Ping test
            self.client.futures_ping()

            # Bakiye kontrol
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

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Sembol bilgileri"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        try:
            if not self.client:
                return {}
                
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info["symbols"]:
                if s["symbol"] == symbol:
                    info: Dict[str, Any] = {
                        "symbol": symbol,
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
                return 0.0

            if ENVIRONMENT == "testnet":
                max_position_value *= 0.1

            raw_qty = max_position_value / price
            info = self.get_symbol_info(symbol)
            
            quantity_precision = int(info.get("quantity_precision", 8))
            qty = round(raw_qty, quantity_precision)

            # Lot size kontrolü
            lot_size = float(info.get("lot_size") or 0.0)
            if lot_size > 0:
                qty_steps = qty / lot_size
                qty = math.floor(qty_steps) * lot_size
                qty = round(qty, quantity_precision)

            # Min quantity kontrolü
            min_qty = float(info.get("min_qty") or 0.0)
            if min_qty > 0 and qty < min_qty:
                return 0.0

            # Min notional kontrolü
            min_notional = float(info.get("min_notional") or 0.0)
            if min_notional > 0:
                notional = qty * price
                if notional < min_notional:
                    required_qty = min_notional / price
                    required_qty = round(required_qty, quantity_precision)
                    
                    if lot_size > 0:
                        qty_steps = required_qty / lot_size
                        required_qty = math.ceil(qty_steps) * lot_size
                        required_qty = round(required_qty, quantity_precision)
                    
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
        except Exception:
            return round(price, 8)

    def open_position(self, signal: Dict[str, Any]) -> bool:
        """Pozisyon açma"""
        try:
            if not self.client:
                logger.error("❌ Binance client bağlı değil")
                return False
                
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()

            if hasattr(config, 'live_positions') and symbol in config.live_positions:
                logger.warning(f"⚠️ {symbol} zaten açık")
                return False

            positions_count = len(config.live_positions) if hasattr(config, 'live_positions') else 0
            if positions_count >= MAX_OPEN_POSITIONS:
                logger.warning(f"⚠️ Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
                return False

            current_price = get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ {symbol} fiyat alınamadı")
                return False

            quantity = self.calculate_position_size(symbol, current_price)
            if quantity <= 0:
                logger.error(f"❌ {symbol} geçersiz quantity")
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

            time.sleep(2)

            order_check = self.client.futures_get_order(
                symbol=symbol,
                orderId=main_order["orderId"],
                recvWindow=60000,
            )

            if order_check["status"] == "FILLED":
                avg_price = float(order_check["avgPrice"])
                executed_qty = float(order_check["executedQty"])

                # SL/TP hesaplama
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
                    logger.info(f"🛑 SL: ${stop_loss:.6f}")
                except Exception as e:
                    logger.error(f"❌ SL emri hatası: {e}")

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
                    logger.info(f"🎯 TP: ${take_profit:.6f}")
                except Exception as e:
                    logger.error(f"❌ TP emri hatası: {e}")

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

                if not hasattr(config, 'live_positions'):
                    config.live_positions = {}
                config.live_positions[symbol] = position_data
                sync_to_config()

                # WebSocket aktif et
                setup_smart_websocket()

                return True
            else:
                logger.error(f"❌ {symbol} market emir başarısız")
                return False

        except Exception as e:
            safe_symbol = signal.get("symbol", "UNKNOWN") if isinstance(signal, dict) else "UNKNOWN"
            logger.error(f"❌ Pozisyon açma hatası {safe_symbol}: {e}")
            return False

    def close_position_manually(self, symbol: str, close_reason: str) -> bool:
        """Manuel pozisyon kapatma"""
        try:
            if not self.client:
                logger.error("❌ Binance client bağlı değil")
                return False
                
            if not hasattr(config, 'live_positions') or symbol not in config.live_positions:
                logger.warning(f"⚠️ {symbol} açık pozisyon bulunamadı")
                return False

            position = config.live_positions[symbol]

            # SL/TP emirlerini iptal et
            if position.get("auto_sltp", False):
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                for order_id, order_name in [(sl_order_id, "SL"), (tp_order_id, "TP")]:
                    if order_id:
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=order_id, recvWindow=60000)
                            logger.info(f"🚫 {symbol} {order_name} emri iptal edildi")
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

            time.sleep(2)
            
            order_check = self.client.futures_get_order(
                symbol=symbol, 
                orderId=close_order["orderId"], 
                recvWindow=60000
            )

            if order_check["status"] == "FILLED":
                exit_price = float(order_check["avgPrice"])
                entry_price = float(position["entry_price"])
                quantity = float(position["quantity"])
                
                # P&L hesaplama
                if position["side"] == "LONG":
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                logger.info(f"✅ MANUEL KAPATMA: {symbol} | P&L: ${pnl:.4f}")

                # CSV kaydet
                csv_success = self._save_closed_trade(symbol, exit_price, pnl, close_reason)

                if csv_success:
                    logger.info(f"✅ {symbol} CSV kayıt başarılı - pozisyon temizleniyor")
                    
                    if symbol in config.live_positions:
                        del config.live_positions[symbol]
                        sync_to_config()
                    
                    return True
                else:
                    logger.error(f"❌ {symbol} CSV kayıt başarısız")
                    return False
            else:
                logger.error(f"❌ {symbol} manuel kapatma başarısız")
                return False

        except Exception as e:
            logger.error(f"❌ Manuel pozisyon kapatma hatası {symbol}: {e}")
            return False

    def _save_closed_trade(self, symbol: str, exit_price: float, pnl: float, close_reason: str) -> bool:
        """CSV kayıt - 3 katmanlı doğrulama"""
        try:
            logger.info(f"📝 CSV kayıt: {symbol} | P&L: ${pnl:.4f}")
            
            # Position data kontrolü
            if not hasattr(config, 'live_positions') or symbol not in config.live_positions:
                logger.error(f"❌ {symbol} pozisyon config'de yok")
                return False
                
            position_data = config.live_positions[symbol]
            
            # CSV verisi hazırla
            csv_data = {
                "timestamp": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "side": position_data.get("side", "UNKNOWN"),
                "quantity": float(position_data.get("quantity", 0)),
                "entry_price": float(position_data.get("entry_price", 0)),
                "exit_price": float(exit_price),
                "invested_amount": float(position_data.get("invested_amount", 0)),
                "current_value": float(position_data.get("quantity", 0)) * float(exit_price),
                "pnl": float(pnl),
                "commission": 0.0,
                "ai_score": float(position_data.get("signal_data", {}).get("ai_score", 0)),
                "run_type": str(position_data.get("signal_data", {}).get("run_type", "")),
                "run_count": int(position_data.get("signal_data", {}).get("run_count", 0)),
                "run_perc": float(position_data.get("signal_data", {}).get("run_perc", 0)),
                "gauss_run": float(position_data.get("signal_data", {}).get("gauss_run", 0)),
                "vol_ratio": float(position_data.get("signal_data", {}).get("vol_ratio", 0)),
                "deviso_ratio": float(position_data.get("signal_data", {}).get("deviso_ratio", 0)),
                "stop_loss": float(position_data.get("stop_loss", 0)),
                "take_profit": float(position_data.get("take_profit", 0)),
                "close_reason": str(close_reason),
                "status": "CLOSED",
            }
            
            # Dosya boyut kontrolü
            csv_file = "ai_crypto_trades.csv"
            original_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
            
            # CSV'ye yaz
            write_success = log_trade_to_csv(csv_data)
            
            if not write_success:
                logger.error(f"❌ {symbol} log_trade_to_csv() başarısız")
                return False
            
            # Yazma doğrulaması
            time.sleep(0.5)
            new_size = os.path.getsize(csv_file) if os.path.exists(csv_file) else 0
            
            if new_size > original_size:
                logger.info(f"✅ {symbol} CSV kayıt DOĞRULANDI")
                return True
            else:
                logger.error(f"❌ {symbol} dosya boyutu artmadı")
                return False
                
        except Exception as e:
            logger.error(f"❌ {symbol} CSV kayıt hatası: {e}")
            return False

    def check_filled_orders_rest(self) -> None:
        """REST API ile SL/TP kontrol"""
        try:
            if not self.client or not hasattr(config, 'live_positions'):
                return
                
            current_positions = config.live_positions.copy()
            
            for symbol, position in current_positions.items():
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                # SL kontrolü
                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(
                            symbol=symbol, orderId=sl_order_id, recvWindow=60000
                        )
                        
                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss tetiklendi (REST)")
                            
                            exit_price = float(sl_order.get("avgPrice", position['entry_price']))
                            entry_price = float(position['entry_price'])
                            quantity = float(position['quantity'])
                            
                            if position['side'] == 'LONG':
                                pnl = (exit_price - entry_price) * quantity
                            else:
                                pnl = (entry_price - exit_price) * quantity
                            
                            # CSV kayıt
                            csv_success = self._save_closed_trade(symbol, exit_price, pnl, "Stop Loss - REST")
                            
                            if csv_success and symbol in config.live_positions:
                                del config.live_positions[symbol]
                                logger.info(f"✅ {symbol} REST-SL pozisyonu temizlendi")
                            
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
                            
                            exit_price = float(tp_order.get("avgPrice", position['entry_price']))
                            entry_price = float(position['entry_price'])
                            quantity = float(position['quantity'])
                            
                            if position['side'] == 'LONG':
                                pnl = (exit_price - entry_price) * quantity
                            else:
                                pnl = (entry_price - exit_price) * quantity
                            
                            # CSV kayıt
                            csv_success = self._save_closed_trade(symbol, exit_price, pnl, "Take Profit - REST")
                            
                            if csv_success and symbol in config.live_positions:
                                del config.live_positions[symbol]
                                logger.info(f"✅ {symbol} REST-TP pozisyonu temizlendi")
                            
                    except Exception as e:
                        logger.debug(f"TP kontrol hatası {symbol}: {e}")
            
            sync_to_config()
            
        except Exception as e:
            logger.error(f"❌ REST emir kontrolü hatası: {e}")

    def fill_empty_positions(self) -> None:
        """Boş pozisyon slotları doldur"""
        try:
            if not live_trading_active:
                return

            current_count = len(config.live_positions) if hasattr(config, 'live_positions') else 0
            
            if current_count >= MAX_OPEN_POSITIONS:
                return

            needed_slots = MAX_OPEN_POSITIONS - current_count

            if not hasattr(config, 'current_data') or config.current_data is None or config.current_data.empty:
                return

            df = config.current_data.copy()
            exclude_symbols = set(config.live_positions.keys()) if hasattr(config, 'live_positions') else set()
            
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
                
                success = self.open_position(signal.to_dict())
                
                if success:
                    opened += 1
                    logger.info(f"🚀 {signal['symbol']} pozisyonu açıldı!")
                    time.sleep(2)

            if opened > 0:
                logger.info(f"🎊 {opened} yeni pozisyon açıldı")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")


# Global bot instance
live_bot = LiveTradingBot()


def live_trading_loop() -> None:
    """Live trading döngüsü"""
    global live_trading_active

    logger.info("🤖 Live Trading döngüsü başlatıldı")
    loop_count = 0

    while live_trading_active:
        try:
            loop_count += 1
            current_positions = config.live_positions if hasattr(config, 'live_positions') else {}
            logger.info(f"🔄 Döngü #{loop_count} - Pozisyon: {len(current_positions)}/{MAX_OPEN_POSITIONS}")

            # Bakiye güncelle
            balance = live_bot.get_account_balance()
            logger.debug(f"💰 Bakiye: ${balance:.2f}")

            # Config senkronize et
            sync_to_config()

            # WebSocket kurulumu
            setup_smart_websocket()

            # REST API ile SL/TP kontrol
            live_bot.check_filled_orders_rest()

            # Boş slotları doldur
            live_bot.fill_empty_positions()

            # Capital log
            log_capital_to_csv()

            # WebSocket durumu
            if current_positions:
                ws_count = len(websocket_active_symbols)
                logger.info(f"📊 Pozisyonlar: {', '.join(current_positions.keys())}")
                logger.info(f"🤖 WebSocket: {', '.join(websocket_active_symbols)}")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Live trading döngüsü hatası: {e}")
            time.sleep(30)

    logger.info("ℹ️ Live trading döngüsü sonlandırıldı")


def start_live_trading() -> bool:
    """Live trading başlat"""
    global live_trading_active, live_trading_thread

    with _cleanup_lock:
        if live_trading_active:
            logger.warning("⚠️ Live trading zaten aktif")
            return False

        if not live_bot.connect_to_binance():
            logger.error("❌ Binance bağlantısı başarısız")
            return False

        logger.info("🚀 Live Trading başlatılıyor...")
        logger.info(f"💰 Bakiye: ${live_bot.account_balance:.2f}")
        logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")

        config.switch_to_live_mode()
        live_trading_active = True

        live_trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
        live_trading_thread.start()

        logger.info("✅ Live Trading başlatıldı")
        return True


def stop_live_trading() -> None:
    """Live trading durdur - tüm pozisyonları kapat"""
    global live_trading_active, live_trading_thread

    with _cleanup_lock:
        if not live_trading_active:
            logger.info("💤 Live trading zaten durdurulmuş")
            return

        logger.info("🛑 Live Trading durduruluyor - TÜM POZİSYONLAR KAPANACAK...")
        live_trading_active = False

        # WebSocket durdur
        stop_websocket()

        # Tüm pozisyonları kapat
        successful_closes = 0
        failed_closes = 0
        
        try:
            if live_bot.client and live_bot.is_connected:
                logger.info("🔍 Tüm açık pozisyonlar kapatılıyor...")
                
                # Binance'den pozisyonları al
                all_positions = live_bot.client.futures_position_information()
                active_positions = [pos for pos in all_positions if abs(float(pos['positionAmt'])) > 0]
                
                if not active_positions:
                    logger.info("ℹ️ Kapatılacak pozisyon yok")
                else:
                    logger.info(f"🔒 {len(active_positions)} pozisyon kapatılacak")
                    
                    for pos in active_positions:
                        symbol = pos['symbol']
                        position_amt = float(pos['positionAmt'])
                        entry_price = float(pos['entryPrice'])
                        quantity = abs(position_amt)
                        side = 'LONG' if position_amt > 0 else 'SHORT'
                        
                        try:
                            logger.info(f"🔒 {symbol} kapatılıyor: {side} {quantity}")
                            
                            # SL/TP emirlerini iptal et
                            try:
                                open_orders = live_bot.client.futures_get_open_orders(symbol=symbol, recvWindow=60000)
                                for order in open_orders:
                                    if order['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                                        live_bot.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'], recvWindow=60000)
                                        logger.info(f"🚫 {symbol} {order['type']} iptal edildi")
                            except Exception as cancel_err:
                                logger.debug(f"⚠️ {symbol} emir iptal hatası: {cancel_err}")
                            
                            # Pozisyonu kapat
                            close_side = SIDE_SELL if side == 'LONG' else SIDE_BUY
                            
                            close_order = live_bot.client.futures_create_order(
                                symbol=symbol,
                                side=close_side,
                                type=ORDER_TYPE_MARKET,
                                quantity=quantity,
                                recvWindow=60000
                            )
                            
                            time.sleep(2)
                            
                            # Emir kontrolü
                            order_check = live_bot.client.futures_get_order(
                                symbol=symbol,
                                orderId=close_order['orderId'],
                                recvWindow=60000
                            )
                            
                            if order_check['status'] == 'FILLED':
                                exit_price = float(order_check.get('avgPrice', entry_price))
                                
                                # P&L hesapla
                                if side == 'LONG':
                                    pnl = (exit_price - entry_price) * quantity
                                else:
                                    pnl = (entry_price - exit_price) * quantity
                                
                                logger.info(f"💰 {symbol} P&L: ${pnl:.4f}")
                                
                                # CSV'ye kaydet
                                csv_success = live_bot._save_closed_trade(symbol, exit_price, pnl, "Live Bot Stopped - Panel")
                                
                                if csv_success:
                                    logger.info(f"✅ {symbol} CSV kayıt başarılı")
                                    successful_closes += 1
                                else:
                                    logger.error(f"❌ {symbol} CSV kayıt başarısız")
                                    failed_closes += 1
                            else:
                                logger.error(f"❌ {symbol} kapatma başarısız")
                                failed_closes += 1
                            
                            time.sleep(1)
                            
                        except Exception as close_err:
                            logger.error(f"❌ {symbol} kapatma hatası: {close_err}")
                            failed_closes += 1
                
                # Sonuç
                total_attempts = successful_closes + failed_closes
                if total_attempts > 0:
                    logger.info(f"📊 Sonuç: {successful_closes} başarılı, {failed_closes} başarısız")
                
        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma genel hatası: {e}")

        # Thread'i durdur
        if live_trading_thread:
            try:
                live_trading_thread.join(timeout=10)
            except Exception as e:
                logger.error(f"❌ Thread durdurma hatası: {e}")
            finally:
                live_trading_thread = None
        
        # Config temizle
        try:
            if hasattr(config, 'live_positions'):
                config.live_positions.clear()
            sync_to_config()
        except Exception as e:
            logger.error(f"❌ Config temizleme hatası: {e}")
        
        logger.info("🏁 Live Trading TAMAMEN durduruldu")
        if successful_closes > 0:
            logger.info("📋 Trade geçmişi tablosunu kontrol et")


def is_live_trading_active() -> bool:
    """Live trading aktif mi?"""
    return live_trading_active


def get_live_trading_status() -> Dict[str, Any]:
    """Live trading durum bilgileri"""
    current_positions = config.live_positions if hasattr(config, 'live_positions') else {}
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
        current_positions = config.live_positions if hasattr(config, 'live_positions') else {}
        
        if symbol in current_positions:
            if symbol in websocket_active_symbols:
                return "✅🤖"  # Açık pozisyon + WebSocket
            else:
                return "✅📱"  # Açık pozisyon + REST
        else:
            return "⭐📱"  # Pozisyon yok, REST kontrol
    except Exception:
        return "❓"


def get_auto_sltp_count() -> int:
    """Otomatik SL/TP pozisyon sayısı"""
    try:
        current_positions = config.live_positions if hasattr(config, 'live_positions') else {}
        return sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    except Exception:
        return 0


# ============================================================================
# ✅ HATA DÜZELTMELERİ TAMAMLANDI
# ============================================================================

"""
🔧 DÜZELTILEN HATALAR:

1. ✅ "return" yalnızca bir işlev içinde kullanılabilir → Düzeltildi
2. ✅ IfTrade bekleniyor → Düzeltildi  
3. ✅ Beklenmeyen girinti → Düzeltildi
4. ✅ Girintiyi kaldırma beklenmiyordu → Düzeltildi
5. ✅ "stop_websocket" tanımlanmadı → Düzeltildi
6. ✅ "LiveTradingBot" tanımlanmadı → Düzeltildi
7. ✅ "e" tanımlanmadı → Düzeltildi
8. ✅ Tüm type hinting problemleri → Düzeltildi

🎯 ANA DÜZELTME KORUNDU:
- CSV ÖNCE kaydet, başarılıysa pozisyonu temizle
- WebSocket/REST hibrit sistem
- 3 katmanlı CSV doğrulama

📊 SONUÇ:
- PyLance hataları tamamen temizlendi
- Type safety eklendi
- Ana sorun çözümü korundu
"""

logger.info("✅ Hata düzeltilmiş live_trader.py yüklendi")
logger.info("🔧 PyLance hataları temizlendi + ana CSV sorunu çözüldü")