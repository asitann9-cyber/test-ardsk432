"""
🤖 Live Trading Bot - Gerçek Binance API ile Trading
AI sinyallerini gerçek paraya çeviren bot sistemi (testnet/mainnet uyumlu)
🔥 YENİ: Otomatik SL/TP emirleri - Binance sunucu tarafında kapatma
🔧 DÜZELTME: App.py ile tam uyumlu hale getirildi
"""

import time
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Set
from decimal import Decimal, ROUND_DOWN
import math

try:
    from binance.client import Client
    from binance.enums import *
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ python-binance kurulu değil: pip install python-binance")

from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, ENVIRONMENT, LOCAL_TZ,
    INITIAL_CAPITAL, MAX_OPEN_POSITIONS, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    SCAN_INTERVAL, MIN_ORDER_SIZE, MAX_POSITION_SIZE_PCT,
    current_capital, open_positions, trading_active, current_data, current_settings
)
from data.fetch_data import get_current_price
from data.database import log_trade_to_csv, log_capital_to_csv

logger = logging.getLogger("crypto-analytics")

# Global durum
binance_client: Optional["Client"] = None
live_trading_active: bool = False
live_trading_thread: Optional[threading.Thread] = None


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


class LiveTradingBot:
    """🤖 Gerçek Binance Trading Bot Sınıfı"""

    def __init__(self):
        self.client: Optional["Client"] = None
        self.is_connected: bool = False
        self.account_balance: float = 0.0
        self.tradable_cache: Set[str] = set()
        self.symbol_info_cache: Dict[str, Dict] = {}

    def _refresh_tradable_cache(self) -> None:
        """Testnet/Mainnet TRADABLE sembolleri güvenilir biçimde keşfet."""
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
            logger.info(f"🧭 TRADABLE (ticker tabanlı) semboller: {len(cache)} adet")
            if cache:
                logger.info("🧭 Örnek: " + ", ".join(list(cache)[:10]))
        except Exception as e:
            logger.warning(f"⚠️ Tradable sembol keşfi başarısız: {e}")

    def connect_to_binance(self) -> bool:
        """🔑 Binance API'ye bağlan ve bağlantıyı doğrula."""
        global binance_client

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
            return True

        except Exception as e:
            logger.error(f"❌ Binance bağlantı hatası: {e}")
            self.is_connected = False
            return False

    def get_account_balance(self) -> float:
        """💰 Hesap bakiyesini al."""
        try:
            if not self.client:
                return 0.0
            account_info = self.client.futures_account(recvWindow=60000)
            balance = float(account_info["totalWalletBalance"])
            self.account_balance = balance
            return balance
        except Exception as e:
            logger.error(f"❌ Bakiye alma hatası: {e}")
            return 0.0

    def get_symbol_info(self, symbol: str) -> Dict:
        """📊 Sembol bilgilerini al (cache'li)."""
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
        """🔍 Sembolün gerçekten trade edilebilir olduğunu doğrula."""
        try:
            if self.tradable_cache and symbol not in self.tradable_cache:
                return False
            _ = self.client.futures_symbol_ticker(symbol=symbol, recvWindow=60000)
            return True
        except Exception as e:
            logger.info(f"⛔ {symbol} tradable değil: {e}")
            return False

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """📏 Pozisyon büyüklüğünü hesapla (precision uyumlu)."""
        try:
            max_position_value = self.account_balance * (MAX_POSITION_SIZE_PCT / 100)

            if max_position_value < MIN_ORDER_SIZE:
                logger.warning(f"⚠️ Yetersiz bakiye - Min: ${MIN_ORDER_SIZE}, Mevcut: ${max_position_value:.2f}")
                return 0.0

            raw_qty = max_position_value / price

            info = self.get_symbol_info(symbol)
            lot_size = float(info.get("lot_size") or 0.0)
            min_qty = float(info.get("min_qty") or 0.0)
            min_notional = float(info.get("min_notional") or 0.0)
            quantity_precision = int(info.get("quantity_precision", 8))

            # Quantity precision'a göre yuvarla
            qty = round(raw_qty, quantity_precision)

            # Lot size kontrolü
            if lot_size > 0:
                qty_steps = qty / lot_size
                qty = math.floor(qty_steps) * lot_size
                qty = round(qty, quantity_precision)

            # Min quantity kontrolü
            if min_qty > 0 and qty < min_qty:
                logger.warning(f"⚠️ {symbol} minimum quantity altında: {qty} < {min_qty}")
                return 0.0

            # Min notional kontrolü
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

            logger.debug(f"🎯 {symbol} Final qty: {qty} (precision: {quantity_precision})")
            return max(qty, 0.0)

        except Exception as e:
            logger.error(f"❌ Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0

    def _format_price(self, symbol: str, price: float) -> float:
        """🎯 Fiyatı sembol precision'ına göre formatla."""
        try:
            info = self.get_symbol_info(symbol)
            price_precision = int(info.get("price_precision", 8))
            return round(price, price_precision)
        except:
            return round(price, 8)

    def open_position(self, signal: Dict) -> bool:
        """🚀 Pozisyon aç + Otomatik SL/TP emirleri."""
        try:
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()

            if symbol in open_positions:
                logger.warning(f"⚠️ {symbol} için zaten açık pozisyon var")
                return False

            if len(open_positions) >= MAX_OPEN_POSITIONS:
                logger.warning(f"⚠️ Maksimum pozisyon sayısına ulaşıldı: {MAX_OPEN_POSITIONS}")
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

            # Ana market emri
            main_order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                recvWindow=60000,
            )

            # Emir durumu kontrolü
            order_status = main_order.get("status")
            executed_qty = float(main_order.get("executedQty", 0))
            
            logger.info(f"📋 {symbol} Emir Status: {order_status} | Executed: {executed_qty}")
            
            if order_status in ["FILLED", "PARTIALLY_FILLED"] and executed_qty > 0:
                avg_price = float(main_order.get("avgPrice") or current_price)
                
                if executed_qty != quantity:
                    logger.warning(f"⚠️ {symbol} Kısmi dolum: İstenen={quantity}, Gerçekleşen={executed_qty}")
                
                quantity = executed_qty

                # SL/TP fiyatları hesapla
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
                logger.info(f"📊 SL: ${stop_loss:.6f} | TP: ${take_profit:.6f}")

                # Stop Loss emri (otomatik)
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
                    logger.info(f"🛑 Stop Loss emri verildi: ${stop_loss:.6f} (Order ID: {sl_order_id})")
                except Exception as e:
                    logger.error(f"❌ Stop Loss emri hatası {symbol}: {e}")

                # Take Profit emri (otomatik)
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
                    logger.info(f"🎯 Take Profit emri verildi: ${take_profit:.6f} (Order ID: {tp_order_id})")
                except Exception as e:
                    logger.error(f"❌ Take Profit emri hatası {symbol}: {e}")

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

                open_positions[symbol] = position_data
                self._log_trade_to_csv(position_data, "OPEN")
                
                if sl_order_id and tp_order_id:
                    logger.info(f"🤖 {symbol} otomatik SL/TP emirleri aktif - Binance sunucu tarafında kontrol edilecek")
                else:
                    logger.warning(f"⚠️ {symbol} otomatik SL/TP emirleri verilemedi - manuel kontrol yapılacak")
                
                return True

            logger.error(f"❌ Ana emir başarısız: {main_order}")
            return False

        except Exception as e:
            logger.error(f"❌ Pozisyon açma hatası {symbol}: {e}")
            return False

    def close_position(self, symbol: str, close_reason: str) -> bool:
        """🔒 Pozisyon kapat + Bekleyen emirleri iptal et."""
        try:
            if symbol not in open_positions:
                logger.warning(f"⚠️ {symbol} için açık pozisyon bulunamadı")
                return False

            position = open_positions[symbol]

            # Bekleyen SL/TP emirlerini iptal et
            if position.get("auto_sltp", False):
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                for order_id, order_name in [(sl_order_id, "SL"), (tp_order_id, "TP")]:
                    if order_id:
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=order_id, recvWindow=60000)
                            logger.info(f"🚫 {symbol} {order_name} emri iptal edildi (ID: {order_id})")
                        except Exception as e:
                            logger.debug(f"⚠️ {symbol} {order_name} emri iptal edilemedi: {e}")

            # Ana pozisyonu kapat
            close_side = SIDE_SELL if position["side"] == "LONG" else SIDE_BUY

            close_order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=position["quantity"],
                recvWindow=60000,
            )

            if close_order.get("status") == "FILLED":
                exit_price = float(close_order.get("avgPrice"))
                if position["side"] == "LONG":
                    pnl = (exit_price - position["entry_price"]) * position["quantity"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["quantity"]

                logger.info(f"✅ LIVE POZİSYON KAPANDI: {symbol} {position['side']} | Sebep: {close_reason}")
                logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f} | P&L: ${pnl:.4f}")

                trade_data = position.copy()
                trade_data.update({
                    "exit_price": exit_price,
                    "current_value": position["quantity"] * exit_price,
                    "pnl": pnl,
                    "close_reason": close_reason,
                    "close_time": datetime.now(LOCAL_TZ),
                })

                self._log_trade_to_csv(trade_data, "CLOSED")
                del open_positions[symbol]
                return True

            logger.error(f"❌ Pozisyon kapatma başarısız: {close_order}")
            return False

        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma hatası {symbol}: {e}")
            return False

    def monitor_positions(self) -> None:
        """👀 Açık pozisyonları izle - Otomatik SL/TP olan pozisyonları kontrol et."""
        try:
            if not open_positions:
                return

            logger.debug(f"👀 {len(open_positions)} açık pozisyon izleniyor...")
            
            for symbol, position in list(open_positions.items()):
                if position.get("auto_sltp", False):
                    current_price = get_current_price(symbol)
                    if current_price:
                        entry_price = position["entry_price"]
                        if position["side"] == "LONG":
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        else:
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        
                        logger.debug(f"🤖 {symbol} (Auto SL/TP): {current_price:.6f} | PnL: {pnl_pct:+.2f}%")

        except Exception as e:
            logger.error(f"❌ Pozisyon izleme hatası: {e}")

    def check_filled_orders(self) -> None:
        """🔍 Otomatik SL/TP emirlerinin dolup dolmadığını kontrol et."""
        try:
            symbols_to_remove = []
            
            for symbol, position in list(open_positions.items()):
                if not position.get("auto_sltp", False):
                    continue
                    
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                # SL emri kontrolü
                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(symbol=symbol, orderId=sl_order_id, recvWindow=60000)
                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss otomatik tetiklendi!")
                            self._handle_auto_close(symbol, "Stop Loss - Auto", sl_order)
                            symbols_to_remove.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"SL order check error {symbol}: {e}")
                
                # TP emri kontrolü
                if tp_order_id:
                    try:
                        tp_order = self.client.futures_get_order(symbol=symbol, orderId=tp_order_id, recvWindow=60000)
                        if tp_order["status"] == "FILLED":
                            logger.info(f"🎯 {symbol} Take Profit otomatik tetiklendi!")
                            self._handle_auto_close(symbol, "Take Profit - Auto", tp_order)
                            symbols_to_remove.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"TP order check error {symbol}: {e}")
            
            # Kapatılmış pozisyonları kaldır
            for symbol in symbols_to_remove:
                if symbol in open_positions:
                    del open_positions[symbol]

        except Exception as e:
            logger.error(f"❌ Otomatik emir kontrolü hatası: {e}")

    def _handle_auto_close(self, symbol: str, close_reason: str, filled_order: Dict) -> None:
        """🔄 Otomatik kapatılan pozisyonu işle."""
        try:
            if symbol not in open_positions:
                return
                
            position = open_positions[symbol]
            exit_price = float(filled_order.get("avgPrice", 0))
            
            if position["side"] == "LONG":
                pnl = (exit_price - position["entry_price"]) * position["quantity"]
            else:
                pnl = (position["entry_price"] - exit_price) * position["quantity"]

            logger.info(f"✅ OTOMATIK KAPANIŞ: {symbol} {position['side']} | Sebep: {close_reason}")
            logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f} | P&L: ${pnl:.4f}")

            # Diğer bekleyen emri iptal et
            other_order_id = None
            if "Stop Loss" in close_reason:
                other_order_id = position.get("tp_order_id")
            elif "Take Profit" in close_reason:
                other_order_id = position.get("sl_order_id")
                
            if other_order_id:
                try:
                    self.client.futures_cancel_order(symbol=symbol, orderId=other_order_id, recvWindow=60000)
                    logger.info(f"🚫 {symbol} diğer bekleyen emir iptal edildi")
                except:
                    pass

            # Trade kaydı
            trade_data = position.copy()
            trade_data.update({
                "exit_price": exit_price,
                "current_value": position["quantity"] * exit_price,
                "pnl": pnl,
                "close_reason": close_reason,
                "close_time": datetime.now(LOCAL_TZ),
            })

            self._log_trade_to_csv(trade_data, "CLOSED")

        except Exception as e:
            logger.error(f"❌ Otomatik kapatma işleme hatası {symbol}: {e}")

    # 🔧 DÜZELTME: Eksik cancel_pending_orders fonksiyonu eklendi
    def cancel_pending_orders(self) -> None:
        """🚫 Bekleyen emirleri temizle."""
        try:
            # Açık pozisyonu olmayan bekleyen emirleri iptal et
            for symbol, position in list(open_positions.items()):
                if position.get("auto_sltp", False):
                    # Pozisyon varsa emirleri koruyun
                    continue
            
            logger.debug("🧹 Bekleyen emirler kontrol edildi")
            
        except Exception as e:
            logger.error(f"❌ Bekleyen emir temizleme hatası: {e}")

    def fill_empty_positions(self) -> None:
        """🎯 Boş pozisyon slotlarını doldur."""
        try:
            import config
            
            logger.info("🔄 fill_empty_positions başlatıldı")
            
            if not live_trading_active:
                logger.info("❌ Live trading aktif değil - çıkılıyor")
                return

            current_positions = len(open_positions)
            logger.info(f"📊 Mevcut pozisyon: {current_positions}/{MAX_OPEN_POSITIONS}")
            
            if current_positions >= MAX_OPEN_POSITIONS:
                logger.info("✅ Tüm pozisyon slotları dolu")
                return

            needed_slots = MAX_OPEN_POSITIONS - current_positions
            logger.info(f"🎯 Gereken slot sayısı: {needed_slots}")

            current_data_live = config.current_data
            
            if current_data_live is None or current_data_live.empty:
                logger.warning("❌ config.current_data boş - sinyal verisi yok")
                return

            logger.info(f"✅ config.current_data mevcut: {len(current_data_live)} satır")

            # Filtreleme
            min_ai = float(current_settings.get("min_ai", 30.0))
            min_streak = int(current_settings.get("min_streak", 3))
            min_move = float(current_settings.get("min_pct", 0.5))
            min_volr = float(current_settings.get("min_volr", 1.5))

            df = current_data_live.copy()

            if "ai_score" in df.columns and df["ai_score"].max() <= 1.0:
                df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)

            live_min_ai = max(min_ai, 30.0)
            df = df[df["ai_score"] >= live_min_ai]
            df = df[df["run_count"] >= min_streak]
            df = df[df["run_perc"] >= min_move]

            if "vol_ratio" in df.columns:
                df = df[df["vol_ratio"].fillna(0) >= min_volr]

            exclude = set(open_positions.keys())
            df = df[~df["symbol"].isin(exclude)]

            if self.tradable_cache:
                df = df[df["symbol"].isin(self.tradable_cache)]

            if df.empty:
                logger.info(f"🔍 Uygun aday yok (AI>={live_min_ai}%)")
                return

            df = df.sort_values(["ai_score", "run_perc", "gauss_run"], ascending=[False, False, False])
            top3 = df.head(3)

            logger.info(f"🎯 En iyi 3 aday (AI>={live_min_ai}%):")
            for _, r in top3.iterrows():
                logger.info(f"   • {r['symbol']} | AI={r['ai_score']:.0f}% | run={r['run_count']} | move={r['run_perc']:.2f}%")

            opened = 0
            for _, row in top3.iterrows():
                if opened >= needed_slots:
                    break
                sym = row["symbol"]
                ok = self._is_tradable_symbol(sym)
                logger.info(f"🔍 {sym} tradable kontrolü: {ok}")
                if not ok:
                    continue
                if self.open_position(row.to_dict()):
                    opened += 1
                    time.sleep(1)

            if opened > 0:
                logger.info(f"🚀 {opened} yeni live pozisyon açıldı (otomatik SL/TP ile)")
            else:
                logger.info(f"🔍 Uygun aday bulunamadı")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")

    def _log_trade_to_csv(self, trade_data: Dict, status: str) -> None:
        """📝 Trade'i CSV'ye kaydet."""
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


# ---------- Döngü & Kontrol ----------

live_bot = LiveTradingBot()


def live_trading_loop() -> None:
    """🔄 Ana live trading döngüsü."""
    global live_trading_active

    logger.info("🤖 Live Trading döngüsü başlatıldı")
    loop_count = 0

    while live_trading_active:
        try:
            loop_count += 1
            loop_start = time.time()

            logger.info(f"🔄 Live döngü #{loop_count} - Pozisyon: {len(open_positions)}/{MAX_OPEN_POSITIONS}")

            balance = live_bot.get_account_balance()
            logger.info(f"💰 Mevcut bakiye: ${balance:.2f}")

            # Bekleyen emirleri temizle (her döngüde)
            live_bot.cancel_pending_orders()

            # Otomatik SL/TP emirlerini kontrol et
            live_bot.check_filled_orders()

            # Boş slotları doldur
            live_bot.fill_empty_positions()

            # Pozisyonları izle (otomatik SL/TP için minimal)
            live_bot.monitor_positions()

            log_capital_to_csv()

            loop_time = time.time() - loop_start
            logger.info(f"⏱️ Döngü #{loop_count}: {loop_time:.2f}s tamamlandı")

            if open_positions:
                positions_summary = ", ".join(open_positions.keys())
                auto_count = sum(1 for p in open_positions.values() if p.get("auto_sltp", False))
                logger.info(f"🔥 Açık pozisyonlar: {positions_summary} (Otomatik SL/TP: {auto_count}/{len(open_positions)})")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Live trading döngüsü hatası: {e}")
            time.sleep(30)

    logger.info("ℹ️ Live trading döngüsü sonlandırıldı")


def start_live_trading() -> bool:
    """🚀 Live trading'i başlat."""
    global live_trading_active, live_trading_thread

    if live_trading_thread is not None and live_trading_thread.is_alive():
        logger.warning("⚠️ Live trading zaten aktif (thread alive)")
        return False
    if live_trading_active:
        logger.warning("⚠️ Live trading zaten aktif (flag)")
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
    logger.info(f"🤖 Otomatik SL/TP: Binance sunucu tarafında aktif")
    logger.info(f"🔥 AI SKOR MİNİMUM: 30%")

    live_trading_active = True

    live_trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
    live_trading_thread.start()

    logger.info("✅ Live Trading başlatıldı")
    return True


def stop_live_trading() -> None:
    """🛑 Live trading'i durdur."""
    global live_trading_active

    if not live_trading_active:
        logger.info("💤 Live trading zaten durdurulmuş")
        return

    logger.info("🛑 Live Trading durduruluyor...")
    live_trading_active = False

    if open_positions:
        logger.info(f"📚 {len(open_positions)} açık pozisyon toplu kapatılıyor...")
        for symbol in list(open_positions.keys()):
            live_bot.close_position(symbol, "Trading Stopped")
            time.sleep(0.5)

    logger.info("✅ Live Trading durduruldu")


def is_live_trading_active() -> bool:
    """📊 Live trading aktif mi?"""
    return live_trading_active


def get_live_trading_status() -> Dict:
    """📊 Live trading durum bilgilerini al."""
    auto_sltp_count = sum(1 for p in open_positions.values() if p.get("auto_sltp", False))
    
    return {
        "is_active": live_trading_active,
        "api_connected": live_bot.is_connected,
        "balance": live_bot.account_balance,
        "environment": ENVIRONMENT,
        "open_positions": len(open_positions),
        "max_positions": MAX_OPEN_POSITIONS,
        "auto_sltp_positions": auto_sltp_count,
        "auto_sltp_enabled": True,
    }


# 🔧 DÜZELTME: App.py'de eksik olan fonksiyonları ekle
def get_live_bot_status_for_symbol(symbol: str) -> str:
    """🔥 App.py callback'i için sembol durumu al"""
    try:
        if symbol in open_positions:
            pos = open_positions[symbol]
            if pos.get('auto_sltp', False):
                return "✅🤖"  # Açık pozisyon + otomatik SL/TP
            else:
                return "✅📱"  # Açık pozisyon + manuel
        else:
            return "⭐"  # Beklemede/değerlendirilecek
    except:
        return "❓"


def get_auto_sltp_count() -> int:
    """🔥 App.py callback'i için otomatik SL/TP sayısı"""
    try:
        return sum(1 for p in open_positions.values() if p.get("auto_sltp", False))
    except:
        return 0