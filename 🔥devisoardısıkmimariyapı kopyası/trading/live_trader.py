"""
🤖 Live Trading Bot - DÜZELTME
✅ API endpoint birleştirmesi 
✅ Esnek position sizing ($100 hedef)
✅ Genişletilmiş candidate pool (5 aday)
✅ Düşürülmüş AI threshold (%70)
✅ Detaylı debug logları
"""

import time
import logging
import threading
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Set
from decimal import Decimal, ROUND_DOWN

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
    SCAN_INTERVAL, MIN_ORDER_SIZE, TARGET_POSITION_VALUE, FLEXIBLE_POSITION_SIZING,
    LIVE_TRADING_AI_PARAMS, BASE,  # 🔥 DÜZELTME: BASE config'ten gelir
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
    """🤖 DÜZELTME: Esnek Live Trading Bot"""

    def __init__(self):
        self.client: Optional["Client"] = None
        self.is_connected: bool = False
        self.account_balance: float = 0.0
        self.tradable_cache: Set[str] = set()

    # ---------- Bağlantı & Keşif (DÜZELTME) ----------

    def connect_to_binance(self) -> bool:
        """🔑 DÜZELTME: Birleşik endpoint ile bağlan"""
        global binance_client

        if not BINANCE_AVAILABLE:
            logger.error("❌ python-binance kütüphanesi yüklü değil")
            return False

        if not BINANCE_API_KEY or BINANCE_SECRET_KEY:
            logger.error("❌ API anahtarları .env dosyasında bulunamadı")
            return False

        try:
            # 🔥 DÜZELTME: Config'teki ENVIRONMENT'a göre client oluştur
            if ENVIRONMENT == "testnet":
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=True,
                )
                # URL override (config'teki BASE kullan)
                self.client.FUTURES_URL = f"{BASE}/fapi"
                logger.info(f"🧪 Testnet bağlantısı: {BASE}")
            else:
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                )
                logger.info(f"🚀 Mainnet bağlantısı: {BASE}")

            # Saat senkronu
            _sync_server_time(self.client)

            # Health check
            self.client.futures_ping()

            # Account bilgisi
            account_info = self.client.futures_account(recvWindow=60000)
            self.account_balance = float(account_info["totalWalletBalance"])
            logger.info(f"✅ API bağlantısı başarılı - Bakiye: ${self.account_balance:.2f}")

            # 🔥 DÜZELTME: Tradable sembolleri config'teki endpoint'ten al
            self._refresh_tradable_cache_unified()

            binance_client = self.client
            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"❌ Binance bağlantı hatası: {e}")
            self.is_connected = False
            return False

    def _refresh_tradable_cache_unified(self) -> None:
        """🔥 DÜZELTME: Unified endpoint'ten tradable sembolleri al"""
        try:
            # Exchange info ve ticker'ı aynı endpoint'ten al
            tickers = self.client.futures_symbol_ticker()
            ticker_symbols = {t["symbol"] for t in tickers if t.get("symbol", "").endswith("USDT")}

            exchange_info = self.client.futures_exchange_info()
            exchange_symbols = {
                s["symbol"] for s in exchange_info.get("symbols", [])
                if s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"
                and s.get("contractType") == "PERPETUAL"
            }

            # Kesişimi al (hem ticker'da hem exchange'de olan)
            self.tradable_cache = ticker_symbols & exchange_symbols
            
            logger.info(f"🧭 Unified tradable semboller: {len(self.tradable_cache)} adet")
            logger.info(f"🧭 Environment: {ENVIRONMENT}")
            if self.tradable_cache:
                sample = list(self.tradable_cache)[:10]
                logger.info(f"🧭 Örnek semboller: {', '.join(sample)}")

        except Exception as e:
            logger.warning(f"⚠️ Unified tradable keşfi başarısız: {e}")

    # ---------- Yardımcılar (DÜZELTME) ----------

    def get_account_balance(self) -> float:
        """💰 Hesap bakiyesini al"""
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
        """📊 Sembol bilgilerini al"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info["symbols"]:
                if s["symbol"] == symbol:
                    lot_size = None
                    min_qty = None
                    min_notional = None
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            lot_size = float(f["stepSize"])
                            min_qty = float(f["minQty"])
                        elif f["filterType"] == "MIN_NOTIONAL":
                            min_notional = float(f.get("notional", 0.0))
                    return {
                        "symbol": symbol,
                        "status": s["status"],
                        "lot_size": lot_size,
                        "min_qty": min_qty,
                        "min_notional": min_notional,
                        "price_precision": s["pricePrecision"],
                        "quantity_precision": s["quantityPrecision"],
                    }
            return {}
        except Exception as e:
            logger.error(f"❌ Sembol bilgisi alma hatası {symbol}: {e}")
            return {}

    def _is_tradable_symbol(self, symbol: str) -> bool:
        """🔍 Sembolün tradable olduğunu doğrula"""
        try:
            # Cache kontrolü
            if self.tradable_cache and symbol not in self.tradable_cache:
                logger.debug(f"⛔ {symbol} cache'de yok")
                return False
            
            # Live ticker kontrolü
            _ = self.client.futures_symbol_ticker(symbol=symbol, recvWindow=60000)
            logger.debug(f"✅ {symbol} tradable")
            return True
        except Exception as e:
            logger.info(f"⛔ {symbol} tradable değil: {e}")
            return False

    def calculate_position_size_flexible(self, symbol: str, price: float) -> float:
        """🔥 DÜZELTME: Esnek pozisyon boyutlandırma ($100 hedef)"""
        try:
            # 🔥 Hedef yatırım: $100 (config'ten)
            target_investment = TARGET_POSITION_VALUE
            
            logger.info(f"💰 {symbol} için hedef yatırım: ${target_investment}")
            logger.info(f"💲 Mevcut fiyat: ${price:.6f}")

            # Bakiye kontrolü
            if self.account_balance < target_investment:
                available = self.account_balance
                logger.warning(f"⚠️ Yetersiz bakiye! Mevcut: ${available:.2f}, Hedef: ${target_investment}")
                if available < MIN_ORDER_SIZE:
                    return 0.0
                target_investment = available * 0.8  # %80'ini kullan

            # Temel quantity hesaplama
            raw_qty = target_investment / price
            logger.info(f"🧮 Ham quantity: {raw_qty:.8f}")

            # Symbol bilgilerini al
            info = self.get_symbol_info(symbol)
            lot_size = float(info.get("lot_size") or 0.001)
            min_qty = float(info.get("min_qty") or 0.001)
            min_notional = float(info.get("min_notional") or 5.0)  # Default 5 USDT

            logger.info(f"📊 Symbol info - lot_size: {lot_size}, min_qty: {min_qty}, min_notional: {min_notional}")

            # Lot size'a yuvarla
            if lot_size > 0:
                qty = float(Decimal(str(raw_qty)).quantize(Decimal(str(lot_size)), rounding=ROUND_DOWN))
            else:
                qty = raw_qty

            logger.info(f"⚙️ Lot size sonrası: {qty:.8f}")

            # Minimum quantity kontrolü
            if qty < min_qty:
                qty = min_qty
                logger.info(f"⬆️ Min qty'ye ayarlandı: {qty:.8f}")

            # 🔥 DÜZELTME: Esnek min notional kontrolü
            notional_value = qty * price
            logger.info(f"💵 Notional değer: ${notional_value:.2f}")

            if min_notional > 0 and notional_value < min_notional:
                # Min notional'ı karşılamak için quantity'yi artır
                required_qty = min_notional / price
                if lot_size > 0:
                    required_qty = float(Decimal(str(required_qty)).quantize(Decimal(str(lot_size)), rounding=ROUND_DOWN))
                
                qty = max(qty, required_qty)
                final_notional = qty * price
                
                logger.info(f"📈 Min notional için qty artırıldı: {qty:.8f}")
                logger.info(f"💵 Final notional: ${final_notional:.2f}")

                # Hâlâ yetersizse esnek yaklaşım
                if final_notional < min_notional * 0.9:  # %90 tolerans
                    logger.warning(f"⚠️ Min notional karşılanamıyor: ${final_notional:.2f} < ${min_notional}")
                    # Yine de dene, belki exchange kabul eder
                    if final_notional >= MIN_ORDER_SIZE:
                        logger.info(f"🎯 MIN_ORDER_SIZE'ı karşılıyor, deneniyor...")
                    else:
                        return 0.0

            # Final kontroller
            final_value = qty * price
            
            # Minimum değer kontrolü
            if final_value < MIN_ORDER_SIZE:
                logger.warning(f"⚠️ Final değer çok küçük: ${final_value:.2f} < ${MIN_ORDER_SIZE}")
                return 0.0

            logger.info(f"✅ Final quantity: {qty:.8f}")
            logger.info(f"✅ Final value: ${final_value:.2f}")

            return max(qty, 0.0)

        except Exception as e:
            logger.error(f"❌ Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0

    # ---------- Emir İşlemleri (DÜZELTME) ----------

    def try_open_position_safe(self, signal: Dict) -> bool:
        """🔥 DÜZELTME: Güvenli pozisyon açma denemesi"""
        symbol = signal["symbol"]
        
        try:
            logger.info(f"🚀 {symbol} için pozisyon açma denemesi...")
            
            # Tradable kontrolü
            if not self._is_tradable_symbol(symbol):
                logger.info(f"⛔ {symbol} tradable değil - atlanıyor")
                return False

            # Fiyat al
            current_price = get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ {symbol} için fiyat alınamadı")
                return False

            # Quantity hesapla
            quantity = self.calculate_position_size_flexible(symbol, current_price)
            if quantity <= 0:
                logger.warning(f"⚠️ {symbol} için geçersiz quantity: {quantity}")
                return False

            # Pozisyon aç
            return self.open_position(signal)

        except Exception as e:
            logger.error(f"❌ {symbol} güvenli açma hatası: {e}")
            return False

    def open_position(self, signal: Dict) -> bool:
        """🚀 Pozisyon aç (MARKET)"""
        try:
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()

            # Çift pozisyon önleme
            if symbol in open_positions:
                logger.warning(f"⚠️ {symbol} için zaten açık pozisyon var")
                return False

            # Maks. pozisyon kontrolü
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                logger.warning(f"⚠️ Maksimum pozisyon sayısına ulaşıldı: {MAX_OPEN_POSITIONS}")
                return False

            # Son fiyat
            current_price = get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ {symbol} için fiyat alınamadı")
                return False

            # Miktar
            quantity = self.calculate_position_size_flexible(symbol, current_price)
            if quantity <= 0:
                logger.error(f"❌ {symbol} için geçersiz pozisyon büyüklüğü")
                return False

            order_side = SIDE_BUY if side_txt == "LONG" else SIDE_SELL

            logger.info(f"📤 Market order gönderiliyor: {symbol} {side_txt} {quantity}")

            # Market emri
            order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                recvWindow=60000,
            )

            if order.get("status") == "FILLED":
                avg_price = float(order.get("avgPrice") or current_price)

                # SL/TP hesapla
                if side_txt == "LONG":
                    stop_loss = avg_price * (1 - STOP_LOSS_PCT)
                    take_profit = avg_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = avg_price * (1 + STOP_LOSS_PCT)
                    take_profit = avg_price * (1 - TAKE_PROFIT_PCT)

                position_data = {
                    "symbol": symbol,
                    "side": side_txt,
                    "quantity": float(order["executedQty"]),
                    "entry_price": avg_price,
                    "invested_amount": float(order["executedQty"]) * avg_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now(LOCAL_TZ),
                    "signal_data": signal,
                    "order_id": order["orderId"],
                }

                open_positions[symbol] = position_data

                logger.info(f"✅ LIVE POZİSYON AÇILDI: {symbol} {side_txt} {quantity} @ ${avg_price:.6f}")
                logger.info(f"💰 Yatırılan: ${position_data['invested_amount']:.2f} | SL: ${stop_loss:.6f} | TP: ${take_profit:.6f}")
                logger.info(f"🤖 AI Skoru: {signal['ai_score']:.0f}% | Run: {signal['run_count']} | Move: {signal['run_perc']:.2f}%")

                self._log_trade_to_csv(position_data, "OPEN")
                return True

            logger.error(f"❌ Order başarısız: {order}")
            return False

        except Exception as e:
            logger.error(f"❌ Pozisyon açma hatası {symbol}: {e}")
            return False

    def close_position(self, symbol: str, close_reason: str) -> bool:
        """🔒 Pozisyon kapat (MARKET)"""
        try:
            if symbol not in open_positions:
                logger.warning(f"⚠️ {symbol} için açık pozisyon bulunamadı")
                return False

            position = open_positions[symbol]
            close_side = SIDE_SELL if position["side"] == "LONG" else SIDE_BUY

            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=position["quantity"],
                recvWindow=60000,
            )

            if order.get("status") == "FILLED":
                exit_price = float(order.get("avgPrice"))
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

            logger.error(f"❌ Pozisyon kapatma başarısız: {order}")
            return False

        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma hatası {symbol}: {e}")
            return False

    # ---------- İzleme & Seçim (DÜZELTME) ----------

    def monitor_positions(self) -> None:
        """👀 Açık pozisyonları izle (SL/TP kontrolü)"""
        try:
            if not open_positions:
                return

            logger.debug(f"👀 {len(open_positions)} açık pozisyon izleniyor...")
            to_close = []

            for symbol, position in list(open_positions.items()):
                current_price = get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"⚠️ {symbol} fiyat alınamadı - atlanıyor")
                    continue

                side = position["side"]
                sl = position["stop_loss"]
                tp = position["take_profit"]

                should_close = False
                reason = ""

                if side == "LONG":
                    if current_price <= sl:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price >= tp:
                        should_close = True
                        reason = "Take Profit"
                else:  # SHORT
                    if current_price >= sl:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price <= tp:
                        should_close = True
                        reason = "Take Profit"

                if should_close:
                    if self.close_position(symbol, reason):
                        to_close.append(symbol)

            if to_close:
                logger.info(f"🔄 Kapanan pozisyonlar: {to_close}")

        except Exception as e:
            logger.error(f"❌ Pozisyon izleme hatası: {e}")

    def fill_empty_positions_improved(self) -> None:
        """🔥 DÜZELTME: Genişletilmiş ve esnek pozisyon açma sistemi"""
        try:
            if not live_trading_active:
                return

            current_positions = len(open_positions)
            if current_positions >= MAX_OPEN_POSITIONS:
                return

            needed_slots = MAX_OPEN_POSITIONS - current_positions
            if needed_slots <= 0:
                return

            if current_data is None or current_data.empty:
                logger.debug("📊 Sinyal verisi yok")
                return

            # 🔥 DÜZELTME: Live trading için esnek parametreler
            live_params = LIVE_TRADING_AI_PARAMS
            min_ai = live_params['min_ai_score']  # %70
            min_streak = live_params['min_streak']  # 3
            min_move = live_params['min_pct']  # 0.5
            min_volr = live_params['min_volr']  # 1.5
            pool_size = live_params['candidate_pool_size']  # 5

            df = current_data.copy()

            # AI skoru 0-1 ise 0-100'e çevir
            try:
                if df["ai_score"].max() <= 1.0:
                    df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
            except Exception:
                pass

            before = len(df)
            logger.info(f"🔍 Live trading filtre başlangıcı: {before} sinyal")

            # Esnek filtreler
            df = df[df["ai_score"] >= min_ai]
            logger.info(f"🤖 AI >= {min_ai}% sonrası: {len(df)}")

            df = df[df["run_count"] >= min_streak]
            logger.info(f"🔢 Streak >= {min_streak} sonrası: {len(df)}")

            df = df[df["run_perc"] >= min_move]
            logger.info(f"📈 Move% >= {min_move} sonrası: {len(df)}")

            if "vol_ratio" in df.columns:
                df = df[df["vol_ratio"].fillna(0) >= min_volr]
                logger.info(f"📊 VolRatio >= {min_volr} sonrası: {len(df)}")

            # Açık pozisyonları hariç tut
            exclude = set(open_positions.keys())
            df = df[~df["symbol"].isin(exclude)]
            logger.info(f"🚫 Açık pozisyonlar hariç sonrası: {len(df)}")

            # Tradable cache filtresi
            if self.tradable_cache:
                before_tr = len(df)
                df = df[df["symbol"].isin(self.tradable_cache)]
                logger.info(f"✅ Tradable cache sonrası: {len(df)} (önce: {before_tr})")

            # 🔥 FALLBACK MODE: Hiç aday yoksa threshold'ları düşür
            if df.empty and live_params['fallback_mode']:
                logger.warning("🆘 Fallback mode: Threshold'lar düşürülüyor...")
                
                df = current_data.copy()
                if df["ai_score"].max() <= 1.0:
                    df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
                
                # Daha esnek kriterler
                df = df[df["ai_score"] >= 50.0]  # %50'ye düş
                df = df[df["run_count"] >= 2]  # 2'ye düş
                df = df[df["run_perc"] >= 0.3]  # 0.3'e düş
                df = df[~df["symbol"].isin(exclude)]
                
                if self.tradable_cache:
                    df = df[df["symbol"].isin(self.tradable_cache)]
                
                logger.warning(f"🆘 Fallback sonrası: {len(df)} aday")

            if df.empty:
                logger.info("🔍 Hiç uygun aday bulunamadı")
                return

            # 🔥 DÜZELTME: Genişletilmiş pool (5 aday)
            df = df.sort_values(["ai_score", "run_perc", "gauss_run"], ascending=[False, False, False])
            top_candidates = df.head(pool_size)

            logger.info(f"🎯 En iyi {len(top_candidates)} aday (hedef: {pool_size}):")
            for _, r in top_candidates.iterrows():
                logger.info(f"   🚀 {r['symbol']} | AI={r['ai_score']:.0f}% | Run={r['run_count']} | Move={r['run_perc']:.2f}%")

            # Pozisyonları aç
            opened = 0
            for _, row in top_candidates.iterrows():
                if opened >= needed_slots:
                    break
                
                symbol = row["symbol"]
                if self.try_open_position_safe(row.to_dict()):
                    opened += 1
                    logger.info(f"✅ {symbol} pozisyonu açıldı ({opened}/{needed_slots})")
                    time.sleep(1)  # Rate limit
                else:
                    logger.info(f"❌ {symbol} pozisyonu açılamadı")

            if opened > 0:
                logger.info(f"🚀 {opened} yeni live pozisyon açıldı!")
                logger.info(f"📊 Toplam pozisyon: {len(open_positions)}/{MAX_OPEN_POSITIONS}")
            else:
                logger.info("🔍 Hiç pozisyon açılamadı - muhtemelen teknik engeller (notional/lot size)")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")

    # ---------- Log ----------

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


# ---------- Döngü & Kontrol (DÜZELTME) ----------

live_bot = LiveTradingBot()


def live_trading_loop() -> None:
    """🔄 DÜZELTME: Ana live trading döngüsü"""
    global live_trading_active

    logger.info("🤖 Live Trading döngüsü başlatıldı")
    loop_count = 0

    while live_trading_active:
        try:
            loop_count += 1
            loop_start = time.time()

            logger.info(f"🔄 Live döngü #{loop_count} - Pozisyon: {len(open_positions)}/{MAX_OPEN_POSITIONS}")

            # Bakiye güncelle
            balance = live_bot.get_account_balance()
            logger.info(f"💰 Mevcut bakiye: ${balance:.2f}")

            # 🔥 DÜZELTME: Improved position filling
            live_bot.fill_empty_positions_improved()

            # Açık pozisyonları izle
            live_bot.monitor_positions()

            # Sermaye durumunu kaydet
            log_capital_to_csv()

            loop_time = time.time() - loop_start
            logger.info(f"⏱️ Döngü #{loop_count}: {loop_time:.2f}s tamamlandı")

            if open_positions:
                positions_summary = ", ".join(open_positions.keys())
                logger.info(f"🔥 Açık pozisyonlar: {positions_summary}")

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            logger.error(f"❌ Live trading döngüsü hatası: {e}")
            time.sleep(30)

    logger.info("ℹ️ Live trading döngüsü sonlandırıldı")


def start_live_trading() -> bool:
    """🚀 Live trading'i başlat"""
    global live_trading_active, live_trading_thread

    if live_trading_thread is not None and live_trading_thread.is_alive():
        logger.warning("⚠️ Live trading zaten aktif (thread alive)")
        return False
    if live_trading_active:
        logger.warning("⚠️ Live trading zaten aktif (flag)")
        return False

    # Bağlantı
    if not live_bot.connect_to_binance():
        logger.error("❌ Binance API bağlantısı başarısız")
        return False

    logger.info("🚀 DÜZELTME: Live Trading başlatılıyor...")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"🌐 API Endpoint: {BASE}")
    logger.info(f"💰 Başlangıç bakiyesi: ${live_bot.account_balance:.2f}")
    logger.info(f"💵 Hedef pozisyon değeri: ${TARGET_POSITION_VALUE}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info(f"🤖 AI threshold: {LIVE_TRADING_AI_PARAMS['min_ai_score']}%")
    logger.info(f"🎯 Candidate pool: {LIVE_TRADING_AI_PARAMS['candidate_pool_size']}")

    live_trading_active = True

    live_trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
    live_trading_thread.start()

    logger.info("✅ DÜZELTME: Live Trading başlatıldı")
    return True


def stop_live_trading() -> None:
    """🛑 Live trading'i durdur"""
    global live_trading_active

    if not live_trading_active:
        logger.info("💤 Live trading zaten durdurulmuş")
        return

    logger.info("🛑 Live Trading durduruluyor...")
    live_trading_active = False

    # Açık pozisyonlar varsa kapat
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
    """📊 Live trading durum bilgilerini al"""
    return {
        "is_active": live_trading_active,
        "api_connected": live_bot.is_connected,
        "balance": live_bot.account_balance,
        "environment": ENVIRONMENT,
        "target_position_value": TARGET_POSITION_VALUE,
        "open_positions": len(open_positions),
        "max_positions": MAX_OPEN_POSITIONS,
        "tradable_symbols": len(live_bot.tradable_cache),
        "ai_threshold": LIVE_TRADING_AI_PARAMS['min_ai_score'],
    }