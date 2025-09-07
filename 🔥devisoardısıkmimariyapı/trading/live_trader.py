"""
🤖 Live Trading Bot - Gerçek Binance API ile Trading
AI sinyallerini gerçek paraya çeviren bot sistemi (testnet/mainnet uyumlu)
- Testnet'te gerçekten TRADABLE olan semboller (ticker tabanlı) filtrelendi
- "En iyi 3" AI skorlu adaydan pozisyon açma kuralı uygulandı
- Min Notional / Lot Size / Min Qty kontrolleri ve detaylı loglar eklendi
- 🔥 AI SKOR MİNİMUM 30% (90% kaldırıldı)
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
            client.timestamp_offset = last_offset  # python-binance bu özelliği kullanır
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
        self.tradable_cache: Set[str] = set()  # Testnet/Mainnet TRADABLE semboller kümesi

    # ---------- Bağlantı & Keşif ----------

    def _refresh_tradable_cache(self) -> None:
        """
        Testnet/Mainnet TRADABLE sembolleri güvenilir biçimde keşfet (ticker tabanlı).
        - futures_symbol_ticker(): testnet'te gerçekten var olan sembolleri döndürür.
        - exchangeInfo kesişimi opsiyoneldir; varsa USDT-PERP ile kesişim alınır.
        """
        try:
            # 1) Ticker tabanlı liste (testnet için güvenilir)
            tickers = self.client.futures_symbol_ticker()
            cache = {t["symbol"] for t in tickers if t.get("symbol", "").endswith("USDT")}

            # 2) Opsiyonel: USDT-PERP (TRADING) ile kesiştir
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
            # Binance Client
            if ENVIRONMENT == "testnet":
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=True,
                )
                # 🔧 FUTURES testnet URL override (bazı sürümler gerekli)
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

            # Önce saat senkronu
            _sync_server_time(self.client)

            # Healthcheck (unsigned)
            self.client.futures_ping()

            # İlk signed çağrı (geniş recvWindow)
            account_info = self.client.futures_account(recvWindow=60000)
            self.account_balance = float(account_info["totalWalletBalance"])
            logger.info(f"✅ API bağlantısı başarılı - Bakiye: ${self.account_balance:.2f}")

            # Tradable sembolleri önbelleğe al (ticker tabanlı)
            self._refresh_tradable_cache()

            binance_client = self.client
            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"❌ Binance bağlantı hatası: {e}")
            self.is_connected = False
            return False

    # ---------- Yardımcılar ----------

    def get_account_balance(self) -> float:
        """💰 Hesap bakiyesini (totalWalletBalance) al."""
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
        """📊 Sembol bilgilerini al (lot size, min quantity, min notional, precision)."""
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
                            # Bazı hesap tiplerinde "notional" anahtarı bulunur
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
        """🔍 Sembolün gerçekten trade edilebilir olduğunu doğrula (son kontrol)."""
        try:
            # Önce cache kontrolü
            if self.tradable_cache and symbol not in self.tradable_cache:
                return False
            # Sonra canlı ticker ile doğrula
            _ = self.client.futures_symbol_ticker(symbol=symbol, recvWindow=60000)
            return True
        except Exception as e:
            logger.info(f"⛔ {symbol} tradable değil veya env'de mevcut değil: {e}")
            return False

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """📏 Pozisyon büyüklüğünü hesapla (min_notional / lot_size uyumlu)."""
        try:
            # Maks. pozisyon değeri
            max_position_value = self.account_balance * (MAX_POSITION_SIZE_PCT / 100)

            # Minimum order kontrolü
            if max_position_value < MIN_ORDER_SIZE:
                logger.warning(f"⚠️ Yetersiz bakiye - Min: ${MIN_ORDER_SIZE}, Mevcut: ${max_position_value:.2f}")
                return 0.0

            # Ham quantity
            raw_qty = max_position_value / price

            # Sembol parametreleri
            info = self.get_symbol_info(symbol)
            lot_size = float(info.get("lot_size") or 0.0)
            min_qty = float(info.get("min_qty") or 0.0)
            min_notional = float(info.get("min_notional") or 0.0)

            logger.debug(
                f"🔎 {symbol} price={price:.8f} raw_qty={raw_qty:.8f} "
                f"lot_size={lot_size} min_qty={min_qty} min_notional={min_notional}"
            )

            qty = raw_qty

            # Lot adımına yuvarla
            if lot_size > 0:
                qty = float(Decimal(str(qty)).quantize(Decimal(str(lot_size)), rounding=ROUND_DOWN))

            # Min qty kontrolü
            if min_qty > 0 and qty < min_qty:
                logger.warning(f"⚠️ {symbol} minimum quantity altında: {qty} < {min_qty}")
                return 0.0

            # Min notional kontrolü
            if min_notional > 0:
                notional = qty * price
                if notional < min_notional:
                    logger.warning(f"⚠️ {symbol} notional yetersiz: {notional:.6f} < {min_notional}")
                    # Eşiğe çek ve tekrar yuvarla
                    qty = (min_notional / price)
                    if lot_size > 0:
                        qty = float(Decimal(str(qty)).quantize(Decimal(str(lot_size)), rounding=ROUND_DOWN))
                    # Son kontrol
                    if qty * price < min_notional:
                        logger.warning(f"⚠️ {symbol} notional hâlâ yetersiz: {(qty*price):.6f} < {min_notional}")
                        return 0.0

            return max(qty, 0.0)

        except Exception as e:
            logger.error(f"❌ Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0

    # ---------- Emir İşlemleri ----------

    def open_position(self, signal: Dict) -> bool:
        """🚀 Pozisyon aç (MARKET)."""
        try:
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()  # 'LONG' or 'SHORT'

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
            quantity = self.calculate_position_size(symbol, current_price)
            if quantity <= 0:
                logger.error(f"❌ {symbol} için geçersiz pozisyon büyüklüğü (minNotional/lotSize olabilir)")
                return False

            order_side = SIDE_BUY if side_txt == "LONG" else SIDE_SELL

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
                logger.info(
                    f"💰 Yatırılan: ${position_data['invested_amount']:.2f} | "
                    f"SL: ${stop_loss:.6f} | TP: ${take_profit:.6f} | AI: {signal['ai_score']:.0f}%"
                )

                self._log_trade_to_csv(position_data, "OPEN")
                return True

            logger.error(f"❌ Order başarısız: {order}")
            return False

        except Exception as e:
            logger.error(f"❌ Pozisyon açma hatası {symbol}: {e}")
            return False

    def close_position(self, symbol: str, close_reason: str) -> bool:
        """🔒 Pozisyon kapat (MARKET)."""
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

    # ---------- İzleme & Seçim ----------

    def monitor_positions(self) -> None:
        """👀 Açık pozisyonları izle (SL/TP kontrolü)."""
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

    def fill_empty_positions(self) -> None:
        """🎯 Boş pozisyon slotlarını doldur → AI skoru en iyi 3 adaydan dene."""
        try:
            # 🔥 GLOBAL IMPORT FIX - Her çağrıda güncel current_data'yı al
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

            # 🔥 YENİ: config modülünden current_data'yı direkt oku
            current_data_live = config.current_data
            
            if current_data_live is None:
                logger.warning("❌ config.current_data None - sinyal verisi yok")
                return
            
            if current_data_live.empty:
                logger.warning("❌ config.current_data boş - sinyal verisi yok")
                return

            logger.info(f"✅ config.current_data mevcut: {len(current_data_live)} satır")

            # --- UI filtreleri ---
            min_ai = float(current_settings.get("min_ai_pct", current_settings.get("min_ai", 30.0)))
            min_streak = int(current_settings.get("min_streak", 3))
            min_move = float(current_settings.get("min_pct", 0.5))
            min_volr = float(current_settings.get("min_volr", 1.5))

            df = current_data_live.copy()

            # AI skoru 0-1 ise 0-100'e çevir
            try:
                if df["ai_score"].max() <= 1.0:
                    df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
            except Exception:
                pass

            before = len(df)
            logger.info(f"🧮 Live filtre öncesi: {before} satır")

            # 🔥 YENİ: AI skor minimum 30% (90% kaldırıldı)
            live_min_ai = max(min_ai, 30.0)  # UI'den gelen değer ile 30'un büyüğü
            df = df[df["ai_score"] >= live_min_ai]
            logger.info(f"🧮 AI >= {live_min_ai}% sonrası: {len(df)}")

            df = df[df["run_count"] >= min_streak]
            logger.info(f"🧮 Streak >= {min_streak} sonrası: {len(df)}")

            df = df[df["run_perc"] >= min_move]
            logger.info(f"🧮 Move% >= {min_move} sonrası: {len(df)}")

            if "vol_ratio" in df.columns:
                df = df[df["vol_ratio"].fillna(0) >= min_volr]
                logger.info(f"🧮 VolRatio >= {min_volr} sonrası: {len(df)}")

            # Zaten açık olanları çıkar
            exclude = set(open_positions.keys())
            df = df[~df["symbol"].isin(exclude)]
            logger.info(f"🧮 Açık pozisyonlar hariç sonrası: {len(df)}")

            # Testnet/Mainnet TRADABLE cache filtresi (varsa)
            if self.tradable_cache:
                before_tr = len(df)
                df = df[df["symbol"].isin(self.tradable_cache)]
                logger.info(f"🧮 Tradable cache filtresi sonrası: {len(df)} (önce: {before_tr})")

            if df.empty:
                sample = (
                    current_data_live.sort_values(["ai_score", "run_perc", "gauss_run"], ascending=[False, False, False])
                    .head(5)[["symbol", "ai_score", "run_count", "run_perc", "vol_ratio"]]
                )
                logger.info(f"🔍 Aday yok (AI>={live_min_ai}%). En iyi 5 örnek:\n{sample.to_string(index=False)}")
                return

            # --- En iyi 3 kuralı ---
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
                    time.sleep(1)  # rate limit

            if opened > 0:
                logger.info(f"🚀 {opened} yeni live pozisyon açıldı (en iyi 3 kuralı, AI>={live_min_ai}%)")
            else:
                logger.info(f"🔍 Uygun aday bulunamadı (AI>={live_min_ai}%, en iyi 3 kuralı) — minNotional/lot engeli olabilir")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # --- UI filtreleri ---
            min_ai = float(current_settings.get("min_ai_pct", current_settings.get("min_ai", 30.0)))
            min_streak = int(current_settings.get("min_streak", 3))
            min_move = float(current_settings.get("min_pct", 0.5))
            min_volr = float(current_settings.get("min_volr", 1.5))

            df = current_data.copy()

            # AI skoru 0-1 ise 0-100'e çevir
            try:
                if df["ai_score"].max() <= 1.0:
                    df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
            except Exception:
                pass

            before = len(df)
            logger.info(f"🧮 Live filtre öncesi: {before} satır")

            # 🔥 YENİ: AI skor minimum 30% (90% kaldırıldı)
            live_min_ai = max(min_ai, 30.0)  # UI'den gelen değer ile 30'un büyüğü
            df = df[df["ai_score"] >= live_min_ai]
            logger.info(f"🧮 AI >= {live_min_ai}% sonrası: {len(df)}")

            df = df[df["run_count"] >= min_streak]
            logger.info(f"🧮 Streak >= {min_streak} sonrası: {len(df)}")

            df = df[df["run_perc"] >= min_move]
            logger.info(f"🧮 Move% >= {min_move} sonrası: {len(df)}")

            if "vol_ratio" in df.columns:
                df = df[df["vol_ratio"].fillna(0) >= min_volr]
                logger.info(f"🧮 VolRatio >= {min_volr} sonrası: {len(df)}")

            # Zaten açık olanları çıkar
            exclude = set(open_positions.keys())
            df = df[~df["symbol"].isin(exclude)]
            logger.info(f"🧮 Açık pozisyonlar hariç sonrası: {len(df)}")

            # Testnet/Mainnet TRADABLE cache filtresi (varsa)
            if self.tradable_cache:
                before_tr = len(df)
                df = df[df["symbol"].isin(self.tradable_cache)]
                logger.info(f"🧮 Tradable cache filtresi sonrası: {len(df)} (önce: {before_tr})")

            if df.empty:
                sample = (
                    current_data.sort_values(["ai_score", "run_perc", "gauss_run"], ascending=[False, False, False])
                    .head(5)[["symbol", "ai_score", "run_count", "run_perc", "vol_ratio"]]
                )
                logger.info(f"🔍 Aday yok (AI>={live_min_ai}%). En iyi 5 örnek:\n{sample.to_string(index=False)}")
                return

            # --- En iyi 3 kuralı ---
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
                    time.sleep(1)  # rate limit

            if opened > 0:
                logger.info(f"🚀 {opened} yeni live pozisyon açıldı (en iyi 3 kuralı, AI>={live_min_ai}%)")
            else:
                logger.info(f"🔍 Uygun aday bulunamadı (AI>={live_min_ai}%, en iyi 3 kuralı) — minNotional/lot engeli olabilir")

        except Exception as e:
            logger.error(f"❌ Pozisyon doldurma hatası: {e}")

    # ---------- Log ----------

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

            # Bakiye
            balance = live_bot.get_account_balance()
            logger.info(f"💰 Mevcut bakiye: ${balance:.2f}")

            # Boş slotları doldur
            live_bot.fill_empty_positions()

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
    """🚀 Live trading'i başlat."""
    global live_trading_active, live_trading_thread

    # Çift thread koruması
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

    logger.info("🚀 Live Trading başlatılıyor...")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"💰 Başlangıç bakiyesi: ${live_bot.account_balance:.2f}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info(f"🛑 Stop Loss: %{STOP_LOSS_PCT * 100}")
    logger.info(f"🎯 Take Profit: %{TAKE_PROFIT_PCT * 100}")
    logger.info(f"🔥 AI SKOR MİNİMUM: 30% (Eski 90% kaldırıldı)")

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

    # Açık pozisyonlar varsa kapat
    if open_positions:
        logger.info(f"📚 {len(open_positions)} açık pozisyon toplu kapatılıyor...")
        for symbol in list(open_positions.keys()):
            live_bot.close_position(symbol, "Trading Stopped")
            time.sleep(0.5)  # rate limit

    logger.info("✅ Live Trading durduruldu")


def is_live_trading_active() -> bool:
    """📊 Live trading aktif mi?"""
    return live_trading_active


def get_live_trading_status() -> Dict:
    """📊 Live trading durum bilgilerini al."""
    return {
        "is_active": live_trading_active,
        "api_connected": live_bot.is_connected,
        "balance": live_bot.account_balance,
        "environment": ENVIRONMENT,
        "open_positions": len(open_positions),
        "max_positions": MAX_OPEN_POSITIONS,
    }