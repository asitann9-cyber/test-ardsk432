"""
🤖 Live Trading Bot - Gerçek Binance API ile Trading
AI sinyallerini gerçek paraya çeviren bot sistemi (testnet/mainnet uyumlu)
🔥 YENİ: Otomatik SL/TP emirleri - Binance sunucu tarafında kapatma
🔧 DÜZELTME: Config senkronizasyonu ile dashboard entegrasyonu
🚨 SORUN GİDERİLDİ: Rastgele 3 emtia seçimi problemi çözüldü
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

# 🔥 YENİ: WebSocket değişkenlerini tanımla
websocket_active_symbols: Set[str] = set()
websocket_manager = None

def stop_websocket():
    """WebSocket bağlantısını kapat (placeholder)"""
    global websocket_manager, websocket_active_symbols
    websocket_manager = None
    websocket_active_symbols.clear()
    logger.info("🛑 WebSocket placeholder - kapatıldı")


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
    """🔥 En basit senkron + backfill: kaybolan sembol için Binance'ten kapanışı çek ve CSV'ye yaz."""
    try:
        # Live moda geç + bakiye
        config.switch_to_live_mode()
        if getattr(live_bot, "is_connected", False):
            try:
                balance = live_bot.get_account_balance()
                config.update_live_capital(balance)
            except Exception:
                pass

        # Güncellemeden önce eski durum
        prev_positions = dict(config.live_positions)
        prev_symbols = set(prev_positions.keys())

        # Taze pozisyonları çek
        fresh_positions = get_current_live_positions() or {}
        new_symbols = set(fresh_positions.keys())

        # Eski olup yeni listede olmayan semboller → kapanmıştır; backfill dene
        disappeared = prev_symbols - new_symbols
        for sym in disappeared:
            old = prev_positions.get(sym, {}) or {}
            if not old:
                continue
            # Sadece tek satır: kapanışı Binance fills'ten yaz
            if live_bot._backfill_closed_from_exchange(old):
                logger.info(f"🔒 Backfill ile {sym} kapatma CSV'ye yazıldı")

        # Son olarak config'i taze pozisyonlarla güncelle
        config.update_live_positions(fresh_positions)
        config.live_trading_active = live_trading_active
        logger.debug("🔄 Config senkronizasyonu tamamlandı (backfill uygulandı)")

    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")



def get_current_live_positions() -> Dict:
    """🔥 DÜZELTME: SL/TP Order ID'lerini koruyarak pozisyonları güncelle"""
    try:
        if not live_bot.client:
            return {}
            
        # Eski pozisyonları sakla (SL/TP ID'leri için)
        old_positions = config.live_positions.copy()
        
        # Binance'den açık pozisyonları al
        binance_positions = live_bot.client.futures_position_information()
        
        live_positions = {}
        
        for pos in binance_positions:
            symbol = pos['symbol']
            position_amt = float(pos['positionAmt'])
            
            # Sadece açık pozisyonları al
            if abs(position_amt) > 0:
                entry_price = float(pos['entryPrice'])
                mark_price = float(pos['markPrice'])
                unrealized_pnl = float(pos['unRealizedProfit'])
                
                # Pozisyon tarafı belirle
                side = 'LONG' if position_amt > 0 else 'SHORT'
                quantity = abs(position_amt)
                
                # SL/TP hesapla (varsayılan değerler)
                if side == 'LONG':
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)
                
                # AI skorunu doğrudan tablodan al
                signal_data = {'ai_score': 50}  # varsayılan
                
                # Config'deki mevcut tablodan AI skorunu al
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
                
                # 🔥 YENİ: Eski pozisyondaki SL/TP ID'lerini koru
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
                    'entry_time': entry_time,  # Eski zamanı koru
                    'auto_sltp': True,
                    'signal_data': signal_data,
                    # 🔥 YENİ: SL/TP ID'lerini koru
                    'sl_order_id': sl_order_id,
                    'tp_order_id': tp_order_id,
                    'main_order_id': main_order_id,
                }
                
                live_positions[symbol] = position_data
        
        logger.debug(f"📊 Binance'den {len(live_positions)} açık pozisyon alındı (SL/TP ID'ler korundu)")
        return live_positions
        
    except Exception as e:
        logger.error(f"❌ Binance pozisyon alma hatası: {e}")
        return {}


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
            
            # Config'i güncelle
            sync_to_config()
            
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
            
            # Config'e otomatik güncelle
            config.update_live_capital(balance)
            
            return balance
        except Exception as e:
            logger.error(f"❌ Bakiye alma hatası: {e}")
            return 0.0
        
    def _backfill_closed_from_exchange(self, old_pos: Dict) -> bool:
        """
        Kapanmış pozisyonu Binance fills üzerinden yakala ve trade geçmişine CLOSED olarak yaz.
        Testnet/Mainnet fark etmez.
        """
        try:
            from datetime import datetime  # yerel import
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

            # Entry zamanından sonraki fill'leri çek
            start_ms = int(entry_time.timestamp() * 1000)
            fills = self.client.futures_account_trades(
                symbol=symbol, startTime=start_ms, recvWindow=60000
            ) or []

            # Çıkış yönü (pozisyonun tersi)
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

                # Hedef miktar tamamlandıysa kır
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

            # Güvenli CSV yazımı (yalnızca CLOSED kaydı)
            self._log_trade_to_csv(trade_data, "CLOSED")

            # 🔥 YENİ: Açık emirleri temizle
            self.cleanup_symbol_orders(symbol)

            logger.info(f"🧾 Backfill CLOSED yazıldı: {symbol} @ {exit_price:.6f} (qty={qty})")
            return True

        except Exception as e:
            logger.error(f"❌ Backfill hata: {e}")
            return False


    
    

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
                logger.debug(f"🔍 {symbol} tradable_cache'de yok")
                return False
            _ = self.client.futures_symbol_ticker(symbol=symbol, recvWindow=60000)
            return True
        except Exception as e:
            logger.debug(f"⛔ {symbol} tradable değil: {e}")
            return False

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """📏 Pozisyon büyüklüğünü hesapla (precision uyumlu)."""
        try:
            max_position_value = self.account_balance * (MAX_POSITION_SIZE_PCT / 100)

            if max_position_value < MIN_ORDER_SIZE:
                logger.warning(f"⚠️ Yetersiz bakiye - Min: ${MIN_ORDER_SIZE}, Mevcut: ${max_position_value:.2f}")
                return 0.0

            # Testnet için küçük pozisyonlar
            if ENVIRONMENT == "testnet":
                max_position_value *= 0.1  # Testnet'te %10'unu kullan
                logger.debug(f"🧪 Testnet modu: Pozisyon boyutu küçültüldü: ${max_position_value:.2f}")

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
        """🚀 Pozisyon aç + Otomatik SL/TP emirleri - DÜZELTME: OPEN kaydı CSV'ye yazılmıyor"""
        try:
            symbol = signal["symbol"]
            side_txt = signal["run_type"].upper()

            # Config'den pozisyon kontrolü - DÜZELTME: Sadece live_positions
            current_positions = config.live_positions
            
            if symbol in current_positions:
                logger.warning(f"⚠️ {symbol} için zaten açık pozisyon var")
                return False

            if len(current_positions) >= MAX_OPEN_POSITIONS:
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

            # Market emir problemi için beklemeli kontrol
            main_order = self.client.futures_create_order(
                symbol=symbol,
                side=order_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                recvWindow=60000,
            )

            # Emir durumu beklemeli kontrol
            time.sleep(3)  # 3 saniye bekle
            
            order_check = self.client.futures_get_order(
                symbol=symbol, 
                orderId=main_order["orderId"], 
                recvWindow=60000
            )
            
            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0))
            
            logger.info(f"📋 {symbol} Emir Status Güncel: {order_status} | Executed: {executed_qty}")
            
            if order_status == "FILLED" and executed_qty > 0:
                avg_price = float(order_check.get("avgPrice") or current_price)
                
                if executed_qty != quantity:
                    logger.warning(f"⚠️ {symbol} Kısmi dolum: İstenen={quantity}, Gerçekleşen={executed_qty}")
                
                quantity = executed_qty

                # SL/TP fiyatları hesapla ve doğrula
                if side_txt == "LONG":
                    stop_loss = self._format_price(symbol, avg_price * (1 - STOP_LOSS_PCT))
                    take_profit = self._format_price(symbol, avg_price * (1 + TAKE_PROFIT_PCT))
                    close_side = SIDE_SELL
                    
                    # Doğrulama
                    sl_pct = ((avg_price - stop_loss) / avg_price) * 100
                    tp_pct = ((take_profit - avg_price) / avg_price) * 100
                    logger.info(f"🔍 {symbol} LONG SL/TP: Entry=${avg_price:.6f}, SL=${stop_loss:.6f}(-{sl_pct:.2f}%), TP=${take_profit:.6f}(+{tp_pct:.2f}%)")
                    
                else:
                    stop_loss = self._format_price(symbol, avg_price * (1 + STOP_LOSS_PCT))
                    take_profit = self._format_price(symbol, avg_price * (1 - TAKE_PROFIT_PCT))
                    close_side = SIDE_BUY
                    
                    # Doğrulama
                    sl_pct = ((stop_loss - avg_price) / avg_price) * 100
                    tp_pct = ((avg_price - take_profit) / avg_price) * 100
                    logger.info(f"🔍 {symbol} SHORT SL/TP: Entry=${avg_price:.6f}, SL=${stop_loss:.6f}(+{sl_pct:.2f}%), TP=${take_profit:.6f}(-{tp_pct:.2f}%)")

                logger.info(f"✅ LIVE POZİSYON AÇILDI: {symbol} {side_txt} {executed_qty} @ ${avg_price:.6f}")
                logger.info(f"💰 Yatırılan: ${executed_qty * avg_price:.2f}")

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

                # Config'e pozisyonu ekle
                config.live_positions[symbol] = position_data
                
                # 🔥 DÜZELTME: OPEN kaydı CSV'ye yazılmıyor artık
                # self._log_trade_to_csv(position_data, "OPEN")  # ❌ KALDIRILDI
                
                # Config'i senkronize et
                sync_to_config()
                
                if sl_order_id and tp_order_id:
                    logger.info(f"🤖 {symbol} otomatik SL/TP emirleri aktif - Binance sunucu tarafında kontrol edilecek")
                else:
                    logger.warning(f"⚠️ {symbol} otomatik SL/TP emirleri verilemedi - manuel kontrol yapılacak")
                
                return True
            else:
                logger.error(f"❌ {symbol} market emir beklemede kaldı: {order_status}")
                return False

        except Exception as e:
            logger.error(f"❌ Pozisyon açma hatası {symbol}: {e}")
            return False

    def close_position(self, symbol: str, close_reason: str) -> bool:
        """🔒 Pozisyon kapat + Temiz emir yönetimi ile"""
        try:
            import time
            from datetime import datetime

            # DÜZELTME: Sadece live_positions kullan
            current_positions = config.live_positions

            if symbol not in current_positions:
                logger.warning(f"⚠️ {symbol} için açık pozisyon bulunamadı")
                return False

            position = current_positions[symbol]

            # Ana pozisyonu kapat
            close_side = SIDE_SELL if position["side"] == "LONG" else SIDE_BUY

            close_order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=position["quantity"],
                recvWindow=60000,
            )

            # Beklemeli kontrol
            logger.info(f"📋 {symbol} kapatma emri verildi - Order ID: {close_order['orderId']}")
            time.sleep(3)  # 3 saniye bekle

            # Emir durumunu kontrol et
            order_check = self.client.futures_get_order(
                symbol=symbol,
                orderId=close_order["orderId"],
                recvWindow=60000,
            )

            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0.0))

            logger.info(f"📋 {symbol} Kapatma Status: {order_status} | Executed: {executed_qty}")

            if order_status == "FILLED" and executed_qty > 0:
                # avgPrice bazı durumlarda None/"" dönebilir → current_price'a düş
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

                # 🔥 YENİ: Tüm açık emirleri temizle (SL/TP dahil tümü)
                self.cleanup_symbol_orders(symbol)

                trade_data = position.copy()
                trade_data.update({
                    "exit_price": exit_price,
                    "current_value": position["quantity"] * exit_price,
                    "pnl": pnl,
                    "close_reason": close_reason,
                    "close_time": datetime.now(LOCAL_TZ),
                })

                # Sadece CLOSED status ile kaydet (tek kayıt)
                self._log_trade_to_csv(trade_data, "CLOSED")

                # Config'den pozisyonu sil
                del config.live_positions[symbol]
                sync_to_config()

                return True
            else:
                logger.error(f"❌ {symbol} kapatma emri beklemede kaldı: {order_status}")
                return False

        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma hatası {symbol}: {e}")
            return False


    def monitor_positions(self) -> None:
        """👀 Açık pozisyonları izle - Otomatik SL/TP olan pozisyonları kontrol et."""
        try:
            # DÜZELTME: Sadece live_positions kullan
            current_positions = config.live_positions
            
            if not current_positions:
                return

            logger.debug(f"👀 {len(current_positions)} açık pozisyon izleniyor...")
            
            for symbol, position in list(current_positions.items()):
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
        """🔍 Otomatik SL/TP emirlerinin dolup dolmadığını kontrol et ve trade geçmişine yaz."""
        try:
            current_positions = config.live_positions
            symbols_to_remove = set()

            # 🔍 DEBUG: Kaç pozisyon kontrol ediliyor
            auto_sltp_positions = {k: v for k, v in current_positions.items() if v.get("auto_sltp", False)}
            if auto_sltp_positions:
                logger.info(f"🔍 {len(auto_sltp_positions)} otomatik SL/TP pozisyonu kontrol ediliyor...")

            for symbol, position in list(current_positions.items()):
                if not position.get("auto_sltp", False):
                    continue

                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")

                # 🔍 DEBUG: Order ID'ler mevcut mu kontrol et
                logger.info(f"🔍 {symbol} - SL ID: {sl_order_id}, TP ID: {tp_order_id}")

                # SL emri kontrolü
                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(
                            symbol=symbol, orderId=sl_order_id, recvWindow=60000
                        )
                        # 🔍 DEBUG: SL emri durumu logla
                        logger.info(f"🔍 {symbol} SL Status: {sl_order['status']}")

                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss otomatik tetiklendi!")
                            self._handle_auto_close(symbol, "Stop Loss - Auto", sl_order)
                            symbols_to_remove.add(symbol)
                            continue
                    except Exception as e:
                        logger.error(f"❌ SL order check error {symbol}: {e}")  # debug → error
                else:
                    logger.warning(f"⚠️ {symbol} SL Order ID bulunamadı!")

                # TP emri kontrolü
                if tp_order_id:
                    try:
                        tp_order = self.client.futures_get_order(
                            symbol=symbol, orderId=tp_order_id, recvWindow=60000
                        )
                        # 🔍 DEBUG: TP emri durumu logla
                        logger.info(f"🔍 {symbol} TP Status: {tp_order['status']}")

                        if tp_order["status"] == "FILLED":
                            logger.info(f"🎯 {symbol} Take Profit otomatik tetiklendi!")
                            self._handle_auto_close(symbol, "Take Profit - Auto", tp_order)
                            symbols_to_remove.add(symbol)
                            continue
                    except Exception as e:
                        logger.error(f"❌ TP order check error {symbol}: {e}")  # debug → error
                else:
                    logger.warning(f"⚠️ {symbol} TP Order ID bulunamadı!")

            # 🔍 DEBUG: Kaç pozisyon silinecek
            if symbols_to_remove:
                logger.info(f"🔍 {len(symbols_to_remove)} pozisyon silinecek: {list(symbols_to_remove)}")

            # Kapatılmış pozisyonları config'ten kaldır (+ WebSocket temizliği)
            for symbol in symbols_to_remove:
                if symbol in config.live_positions:
                    del config.live_positions[symbol]
                    logger.info(f"🧹 {symbol} config'ten silindi")

                # WebSocket'ten de kaldır (varsa)
                try:
                    if symbol in websocket_active_symbols:
                        websocket_active_symbols.remove(symbol)
                        logger.debug(f"🧹 {symbol} WebSocket'ten kaldırıldı (auto-close cleanup)")
                except NameError:
                    # websocket_active_symbols tanımlı değilse sessiz geç
                    pass

            if symbols_to_remove:
                sync_to_config()

                # Tüm pozisyonlar kapandıysa WebSocket'i kapat
                try:
                    if len(config.live_positions) == 0 and len(websocket_active_symbols) == 0 and websocket_manager:
                        stop_websocket()
                        logger.info("🛑 WebSocket kapatıldı (auto-close - tüm pozisyonlar kapandı)")
                except NameError:
                    # websocket_active_symbols / websocket_manager tanımlı değilse sessiz geç
                    pass

        except Exception as e:
            logger.error(f"❌ Otomatik emir kontrolü hatası: {e}")
            import traceback
            logger.error(f"🔍 DEBUG Traceback: {traceback.format_exc()}")


    def cleanup_symbol_orders(self, symbol: str) -> None:
        """🧹 Sembol için tüm açık emirleri temizle (pozisyon kapandıktan sonra)"""
        try:
            if not self.client:
                return

            # O sembol için tüm açık emirleri al
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
                    logger.info(f"🚫 {symbol} emir iptal edildi: {order_type} {order_side} (ID: {order_id})")

                except Exception as e:
                    failed_count += 1
                    logger.warning(f"⚠️ {symbol} emir iptal hatası (ID: {order_id}): {e}")

            if cancelled_count > 0:
                logger.info(f"✅ {symbol} temizlik: {cancelled_count} emir iptal, {failed_count} başarısız")

        except Exception as e:
            logger.error(f"❌ {symbol} emir temizlik hatası: {e}")


    # _handle_auto_close fonksiyonunu güncelle:
    def _handle_auto_close(self, symbol: str, close_reason: str, filled_order: dict) -> None:
        """🔄 Otomatik kapatılan pozisyonu işle - CLOSED kaydı CSV'ye + temiz emir yönetimi"""
        try:
            from datetime import datetime  # yerel import

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

            # 🔥 YENİ: Tüm açık emirleri temizle (SL/TP dahil)
            self.cleanup_symbol_orders(symbol)

            # Trade kaydı - Sadece CLOSED status ile CSV'ye
            trade_data = position.copy()
            trade_data.update({
                "exit_price": exit_price,
                "current_value": position["quantity"] * exit_price,
                "pnl": pnl,
                "close_reason": close_reason,
                "close_time": datetime.now(LOCAL_TZ),
            })

            self._log_trade_to_csv(trade_data, "CLOSED")

            # Pozisyonu config'ten kaldır
            if symbol in config.live_positions:
                del config.live_positions[symbol]
                sync_to_config()
                logger.info(f"🧹 {symbol} config.live_positions listesinden silindi")

        except Exception as e:
            logger.error(f"❌ Otomatik kapatma işleme hatası {symbol}: {e}")


    def cancel_pending_orders(self) -> None:
        """🚫 Bekleyen emirleri temizle."""
        try:
            current_positions = config.live_positions
            for symbol, position in list(current_positions.items()):
                if position.get("auto_sltp", False):
                    continue
            logger.debug("🧹 Bekleyen emirler kontrol edildi")
        except Exception as e:
            logger.error(f"❌ Bekleyen emir temizleme hatası: {e}")

    def fill_empty_positions(self) -> None:
        """🎯 UI'deki filtrelenmiş en iyi sinyalleri al - Basit kod yaklaşımı"""
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
                logger.info(f"🔒 Hariç tutulan: {list(exclude_symbols)}")

            if df.empty:
                logger.info("ℹ️ Uygun yeni sembol yok (hepsi açık pozisyonlarda)")
                return

            logger.info("📊 UI'deki sıralama korunuyor - ilk N sinyali alınıyor")
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

            # 🔥 DÜZELTME: Sadece live_positions kullan
            current_positions = config.live_positions
            logger.info(f"🔄 Live döngü #{loop_count} - Pozisyon: {len(current_positions)}/{MAX_OPEN_POSITIONS}")

            balance = live_bot.get_account_balance()
            logger.info(f"💰 Mevcut bakiye: ${balance:.2f}")

            # Config'i her döngüde senkronize et
            sync_to_config()

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

            if current_positions:
                positions_summary = ", ".join(current_positions.keys())
                auto_count = sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
                logger.info(f"🔥 Açık pozisyonlar: {positions_summary} (Otomatik SL/TP: {auto_count}/{len(current_positions)})")

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
    logger.info(f"🚨 SORUN GİDERİLDİ: En yüksek AI skorlu sinyaller seçiliyor")

    # Live mode'a geç
    config.switch_to_live_mode()
    live_trading_active = True

    live_trading_thread = threading.Thread(target=live_trading_loop, daemon=True)
    live_trading_thread.start()

    logger.info("✅ Live Trading başlatıldı")
    return True


def stop_live_trading() -> None:
    """🛑 Live trading'i durdur - DÜZELTME: Sadece live positions"""
    global live_trading_active

    if not live_trading_active:
        logger.info("💤 Live trading zaten durdurulmuş")
        return

    logger.info("🛑 Live Trading durduruluyor...")
    live_trading_active = False

    # 🔥 DÜZELTME: Sadece live_positions kullan (paper yok artık)
    current_positions = config.live_positions.copy()  # Kopya al güvenlik için
    
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
                time.sleep(1)  # 1 saniye bekle
            except Exception as e:
                failed_closes += 1
                logger.error(f"❌ {symbol} kapatma hatası: {e}")
        
        logger.info(f"📊 Kapatma özeti: ✅{successful_closes} başarılı, ❌{failed_closes} başarısız")
    
    # 🔥 KALDIRILAN: config.switch_to_paper_mode() - artık yok
    # Config'i senkronize et
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
    """📊 Live trading durum bilgilerini al - DÜZELTME"""
    # 🔥 DÜZELTME: Sadece live_positions kullan
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
    }


# 🔥 YENİ: Config uyumlu fonksiyonlar - DÜZELTME
def get_live_bot_status_for_symbol(symbol: str) -> str:
    """App.py callback'i için sembol durumu al - DÜZELTME"""
    try:
        # 🔥 DÜZELTME: Sadece live_positions kullan
        current_positions = config.live_positions
        
        if symbol in current_positions:
            pos = current_positions[symbol]
            if pos.get('auto_sltp', False):
                return "✅🤖"  # Açık pozisyon + otomatik SL/TP
            else:
                return "✅📱"  # Açık pozisyon + manuel
        else:
            return "⭐"  # Beklemede/değerlendirilecek
    except:
        return "❓"


def get_auto_sltp_count() -> int:
    """App.py callback'i için otomatik SL/TP sayısı - DÜZELTME"""
    try:
        # 🔥 DÜZELTME: Sadece live_positions kullan
        current_positions = config.live_positions
        return sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    except:
        return 0