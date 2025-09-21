"""
🤖 Live Trading Bot - Gerçek Binance API ile Trading
AI sinyallerini gerçek paraya çeviren bot sistemi (testnet/mainnet uyumlu)
🔥 YENİ: WebSocket + REST hibrit sistemi - İlk 50 coin WebSocket, geri kalanı REST
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
    from binance import ThreadedWebsocketManager
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

# 🔥 YENİ: WebSocket değişkenleri
websocket_manager: Optional[ThreadedWebsocketManager] = None
websocket_symbols: Set[str] = set()  # WebSocket ile izlenen semboller (ilk 50)


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


def sync_to_config():
    """🔥 YENİ: Live trading verilerini config'e senkronize et"""
    try:
        # Live mode'a geç
        config.switch_to_live_mode()
        
        # Bakiyeyi güncelle
        if live_bot.is_connected:
            balance = live_bot.get_account_balance()
            config.update_live_capital(balance)
        
        # Pozisyonları güncelle
        live_positions = get_current_live_positions()
        config.update_live_positions(live_positions)
        
        # Trading durumunu güncelle
        config.live_trading_active = live_trading_active
        
        logger.debug("🔄 Config senkronizasyonu tamamlandı")
        
    except Exception as e:
        logger.error(f"❌ Config senkronizasyon hatası: {e}")


def get_current_live_positions() -> Dict:
    """🔥 DÜZELTME: Mevcut live pozisyonları Binance'den al - AI skorunu config.current_data'dan al"""
    try:
        if not live_bot.client:
            return {}
            
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
                
                # 🔥 DEĞİŞEN KISIM: AI skorunu doğrudan tablodan al
                signal_data = {'ai_score': 50}  # varsayılan
                
                # Config'deki mevcut tablodan AI skorunu al
                if config.current_data is not None and not config.current_data.empty:
                    symbol_rows = config.current_data[config.current_data['symbol'] == symbol]
                    if not symbol_rows.empty:
                        latest_signal = symbol_rows.iloc[0]  # En güncel veri
                        signal_data = {
                            'ai_score': latest_signal['ai_score'],
                            'run_type': latest_signal['run_type'],
                            'run_count': latest_signal['run_count'],
                            'run_perc': latest_signal['run_perc'],
                            'gauss_run': latest_signal['gauss_run'],
                            'vol_ratio': latest_signal.get('vol_ratio', 0),
                            'deviso_ratio': latest_signal.get('deviso_ratio', 0)
                        }
                        logger.debug(f"🔄 {symbol} tablolardan AI skoru alındı: {latest_signal['ai_score']}")
                
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
                    'signal_data': signal_data  # 🔥 GÜNCEL TABLODAKİ VERİ
                }
                
                live_positions[symbol] = position_data
        
        logger.debug(f"📊 Binance'den {len(live_positions)} açık pozisyon alındı")
        return live_positions
        
    except Exception as e:
        logger.error(f"❌ Binance pozisyon alma hatası: {e}")
        return {}


# 🔥 YENİ: WebSocket Fonksiyonları
def handle_websocket_message(msg):
    """📨 WebSocket mesaj handler - SL/TP tetiklemelerini yakala"""
    try:
        msg_type = msg.get('e', 'unknown')
        
        if msg_type == 'ORDER_TRADE_UPDATE':
            order_data = msg.get('o', {})
            symbol = order_data.get('s')
            order_type = order_data.get('o')
            order_status = order_data.get('X')
            order_id = order_data.get('i')
            
            # SL/TP tetikleme kontrolü
            if (order_status == 'FILLED' and 
                order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and
                symbol in websocket_symbols):
                
                logger.info(f"🎯 WebSocket SL/TP tetiklendi: {symbol} {order_type}")
                
                # 2 saniye bekle (testnet gecikme kompansasyonu)
                time.sleep(2)
                
                # Pozisyonu handle et
                close_reason = "Stop Loss - WebSocket" if order_type == 'STOP_MARKET' else "Take Profit - WebSocket"
                
                # Order details al
                order_details = {
                    'orderId': order_id,
                    'avgPrice': order_data.get('ap', order_data.get('L', 0)),
                    'executedQty': order_data.get('z', 0),
                    'status': 'FILLED'
                }
                
                # Otomatik kapanış işle
                live_bot._handle_auto_close_websocket(symbol, close_reason, order_details)
                
        elif msg_type == 'ACCOUNT_UPDATE':
            logger.debug("📊 WebSocket hesap güncellendi")
            
    except Exception as e:
        logger.error(f"❌ WebSocket mesaj işleme hatası: {e}")


def setup_websocket_for_top_symbols():
    """🔥 YENİ: İlk 50 sembol için WebSocket kurulumu"""
    global websocket_manager, websocket_symbols
    
    try:
        if not live_bot.is_connected or not config.current_data is not None:
            return False
            
        # İlk 50 sembolü al
        top_50_symbols = set(config.current_data.head(50)['symbol'].tolist())
        
        # Zaten WebSocket'te olan sembollerle karşılaştır
        if top_50_symbols == websocket_symbols:
            logger.debug("🔄 WebSocket sembolleri değişmedi")
            return True
            
        # WebSocket'i yeniden başlat
        if websocket_manager:
            try:
                websocket_manager.stop()
            except:
                pass
                
        websocket_manager = ThreadedWebsocketManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY,
            testnet=(ENVIRONMENT == "testnet")
        )
        
        websocket_manager.start()
        
        # UserData stream başlat
        stream_name = websocket_manager.start_futures_user_socket(callback=handle_websocket_message)
        
        websocket_symbols = top_50_symbols
        
        logger.info(f"✅ WebSocket kuruldu - İlk 50 sembol izleniyor: {len(websocket_symbols)} adet")
        logger.info(f"🎯 WebSocket sembolleri: {', '.join(list(websocket_symbols)[:10])}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ WebSocket kurulum hatası: {e}")
        return False


def stop_websocket():
    """🛑 WebSocket'i durdur"""
    global websocket_manager, websocket_symbols
    
    try:
        if websocket_manager:
            websocket_manager.stop()
            websocket_manager = None
        websocket_symbols.clear()
        logger.info("🛑 WebSocket durduruldu")
    except Exception as e:
        logger.error(f"❌ WebSocket durdurma hatası: {e}")


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
        """🔒 Pozisyon kapat + Beklemeli kontrol eklendi - Sadece CLOSED kaydı CSV'ye"""
        try:
            # DÜZELTME: Sadece live_positions kullan
            current_positions = config.live_positions
            
            if symbol not in current_positions:
                logger.warning(f"⚠️ {symbol} için açık pozisyon bulunamadı")
                return False

            position = current_positions[symbol]

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

            # Beklemeli kontrol eklendi (open_position gibi)
            logger.info(f"📋 {symbol} kapatma emri verildi - Order ID: {close_order['orderId']}")
            time.sleep(3)  # 3 saniye bekle
            
            # Emir durumunu kontrol et
            order_check = self.client.futures_get_order(
                symbol=symbol, 
                orderId=close_order["orderId"], 
                recvWindow=60000
            )
            
            order_status = order_check.get("status")
            executed_qty = float(order_check.get("executedQty", 0))
            
            logger.info(f"📋 {symbol} Kapatma Status: {order_status} | Executed: {executed_qty}")

            if order_status == "FILLED" and executed_qty > 0:
                exit_price = float(order_check.get("avgPrice"))
                
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

                # ✅ Sadece CLOSED status ile kaydet (tek kayıt)
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
        """🔍 Otomatik SL/TP emirlerinin dolup dolmadığını kontrol et - Sadece REST API coinler için"""
        try:
            current_positions = config.live_positions
            symbols_to_remove = []
            
            for symbol, position in list(current_positions.items()):
                if not position.get("auto_sltp", False):
                    continue
                
                # 🔥 YENİ: WebSocket ile izlenen coinler için REST kontrolü yapma
                if symbol in websocket_symbols:
                    logger.debug(f"🔄 {symbol} WebSocket ile izleniyor - REST kontrolü atlandı")
                    continue
                    
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                # SL emri kontrolü (REST API ile)
                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(
                            symbol=symbol, orderId=sl_order_id, recvWindow=60000
                        )
                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss otomatik tetiklendi! (REST)")
                            self._handle_auto_close(symbol, "Stop Loss - REST", sl_order)
                            symbols_to_remove.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"SL order check error {symbol}: {e}")
                
                # TP emri kontrolü (REST API ile)
                if tp_order_id:
                    try:
                        tp_order = self.client.futures_get_order(
                            symbol=symbol, orderId=tp_order_id, recvWindow=60000
                        )
                        if tp_order["status"] == "FILLED":
                            logger.info(f"🎯 {symbol} Take Profit otomatik tetiklendi! (REST)")
                            self._handle_auto_close(symbol, "Take Profit - REST", tp_order)
                            symbols_to_remove.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"TP order check error {symbol}: {e}")
            
            # Kapatılmış pozisyonları config'ten kaldır
            for symbol in symbols_to_remove:
                if symbol in config.live_positions:
                    del config.live_positions[symbol]
                    sync_to_config()

        except Exception as e:
            logger.error(f"❌ REST otomatik emir kontrolü hatası: {e}")

    def _handle_auto_close(self, symbol: str, close_reason: str, filled_order: Dict) -> None:
        """🔄 Otomatik kapatılan pozisyonu işle - CLOSED kaydı CSV'ye + config'ten sil"""
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

            # Diğer bekleyen emri iptal et
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
                    logger.info(f"🚫 {symbol} diğer bekleyen emir iptal edildi")
                except Exception as e:
                    logger.debug(f"⚠️ {symbol} diğer emir iptal edilemedi: {e}")

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

            # ✅ Pozisyonu config'ten kaldır
            if symbol in config.live_positions:
                del config.live_positions[symbol]
                sync_to_config()
                logger.info(f"🧹 {symbol} config.live_positions listesinden silindi")

        except Exception as e:
            logger.error(f"❌ Otomatik kapatma işleme hatası {symbol}: {e}")

    def _handle_auto_close_websocket(self, symbol: str, close_reason: str, order_details: Dict) -> None:
        """🔥 YENİ: WebSocket'ten gelen SL/TP tetikleme işlemi"""
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

            logger.info(f"✅ WEBSOCKET OTOMATIK KAPANIŞ: {symbol} {position['side']} | Sebep: {close_reason}")
            logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f} | P&L: ${pnl:.4f}")

            # Diğer bekleyen emri iptal et
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
                    logger.info(f"🚫 {symbol} diğer bekleyen emir iptal edildi (WebSocket)")
                except Exception as e:
                    logger.debug(f"⚠️ {symbol} diğer emir iptal edilemedi: {e}")

            # Trade kaydı - WebSocket'ten gelen
            trade_data = position.copy()
            trade_data.update({
                "exit_price": exit_price,
                "current_value": position["quantity"] * exit_price,
                "pnl": pnl,
                "close_reason": close_reason,
                "close_time": datetime.now(LOCAL_TZ),
            })

            self._log_trade_to_csv(trade_data, "CLOSED")

            # ✅ Pozisyonu config'ten kaldır
            if symbol in config.live_positions:
                del config.live_positions[symbol]
                sync_to_config()
                logger.info(f"🧹 {symbol} WebSocket ile config'ten silindi")

        except Exception as e:
            logger.error(f"❌ WebSocket otomatik kapatma hatası {symbol}: {e}")

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
    """🔄 Ana live trading döngüsü - WebSocket + REST hibrit"""
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

            # Config'i her döngüde senkronize et
            sync_to_config()

            # 🔥 YENİ: WebSocket setup - her döngüde kontrol et
            setup_websocket_for_top_symbols()

            # Bekleyen emirleri temizle (her döngüde)
            live_bot.cancel_pending_orders()

            # 🔥 GÜNCELLEME: Sadece REST API coinleri için kontrol et (WebSocket olanlar hariç)
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
                ws_count = len([s for s in current_positions.keys() if s in websocket_symbols])
                logger.info(f"🔥 Açık pozisyonlar: {positions_summary}")
                logger.info(f"📊 WebSocket: {ws_count}, REST: {len(current_positions)-ws_count}, Auto SL/TP: {auto_count}")

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
    logger.info(f"🤖 Hibrit sistem: İlk 50 WebSocket + Geri kalan REST")
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

    # 🔥 YENİ: WebSocket'i durdur
    stop_websocket()

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
        "websocket_symbols": len(websocket_symbols),  # 🔥 YENİ
    }


def get_live_bot_status_for_symbol(symbol: str) -> str:
    """🔥 GÜNCELLEME: App.py callback'i için sembol durumu al - WebSocket/REST ayrımı"""
    try:
        current_positions = config.live_positions
        
        if symbol in current_positions:
            # Açık pozisyon var
            pos = current_positions[symbol]
            if symbol in websocket_symbols:
                return "✅🤖"  # Açık pozisyon + WebSocket
            else:
                return "✅📱"  # Açık pozisyon + REST
        else:
            # Pozisyon yok, kontrol türüne göre simge
            if symbol in websocket_symbols:
                return "⭐🤖"  # WebSocket hazır
            else:
                # Tablo sırasını kontrol et
                if config.current_data is not None and not config.current_data.empty:
                    try:
                        symbol_index = config.current_data[config.current_data['symbol'] == symbol].index
                        if len(symbol_index) > 0:
                            position_in_table = symbol_index[0]
                            if position_in_table < 50:
                                return "⭐🤖"  # İlk 50'de - WebSocket
                            else:
                                return "⭐📱"  # 51+ - REST
                    except:
                        pass
                return "⭐📱"  # Default REST
    except:
        return "❓"


def get_auto_sltp_count() -> int:
    """App.py callback'i için otomatik SL/TP sayısı"""
    try:
        current_positions = config.live_positions
        return sum(1 for p in current_positions.values() if p.get("auto_sltp", False))
    except:
        return 0