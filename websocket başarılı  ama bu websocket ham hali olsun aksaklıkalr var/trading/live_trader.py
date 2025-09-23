"""
🤖 Live Trading Bot - Hibrit WebSocket/REST Sistemi
DÜZELTME: TP/SL tetiklenme ve manuel kapatma sorunları çözüldü
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
    """🔧 DÜZELTME: TestBinance ile senkronize pozisyon alma"""
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
        
        logger.debug(f"📊 TestBinance'den {len(live_positions)} pozisyon senkronize edildi")
        return live_positions
        
    except Exception as e:
        logger.error(f"❌ Binance pozisyon senkronizasyon hatası: {e}")
        return {}


def handle_websocket_message(msg):
    """🔧 DÜZELTME: WebSocket mesaj handler - TP/SL tetiklenme kayıt sistemi + HATA KORUNMALI"""
    try:
        msg_type = msg.get('e', 'unknown')
        
        if msg_type == 'ORDER_TRADE_UPDATE':
            order_data = msg.get('o', {})
            symbol = order_data.get('s')
            order_type = order_data.get('o')
            order_status = order_data.get('X')
            order_id = order_data.get('i')
            avg_price = order_data.get('ap', '0')
            filled_qty = order_data.get('z', '0')
            
            logger.info(f"📨 WebSocket: {symbol} {order_type} {order_status} (Order: {order_id})")
            
            # 🔧 DÜZELTME: SL/TP tetiklendiğinde ANINDA trade geçmişine kaydet + DOĞRULAMA
            if (order_status == 'FILLED' and 
                order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and
                symbol in websocket_active_symbols and
                symbol in config.live_positions):
                
                logger.info(f"🎯 SL/TP TETİKLENDİ (WebSocket): {symbol} {order_type}")
                
                try:
                    # Position data kontrolü
                    if symbol not in config.live_positions:
                        logger.error(f"❌ {symbol} pozisyon config'de bulunamadı")
                        return
                    
                    position = config.live_positions[symbol]
                    entry_price = position['entry_price']
                    exit_price = float(avg_price)
                    quantity = position['quantity']
                    
                    # Kar/zarar hesapla
                    if position['side'] == 'LONG':
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - exit_price) * quantity
                    
                    close_reason = "Stop Loss - WebSocket" if order_type == 'STOP_MARKET' else "Take Profit - WebSocket"
                    
                    logger.info(f"💰 Entry: ${entry_price:.6f} → Exit: ${exit_price:.6f} | P&L: ${pnl:.4f}")
                    
                    # 🔥 YENİ: CSV kaydı + RETURN VALUE KONTROLÜ
                    logger.info("📝 WebSocket CSV kaydı başlatılıyor...")
                    
                    csv_success = live_bot._save_closed_trade(symbol, exit_price, pnl, close_reason)
                    
                    if csv_success:
                        logger.info(f"✅ {symbol} WebSocket CSV kaydı BAŞARILI - Trade geçmişinde görünecek")
                        
                        # Pozisyonu config'den temizle
                        if symbol in config.live_positions:
                            del config.live_positions[symbol]
                            sync_to_config()
                            logger.info(f"🗑️ {symbol} pozisyonu config'den temizlendi")
                    else:
                        logger.error(f"❌ {symbol} WebSocket CSV kaydı BAŞARISIZ")
                        
                        # 🔧 YENİ: CSV kaydı başarısız olursa fallback
                        logger.warning(f"⚠️ {symbol} için fallback CSV kaydı deneniyor...")
                        
                        # Manuel CSV kayıt denemesi
                        from data.database import log_trade_to_csv
                        manual_csv_data = {
                            "timestamp": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": symbol,
                            "side": position['side'],
                            "quantity": quantity,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "close_reason": f"{close_reason} - FALLBACK",
                            "status": "CLOSED"
                        }
                        
                        fallback_success = log_trade_to_csv(manual_csv_data)
                        
                        if fallback_success:
                            logger.info(f"✅ {symbol} Fallback CSV kaydı başarılı - Trade geçmişinde görünecek")
                            
                            # Pozisyonu temizle
                            if symbol in config.live_positions:
                                del config.live_positions[symbol]
                                sync_to_config()
                                logger.info(f"🗑️ {symbol} pozisyonu temizlendi (fallback)")
                        else:
                            logger.error(f"❌ {symbol} Fallback CSV kaydı da başarısız")
                            
                            # Son çare: REST API kontrolünü tetikle
                            logger.warning(f"🔄 {symbol} için REST API kontrolü tetikleniyor...")
                            try:
                                live_bot.check_filled_orders_rest()
                            except Exception as rest_err:
                                logger.error(f"❌ REST API kontrolü hatası: {rest_err}")
                
                except Exception as sltp_err:
                    logger.error(f"❌ {symbol} SL/TP işleme hatası: {sltp_err}")
                    
                    # Exception durumunda da pozisyonu temizle
                    if symbol in config.live_positions:
                        logger.warning(f"⚠️ {symbol} exception nedeniyle pozisyon temizleniyor")
                        del config.live_positions[symbol]
                        sync_to_config()
            
            # 🔧 YENİ: EXPIRED durumunda da kontrol et (SL otomatik tetiklenme)
            elif (order_status == 'EXPIRED' and 
                  order_type == 'STOP_MARKET' and 
                  symbol in config.live_positions):
                
                logger.info(f"⚠️ {symbol} Stop Loss EXPIRED - market emri takip ediliyor")
                # EXPIRED sonrası MARKET emrini bekleyeceğiz
            
            # 🔧 YENİ: MARKET emri FILLED ise ve önceden EXPIRED SL varsa
            elif (order_status == 'FILLED' and 
                  order_type == 'MARKET' and 
                  symbol in config.live_positions):
                
                logger.info(f"🔍 {symbol} MARKET emri FILLED - SL tetiklenme sonucu olabilir")
                
                # Bu MARKET emrinin SL tetiklenmesinden gelip gelmediğini kontrol et
                # Eğer config'de pozisyon varsa ve bu MARKET emri ise, büyük ihtimalle SL tetiklenmiştir
                try:
                    position = config.live_positions[symbol]
                    entry_price = position['entry_price']
                    exit_price = float(avg_price) if avg_price != '0' else entry_price
                    quantity = position['quantity']
                    
                    # P&L hesapla
                    if position['side'] == 'LONG':
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - exit_price) * quantity
                    
                    # Eğer zarar varsa büyük ihtimalle SL tetiklenmiştir
                    if pnl < 0:
                        close_reason = "Stop Loss - Market (Auto)"
                        logger.info(f"🛑 {symbol} SL tetiklenme tespit edildi - CSV'ye kaydediliyor")
                    else:
                        close_reason = "Manual Close - Market"
                        logger.info(f"📝 {symbol} Manuel kapatma tespit edildi")
                    
                    logger.info(f"💰 Entry: ${entry_price:.6f} → Exit: ${exit_price:.6f} | P&L: ${pnl:.4f}")
                    
                    # CSV'ye kaydet
                    csv_success = live_bot._save_closed_trade(symbol, exit_price, pnl, close_reason)
                    
                    if csv_success:
                        logger.info(f"✅ {symbol} MARKET CSV kaydı BAŞARILI - Trade geçmişinde görünecek")
                    else:
                        logger.error(f"❌ {symbol} MARKET CSV kaydı BAŞARISIZ")
                    
                    # Pozisyonu temizle
                    if symbol in config.live_positions:
                        del config.live_positions[symbol]
                        sync_to_config()
                        logger.info(f"🗑️ {symbol} pozisyonu config'den temizlendi")
                
                except Exception as market_err:
                    logger.error(f"❌ {symbol} MARKET işleme hatası: {market_err}")
            
            else:
                # Normal order update (NEW, PARTIALLY_FILLED, etc.)
                logger.debug(f"📊 Normal order update: {symbol} {order_type} {order_status}")
        
        elif msg_type == 'ACCOUNT_UPDATE':
            logger.debug("📊 WebSocket hesap güncellendi")
            
            # Account update durumunda da sync tetikle
            try:
                sync_to_config()
            except Exception as sync_err:
                logger.debug(f"Account update sync hatası: {sync_err}")
        
        else:
            logger.debug(f"📨 WebSocket diğer mesaj tipi: {msg_type}")
            
    except Exception as e:
        logger.error(f"❌ WebSocket mesaj işleme hatası: {str(e)}")
        logger.error(f"   Mesaj içeriği: {msg}")
        
        # Exception durumunda da temel bilgileri logla
        try:
            import traceback
            logger.debug(f"   WebSocket traceback: {traceback.format_exc()}")
        except:
            pass


def setup_smart_websocket():
    """🔧 DÜZELTME: AKILLI WebSocket + Binance gerçek pozisyon kontrolü"""
    global websocket_manager, websocket_active_symbols
    
    try:
        # 🔧 YENİ: Önce Binance'deki gerçek pozisyonları kontrol et
        binance_positions = set()
        if live_bot.client and live_bot.is_connected:
            try:
                positions = live_bot.client.futures_position_information()
                for pos in positions:
                    position_amt = float(pos['positionAmt'])
                    if abs(position_amt) > 0:
                        binance_positions.add(pos['symbol'])
                logger.debug(f"🔍 Binance gerçek pozisyonları: {binance_positions}")
            except Exception as e:
                logger.debug(f"⚠️ Binance pozisyon kontrolü hatası: {e}")
        
        # Config'deki pozisyonları al
        config_positions = set(config.live_positions.keys())
        logger.debug(f"🔍 Config pozisyonları: {config_positions}")
        
        # 🔧 YENİ: Birleştirilmiş pozisyon listesi (Binance + Config)
        all_positions = binance_positions.union(config_positions)
        logger.debug(f"🔍 Toplam izlenecek pozisyonlar: {all_positions}")
        
        # WebSocket'e ihtiyaç var mı?
        if not all_positions:
            # Hiç pozisyon yoksa WebSocket'i durdur
            if websocket_manager:
                stop_websocket()
            logger.debug("📱 Hiç pozisyon yok - WebSocket kapalı, sadece REST aktif")
            return False
        
        # 🔧 YENİ: Config'i Binance pozisyonlarıyla senkronize et
        for symbol in binance_positions:
            if symbol not in config_positions:
                logger.info(f"🔄 {symbol} Binance'de var ama config'de yok - senkronize ediliyor")
                # Binance pozisyonunu config'e ekle
                try:
                    binance_pos_data = next(pos for pos in positions if pos['symbol'] == symbol)
                    position_amt = float(binance_pos_data['positionAmt'])
                    entry_price = float(binance_pos_data['entryPrice'])
                    
                    config.live_positions[symbol] = {
                        'symbol': symbol,
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'quantity': abs(position_amt),
                        'entry_price': entry_price,
                        'auto_sltp': True,
                        'signal_data': {'ai_score': 50, 'run_type': 'long' if position_amt > 0 else 'short'}
                    }
                    logger.info(f"✅ {symbol} config'e eklendi")
                except Exception as sync_err:
                    logger.error(f"❌ {symbol} config senkronizasyon hatası: {sync_err}")
        
        # Semboller değişti mi?
        if all_positions == websocket_active_symbols and websocket_manager:
            logger.debug("🔄 WebSocket sembolleri değişmedi")
            return True
        
        # 🔧 ZORLA WebSocket kurulumu (pozisyon varsa mutlaka kur)
        logger.info(f"🔧 WebSocket kuruluyor - {len(all_positions)} pozisyon izlenecek")
        
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
        
        # 🔧 YENİ: Tüm pozisyonları aktif sembollere ekle
        websocket_active_symbols = all_positions
        
        logger.info(f"✅ WebSocket ZORLA aktif - {len(websocket_active_symbols)} pozisyon izleniyor")
        logger.info(f"🎯 WebSocket izlenen: {', '.join(websocket_active_symbols)}")
        
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
                recvWindow=60000,
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

                # 🔧 YENİ: Pozisyon açıldığında ZORLA WebSocket aktif et
                websocket_success = force_websocket_for_position(symbol)

                if websocket_success:
                    logger.info(f"🤖 {symbol} WebSocket izlemeye alındı")
                else:
                    logger.warning(f"⚠️ {symbol} WebSocket aktif edilemedi - sadece REST ile izlenecek")

                return True
            else:
                logger.error(f"❌ {symbol} market emir başarısız: {order_status}")
                return False

        except Exception as e:
            safe_symbol = signal.get("symbol") if isinstance(signal, dict) else "UNKNOWN"
            logger.error(f"❌ Pozisyon açma hatası {safe_symbol}: {e}")
            return False


    def close_position_manually(self, symbol: str, close_reason: str) -> bool:
        """🔧 DÜZELTME: Manuel pozisyon kapatma - trade geçmişine kaydet"""
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
                entry_price = position["entry_price"]
                
                # Kar/zarar hesaplama
                if position["side"] == "LONG":
                    pnl = (exit_price - entry_price) * position["quantity"]
                else:
                    pnl = (entry_price - exit_price) * position["quantity"]

                logger.info(f"✅ MANUEL KAPATMA: {symbol} | Sebep: {close_reason} | P&L: ${pnl:.4f}")

                # Trade geçmişine kaydet
                self._save_closed_trade(symbol, exit_price, pnl, close_reason)

                # Pozisyonu config'den sil
                del config.live_positions[symbol]
                sync_to_config()
                
                return True
            else:
                logger.error(f"❌ {symbol} manuel kapatma emri başarısız: {order_status}")
                return False

        except Exception as e:
            logger.error(f"❌ Manuel pozisyon kapatma hatası {symbol}: {e}")
            return False


    def _save_closed_trade(self, symbol: str, exit_price: float, pnl: float, close_reason: str) -> bool:
            """🔧 DÜZELTME: Kapatılan trade'i CSV'ye kaydet - HATA KORUNMALI + DOĞRULAMA"""
            try:
                logger.info(f"📝 CSV kayıt başlatılıyor: {symbol} | P&L: ${pnl:.4f} | Reason: {close_reason}")
                
                # 1. Position data kontrolü
                if symbol not in config.live_positions:
                    logger.warning(f"⚠️ {symbol} pozisyon bilgisi config'de bulunamadı - minimal kayıt oluşturulacak")
                    
                    # Minimal position data oluştur
                    current_price = get_current_price(symbol) or exit_price
                    
                    position_data = {
                        "symbol": symbol,
                        "side": "LONG" if pnl > 0 else "SHORT",  # Tahmin et
                        "quantity": 0.001,  # Default minimal
                        "entry_price": exit_price - (pnl / 0.001),  # Tersine hesapla
                        "invested_amount": abs(exit_price - (pnl / 0.001)) * 0.001,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "signal_data": {
                            "ai_score": 50.0,
                            "run_type": "long" if pnl > 0 else "short",
                            "run_count": 1,
                            "run_perc": 0.5,
                            "gauss_run": 1.0,
                            "vol_ratio": 1.0,
                            "deviso_ratio": 0.0
                        }
                    }
                else:
                    position_data = config.live_positions[symbol]
                
                # 2. CSV kayıt verisi hazırla
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
                    "commission": 0.0,  # Testnet'te komisyon yok
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
                
                # 3. Kayıt öncesi kontrol
                from data.database import load_trades_from_csv
                trades_before = load_trades_from_csv()
                records_before = len(trades_before)
                logger.debug(f"📊 Kayıt öncesi CSV satır sayısı: {records_before}")
                
                # 4. CSV'ye yazma işlemi
                write_success = log_trade_to_csv(csv_data)
                
                if not write_success:
                    logger.error(f"❌ {symbol} CSV yazma fonksiyonu FALSE döndü")
                    return False
                
                # 5. Yazma sonrası doğrulama
                time.sleep(0.5)  # Dosya yazma için kısa bekleme
                
                trades_after = load_trades_from_csv()
                records_after = len(trades_after)
                
                logger.debug(f"📊 Kayıt sonrası CSV satır sayısı: {records_after}")
                
                if records_after <= records_before:
                    logger.error(f"❌ {symbol} CSV satır sayısı artmadı! Öncesi: {records_before}, Sonrası: {records_after}")
                    return False
                
                # 6. Son kaydı doğrula
                if not trades_after.empty:
                    latest_record = trades_after.tail(1).iloc[0]
                    
                    if (latest_record.get('symbol') == symbol and 
                        close_reason in str(latest_record.get('close_reason', ''))):
                        
                        logger.info(f"✅ {symbol} CSV kaydı DOĞRULANDI: ${pnl:.4f} | {close_reason}")
                        logger.info(f"📈 Entry: ${position_data.get('entry_price', 0):.6f} → Exit: ${exit_price:.6f}")
                        logger.info(f"🔄 CSV: {records_before} → {records_after} (+{records_after - records_before} kayıt)")
                        
                        return True
                    else:
                        logger.error(f"❌ {symbol} Son kayıt doğrulanamadı - symbol veya close_reason eşleşmedi")
                        return False
                else:
                    logger.error(f"❌ {symbol} CSV boş veya okunamadı")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ {symbol} CSV kayıt hatası: {str(e)}")
                logger.error(f"   Exit Price: {exit_price}, P&L: {pnl}, Reason: {close_reason}")
                return False


    def check_filled_orders_rest(self) -> None:
        """🔧 DÜZELTME: REST API ile SL/TP kontrol + Binance pozisyon tarama + trade geçmişine kayıt"""
        try:
            # 🔧 YENİ: Önce Binance'deki TÜM pozisyonları kontrol et
            binance_positions = {}
            if self.client and self.is_connected:
                try:
                    all_binance_positions = self.client.futures_position_information()
                    for pos in all_binance_positions:
                        position_amt = float(pos['positionAmt'])
                        if abs(position_amt) > 0:
                            binance_positions[pos['symbol']] = pos
                    logger.debug(f"🔍 Binance'de aktif pozisyonlar: {list(binance_positions.keys())}")
                except Exception as e:
                    logger.debug(f"⚠️ Binance pozisyon tarama hatası: {e}")
            
            # Config pozisyonları + Binance pozisyonları
            current_positions = config.live_positions
            all_symbols_to_check = set(current_positions.keys()).union(set(binance_positions.keys()))
            symbols_to_process = []
            
            logger.debug(f"🔍 Kontrol edilecek semboller: {all_symbols_to_check}")
            
            for symbol in all_symbols_to_check:
                position = current_positions.get(symbol)
                
                # 🔧 YENİ: Config'de yoksa Binance pozisyonundan oluştur
                if not position and symbol in binance_positions:
                    binance_pos = binance_positions[symbol]
                    position_amt = float(binance_pos['positionAmt'])
                    entry_price = float(binance_pos['entryPrice'])
                    
                    # Geçici pozisyon verisi oluştur
                    position = {
                        'symbol': symbol,
                        'side': 'LONG' if position_amt > 0 else 'SHORT',
                        'quantity': abs(position_amt),
                        'entry_price': entry_price,
                        'auto_sltp': True,
                        'signal_data': {'ai_score': 50, 'run_type': 'long' if position_amt > 0 else 'short'},
                        'sl_order_id': None,
                        'tp_order_id': None
                    }
                    logger.info(f"🔄 {symbol} Binance pozisyonu geçici olarak oluşturuldu")
                
                if not position:
                    continue
                
                # 🔧 YENİ: WebSocket aktif olsa da REST kontrolü yap (double-check)
                # Eski kod: WebSocket varsa atla
                # Yeni kod: Her zaman kontrol et
                
                # Pozisyonun SL/TP emirleri var mı kontrol et
                sl_order_id = position.get("sl_order_id")
                tp_order_id = position.get("tp_order_id")
                
                # 🔧 YENİ: Emir ID'si yoksa Binance'den açık emirleri kontrol et
                if not sl_order_id and not tp_order_id:
                    try:
                        open_orders = self.client.futures_get_open_orders(symbol=symbol, recvWindow=60000)
                        for order in open_orders:
                            order_type = order.get('type')
                            if order_type == 'STOP_MARKET':
                                sl_order_id = order['orderId']
                                logger.debug(f"🔍 {symbol} SL emri bulundu: {sl_order_id}")
                            elif order_type == 'TAKE_PROFIT_MARKET':
                                tp_order_id = order['orderId'] 
                                logger.debug(f"🔍 {symbol} TP emri bulundu: {tp_order_id}")
                    except Exception as e:
                        logger.debug(f"⚠️ {symbol} açık emir kontrolü hatası: {e}")
                
                if not sl_order_id and not tp_order_id:
                    logger.debug(f"⚠️ {symbol} için SL/TP emri bulunamadı")
                    continue
                
                # SL kontrolü
                if sl_order_id:
                    try:
                        sl_order = self.client.futures_get_order(
                            symbol=symbol, orderId=sl_order_id, recvWindow=60000
                        )
                        if sl_order["status"] == "FILLED":
                            logger.info(f"🛑 {symbol} Stop Loss tetiklendi (REST)")
                            
                            # Entry price ile kar/zarar hesapla
                            entry_price = position['entry_price']
                            exit_price = float(sl_order.get("avgPrice", entry_price))
                            
                            if position['side'] == 'LONG':
                                pnl = (exit_price - entry_price) * position['quantity']
                            else:
                                pnl = (entry_price - exit_price) * position['quantity']
                            
                            logger.info(f"💰 Entry: ${entry_price:.6f} → Exit: ${exit_price:.6f} | P&L: ${pnl:.4f}")
                            
                            # Trade geçmişine kaydet
                            self._save_closed_trade(symbol, exit_price, pnl, "Stop Loss - REST")
                            
                            # Pozisyonu config'den sil
                            symbols_to_process.append(symbol)
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
                            
                            # Entry price ile kar/zarar hesapla
                            entry_price = position['entry_price']
                            exit_price = float(tp_order.get("avgPrice", entry_price))
                            
                            if position['side'] == 'LONG':
                                pnl = (exit_price - entry_price) * position['quantity']
                            else:
                                pnl = (entry_price - exit_price) * position['quantity']
                            
                            logger.info(f"💰 Entry: ${entry_price:.6f} → Exit: ${exit_price:.6f} | P&L: ${pnl:.4f}")
                            
                            # Trade geçmişine kaydet
                            self._save_closed_trade(symbol, exit_price, pnl, "Take Profit - REST")
                            
                            # Pozisyonu config'den sil
                            symbols_to_process.append(symbol)
                            continue
                    except Exception as e:
                        logger.debug(f"TP kontrol hatası {symbol}: {e}")
            
            # Kapatılan pozisyonları temizle
            for symbol in symbols_to_process:
                if symbol in config.live_positions:
                    del config.live_positions[symbol]
                    sync_to_config()
                    logger.info(f"✅ {symbol} pozisyonu panelden temizlendi")

        except Exception as e:
            logger.error(f"❌ REST emir kontrolü hatası: {e}")

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

            # 🔧 YENİ: Her döngüde WebSocket ZORLA kontrol et
            if current_positions:
                # Pozisyon varsa WebSocket mutlaka aktif olmalı
                if not websocket_manager or len(websocket_active_symbols) == 0:
                    logger.warning("⚠️ Pozisyon var ama WebSocket kapalı - ZORLA aktif ediliyor!")
                    
                    for symbol in current_positions.keys():
                        force_websocket_for_position(symbol)
                        time.sleep(1)  # Sembolleri teker teker ekle

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

            # 🔧 YENİ: WebSocket durumu detaylı log
            if current_positions:
                ws_count = len(websocket_active_symbols)
                logger.info(f"📊 Açık pozisyonlar: {', '.join(current_positions.keys())}")
                logger.info(f"🤖 WebSocket aktif semboller: {', '.join(websocket_active_symbols)}")
                
                if ws_count != len(current_positions):
                    logger.warning(f"⚠️ WebSocket/Pozisyon sayı uyumsuzluğu: WS={ws_count}, POS={len(current_positions)}")

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
    """🔧 DÜZELTME: Live trading durdur - ZORLA tüm pozisyonları kapat ve CSV'ye kaydet"""
    global live_trading_active, live_trading_thread

    with _cleanup_lock:
        if not live_trading_active:
            logger.info("💤 Live trading zaten durdurulmuş")
            return

        logger.info("🛑 Live Trading durduruluyor - TÜM POZİSYONLAR KAPANACAK...")
        live_trading_active = False

        # WebSocket durdur
        stop_websocket()

        # 🔧 KRITIK: ZORLA tüm Binance pozisyonlarını kapat
        successful_closes = 0
        failed_closes = 0
        
        try:
            if live_bot.client and live_bot.is_connected:
                logger.info("🔍 Binance TestNet'teki TÜM açık pozisyonlar kontrol ediliyor...")
                
                # Binance'den gerçek pozisyonları al
                all_positions = live_bot.client.futures_position_information()
                active_positions = []
                
                for pos in all_positions:
                    position_amt = float(pos['positionAmt'])
                    if abs(position_amt) > 0:  # Açık pozisyon var
                        active_positions.append(pos)
                        logger.info(f"🔍 Açık pozisyon bulundu: {pos['symbol']} - {position_amt}")
                
                if not active_positions:
                    logger.info("ℹ️ Kapatılacak pozisyon bulunamadı")
                else:
                    logger.info(f"🔒 {len(active_positions)} pozisyon ZORLA kapatılacak ve CSV'ye kaydedilecek")
                    
                    for pos in active_positions:
                        symbol = pos['symbol']
                        position_amt = float(pos['positionAmt'])
                        entry_price = float(pos['entryPrice'])
                        quantity = abs(position_amt)
                        side = 'LONG' if position_amt > 0 else 'SHORT'
                        
                        try:
                            logger.info(f"🔒 {symbol} kapatılıyor: {side} {quantity} @ {entry_price}")
                            
                            # 1. Önce SL/TP emirlerini iptal et
                            try:
                                open_orders = live_bot.client.futures_get_open_orders(symbol=symbol, recvWindow=60000)
                                for order in open_orders:
                                    order_type = order.get('type', '')
                                    if order_type in ['STOP_MARKET', 'TAKE_PROFIT_MARKET']:
                                        live_bot.client.futures_cancel_order(
                                            symbol=symbol,
                                            orderId=order['orderId'],
                                            recvWindow=60000
                                        )
                                        logger.info(f"🚫 {symbol} {order_type} emri iptal edildi: {order['orderId']}")
                            except Exception as cancel_err:
                                logger.debug(f"⚠️ {symbol} emir iptal hatası: {cancel_err}")
                            
                            # 2. Pozisyonu MARKET emriyle kapat
                            close_side = SIDE_SELL if side == 'LONG' else SIDE_BUY
                            
                            close_order = live_bot.client.futures_create_order(
                                symbol=symbol,
                                side=close_side,
                                type=ORDER_TYPE_MARKET,
                                quantity=quantity,
                                recvWindow=60000
                            )
                            
                            logger.info(f"📝 {symbol} kapatma emri verildi: {close_order['orderId']}")
                            
                            # 3. Emir kontrolü yap
                            time.sleep(2)  # Emir işlenmesi için bekle
                            
                            order_check = live_bot.client.futures_get_order(
                                symbol=symbol,
                                orderId=close_order['orderId'],
                                recvWindow=60000
                            )
                            
                            if order_check['status'] == 'FILLED':
                                exit_price = float(order_check.get('avgPrice', entry_price))
                                
                                # 4. P&L hesapla
                                if side == 'LONG':
                                    pnl = (exit_price - entry_price) * quantity
                                else:
                                    pnl = (entry_price - exit_price) * quantity
                                
                                logger.info(f"💰 {symbol} P&L: ${pnl:.4f} (Entry: {entry_price} → Exit: {exit_price})")
                                
                                # 5. CSV'ye kaydet - KRITIK KIŞIM
                                try:
                                    csv_success = live_bot._save_closed_trade(
                                        symbol, 
                                        exit_price, 
                                        pnl, 
                                        "Live Bot Stopped - Panel"
                                    )
                                    
                                    if csv_success:
                                        logger.info(f"✅ {symbol} CSV kaydı BAŞARILI - Trade geçmişinde görünecek")
                                        successful_closes += 1
                                    else:
                                        logger.error(f"❌ {symbol} CSV kaydı BAŞARISIZ")
                                        failed_closes += 1
                                        
                                except Exception as csv_err:
                                    logger.error(f"❌ {symbol} CSV kayıt hatası: {csv_err}")
                                    failed_closes += 1
                                
                            else:
                                logger.error(f"❌ {symbol} kapatma emri başarısız: {order_check['status']}")
                                failed_closes += 1
                            
                            time.sleep(1)  # Emirler arası güvenli bekle
                            
                        except Exception as close_err:
                            logger.error(f"❌ {symbol} pozisyon kapatma hatası: {close_err}")
                            failed_closes += 1
                
                # Sonuç özeti
                total_attempts = successful_closes + failed_closes
                if total_attempts > 0:
                    logger.info(f"📊 Pozisyon kapatma özeti: {successful_closes} başarılı, {failed_closes} başarısız")
                    if successful_closes > 0:
                        logger.info("✅ Trade geçmişi tablosunda görünecek")
                
            else:
                logger.warning("⚠️ Binance client bağlı değil - pozisyon kontrolü yapılamadı")
                
        except Exception as e:
            logger.error(f"❌ Pozisyon kapatma genel hatası: {e}")

        # Thread'i durdur
        if live_trading_thread is not None:
            try:
                live_trading_thread.join(timeout=10)  # Daha uzun timeout
            except Exception as e:
                logger.error(f"❌ Thread durdurma hatası: {e}")
            finally:
                live_trading_thread = None
        
        # Config'i temizle
        try:
            config.live_positions.clear()  # Tüm pozisyonları temizle
            sync_to_config()
        except Exception as e:
            logger.error(f"❌ Config temizleme hatası: {e}")
        
        # Client bağlantısını kapat
        try:
            if live_bot.client:
                live_bot.is_connected = False
                # Client'ı tamamen kapatma (çünkü tekrar başlatılabilir)
        except Exception as e:
            logger.debug(f"Client cleanup hatası: {e}")
        
        logger.info("🏁 Live Trading TAMAMEN durduruldu")
        if successful_closes > 0:
            logger.info("📋 Trade geçmişi tablosunu kontrol et - kapatılan pozisyonlar orada görünecek")



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
    

def force_websocket_for_position(symbol: str) -> bool:
    """🔧 YENİ: Pozisyon açıldığında ZORLA WebSocket aktif et"""
    global websocket_manager, websocket_active_symbols
    
    try:
        logger.info(f"🔧 {symbol} için ZORLA WebSocket aktif ediliyor...")
        
        # WebSocket yok veya kapalıysa başlat
        if not websocket_manager:
            logger.info("🚀 WebSocket manager başlatılıyor...")
            
            websocket_manager = ThreadedWebsocketManager(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_SECRET_KEY,
                testnet=(ENVIRONMENT == "testnet")
            )
            
            websocket_manager.start()
            time.sleep(2)  # Başlama zamanı ver
            
            # User socket başlat
            stream_name = websocket_manager.start_futures_user_socket(
                callback=handle_websocket_message
            )
            
            logger.info(f"✅ WebSocket user socket başlatıldı: {stream_name}")
        
        # Sembolü aktif listeye ekle
        websocket_active_symbols.add(symbol)
        
        logger.info(f"✅ {symbol} WebSocket aktif listesine eklendi")
        logger.info(f"🎯 Toplam WebSocket izlenen: {len(websocket_active_symbols)} sembol")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {symbol} ZORLA WebSocket aktif etme hatası: {e}")
        return False

