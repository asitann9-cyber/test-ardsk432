"""
💰 Deviso Live Trading - Gerçek Para Trading Bot
Timeframe adaptive gerçek trading sistemi (Binance Futures API)
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal, ROUND_DOWN

try:
    from binance.client import Client
    from binance.enums import *
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY, ENVIRONMENT, LOCAL_TZ,
    DEVISO_LIVE_TRADING, DEVISO_TIMEFRAME_SIGNALS,
    get_strategy_from_timeframe, get_adaptive_ai_threshold, get_scan_interval_for_strategy,
    deviso_live_trading_positions, current_data, current_settings
)
from data.fetch_data import get_current_price
from trading.deviso_signals import (
    calculate_timeframe_adaptive_signals, select_top3_adaptive_signals,
    validate_signal_quality
)

logger = logging.getLogger("crypto-analytics")


class DevisoLiveTrading:
    """💰 Deviso Gerçek Trading Bot - Timeframe Adaptive"""
    
    def __init__(self):
        self.client = None
        self.futures_balance = DEVISO_LIVE_TRADING['futures_balance']
        self.max_trades = 3  # En iyi 3 kuralı
        self.positions = {}  # Gerçek pozisyonlar
        
        # Timeframe ayarları
        self.current_timeframe = '15m'
        self.current_strategy = 'swing'
        self.scan_interval = 60
        self.min_ai_score = 90
        
        # Risk yönetimi
        self.max_daily_loss = 0.1  # %10 günlük maksimum kayıp
        self.daily_pnl = 0.0
        
        # Bot durumu
        self.is_active = False
        self.trading_thread = None
        self.last_top3_selection = []
        self.tradable_cache = set()
        
        # İstatistikler
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'timeframe_stats': {},
            'daily_trades': 0
        }
        
        logger.info("💰 Deviso Live Trading bot oluşturuldu")
    
    def start(self) -> bool:
        """Gerçek trading bot'unu başlat"""
        try:
            if not BINANCE_AVAILABLE:
                logger.error("❌ python-binance kütüphanesi yüklü değil")
                return False
            
            if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
                logger.error("❌ Binance API anahtarları .env dosyasında bulunamadı")
                return False
            
            if self.is_active:
                logger.warning("⚠️ Deviso Live Trading zaten aktif")
                return False
            
            # Binance bağlantısı
            if not self._connect_to_binance():
                return False
            
            logger.info("💰 Deviso Live Trading başlatılıyor...")
            logger.info(f"🏦 Futures bakiye: ${self.futures_balance:.2f}")
            logger.info(f"📊 Maksimum pozisyon: {self.max_trades} (En iyi 3)")
            logger.info(f"🎯 Timeframe: {self.current_timeframe} | Strateji: {self.current_strategy}")
            logger.warning("⚠️ GERÇEK PARA ile işlem yapılacak!")
            
            self.is_active = True
            
            # Trading thread'ini başlat
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info("✅ Deviso Live Trading başarıyla başlatıldı")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deviso Live Trading başlatma hatası: {e}")
            return False
    
    def stop(self):
        """Gerçek trading bot'unu durdur"""
        try:
            if not self.is_active:
                logger.info("ℹ️ Deviso Live Trading zaten durdurulmuş")
                return
            
            logger.info("🛑 Deviso Live Trading durduruluyor...")
            self.is_active = False
            
            # Açık pozisyonları kapat
            if self.positions:
                logger.info(f"📚 {len(self.positions)} gerçek pozisyon kapatılıyor...")
                for symbol in list(self.positions.keys()):
                    self._close_position(symbol, "Bot Durduruldu")
                    time.sleep(0.5)  # Rate limit
            
            logger.info("✅ Deviso Live Trading durduruldu")
            
        except Exception as e:
            logger.error(f"❌ Deviso Live Trading durdurma hatası: {e}")
    
    def _connect_to_binance(self) -> bool:
        """Binance API'ye bağlan"""
        try:
            if ENVIRONMENT == "testnet":
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                    testnet=True,
                )
                logger.info("🧪 Binance Futures Testnet'e bağlanıldı")
            else:
                self.client = Client(
                    api_key=BINANCE_API_KEY,
                    api_secret=BINANCE_SECRET_KEY,
                )
                logger.info("🚀 Binance Mainnet'e bağlanıldı")
            
            # API test
            self.client.futures_ping()
            account_info = self.client.futures_account(recvWindow=60000)
            self.futures_balance = float(account_info["totalWalletBalance"])
            
            logger.info(f"✅ Deviso API bağlantısı başarılı - Bakiye: ${self.futures_balance:.2f}")
            
            # Tradable sembolleri önbelleğe al
            self._refresh_tradable_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deviso Binance bağlantı hatası: {e}")
            return False
    
    def _refresh_tradable_cache(self):
        """Tradable sembolleri güncelle"""
        try:
            tickers = self.client.futures_symbol_ticker()
            self.tradable_cache = {t["symbol"] for t in tickers if t.get("symbol", "").endswith("USDT")}
            logger.info(f"🧭 Deviso TRADABLE semboller: {len(self.tradable_cache)} adet")
        except Exception as e:
            logger.warning(f"⚠️ Deviso tradable sembol keşfi başarısız: {e}")
    
    def update_timeframe_settings(self, timeframe: str, min_ai_score: float):
        """Timeframe ayarlarını güncelle"""
        try:
            old_strategy = self.current_strategy
            
            self.current_timeframe = timeframe
            self.current_strategy = get_strategy_from_timeframe(timeframe)
            self.scan_interval = get_scan_interval_for_strategy(self.current_strategy)
            self.min_ai_score = min_ai_score or get_adaptive_ai_threshold(self.current_strategy)
            
            logger.info(f"🎯 Deviso Live Trading timeframe güncellendi:")
            logger.info(f"   📊 Timeframe: {timeframe}")
            logger.info(f"   🔄 Strateji: {old_strategy} → {self.current_strategy}")
            logger.info(f"   🤖 AI Eşiği: {self.min_ai_score}%")
            logger.info(f"   ⏱️ Tarama: {self.scan_interval}s")
            
        except Exception as e:
            logger.error(f"❌ Deviso timeframe güncelleme hatası: {e}")
    
    def _trading_loop(self):
        """Ana gerçek trading döngüsü"""
        logger.info("🔄 Deviso Live Trading döngüsü başlatıldı")
        loop_count = 0
        
        while self.is_active:
            try:
                loop_count += 1
                loop_start = time.time()
                
                logger.info(f"💰 Live döngü #{loop_count} - Pozisyon: {len(self.positions)}/{self.max_trades}")
                
                # Günlük kayıp kontrolü
                if self.daily_pnl <= -(self.futures_balance * self.max_daily_loss):
                    logger.warning(f"⚠️ Günlük maksimum kayıp limitine ulaşıldı: ${self.daily_pnl:.2f}")
                    self.stop()
                    break
                
                # Bakiye güncelle
                self._update_balance()
                
                # Pozisyonları izle
                self._monitor_positions()
                
                # Boş slotları doldur
                self._fill_empty_positions()
                
                # İstatistikleri güncelle
                self._update_stats()
                
                loop_time = time.time() - loop_start
                logger.info(f"⏱️ Live döngü #{loop_count}: {loop_time:.2f}s tamamlandı")
                
                if self.positions:
                    positions_summary = ", ".join(self.positions.keys())
                    logger.info(f"💰 Live pozisyonlar: {positions_summary}")
                
                # Scan interval kadar bekle
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"❌ Deviso Live trading döngüsü hatası: {e}")
                time.sleep(30)
        
        logger.info("ℹ️ Deviso Live Trading döngüsü sonlandırıldı")
    
    def _update_balance(self):
        """Bakiyeyi güncelle"""
        try:
            account_info = self.client.futures_account(recvWindow=60000)
            self.futures_balance = float(account_info["totalWalletBalance"])
        except Exception as e:
            logger.debug(f"Deviso bakiye güncelleme hatası: {e}")
    
    def _fill_empty_positions(self):
        """Boş pozisyon slotlarını doldur - En iyi 3 kuralı"""
        try:
            if not self.is_active:
                return
            
            current_positions = len(self.positions)
            if current_positions >= self.max_trades:
                return
            
            needed_slots = self.max_trades - current_positions
            
            # Mevcut sinyalleri kullan
            if current_data is None or current_data.empty:
                logger.debug("📊 Live: Sinyal verisi yok")
                return
            
            df = current_data.copy()
            
            # AI skoru dönüşümü
            try:
                if df["ai_score"].max() <= 1.0:
                    df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
            except Exception:
                pass
            
            # Gerçek trading için sıkı filtreler
            min_ai = max(85, self.min_ai_score)  # Minimum %85
            min_streak = int(current_settings.get('min_streak', 4))  # Daha sıkı
            min_move = float(current_settings.get('min_pct', 1.0))   # Daha sıkı
            min_volr = float(current_settings.get('min_volr', 2.0))   # Daha sıkı
            
            logger.info(f"🧮 Live filtre: AI≥{min_ai}%, streak≥{min_streak}, move≥{min_move}%")
            
            df = df[
                (df['ai_score'] >= min_ai) &
                (df['run_count'] >= min_streak) &
                (df['run_perc'] >= min_move)
            ]
            
            if 'vol_ratio' in df.columns:
                df = df[df['vol_ratio'].fillna(0) >= min_volr]
            
            # Tradable cache filtresi
            if self.tradable_cache:
                before_tr = len(df)
                df = df[df['symbol'].isin(self.tradable_cache)]
                logger.info(f"🧮 Tradable filtresi: {len(df)} (önce: {before_tr})")
            
            # Zaten açık pozisyonları hariç tut
            exclude_symbols = set(self.positions.keys())
            df = df[~df['symbol'].isin(exclude_symbols)]
            
            if df.empty:
                logger.debug("💰 Live: Uygun sinyal yok")
                return
            
            # En iyi 3 seçimi
            df = df.sort_values(['ai_score', 'run_perc', 'gauss_run'], ascending=[False, False, False])
            top3 = df.head(3)
            
            # Son top3 seçimini kaydet
            self.last_top3_selection = []
            for _, row in top3.iterrows():
                self.last_top3_selection.append({
                    'symbol': row['symbol'],
                    'ai_score': row['ai_score'],
                    'strategy': self.current_strategy,
                    'timeframe': self.current_timeframe
                })
            
            logger.info("🎯 Live En iyi 3 aday:")
            for _, r in top3.iterrows():
                logger.info(f"   💰 {r['symbol']} | AI={r['ai_score']:.0f}% | {r['run_type']} | move={r['run_perc']:.2f}%")
            
            # Pozisyonları aç
            opened = 0
            for _, row in top3.iterrows():
                if opened >= needed_slots:
                    break
                if self._open_position(row.to_dict()):
                    opened += 1
                    time.sleep(1)  # Rate limit
            
            if opened > 0:
                logger.info(f"💰 {opened} yeni live pozisyon açıldı")
            
        except Exception as e:
            logger.error(f"❌ Live pozisyon doldurma hatası: {e}")
    
    def _open_position(self, signal: Dict) -> bool:
        """Gerçek pozisyon aç"""
        try:
            symbol = signal['symbol']
            side_txt = signal['run_type'].upper()
            
            if symbol in self.positions:
                logger.warning(f"⚠️ {symbol} için zaten live pozisyon var")
                return False
            
            # Fiyat ve miktar hesaplama
            current_price = get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ {symbol} live fiyat alınamadı")
                return False
            
            # Risk adaptive pozisyon büyüklüğü
            risk_pct = DEVISO_LIVE_TRADING['risk_per_trade_adaptive'][self.current_strategy]
            position_value = self.futures_balance * risk_pct
            
            # Minimum order kontrolü
            if position_value < 10:  # Min $10
                logger.warning(f"⚠️ {symbol}: Pozisyon değeri çok düşük: ${position_value:.2f}")
                return False
            
            # Leverage adaptive
            leverage = DEVISO_LIVE_TRADING['leverage_adaptive'][self.current_strategy]
            quantity = (position_value * leverage) / current_price
            
            # Lot size uyumu (basitleştirilmiş)
            quantity = round(quantity, 6)
            
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
                
                # SL/TP hesapla (strateji adaptive)
                if self.current_strategy == 'scalping':
                    sl_pct, tp_pct = 0.015, 0.03  # %1.5 SL, %3 TP
                elif self.current_strategy == 'swing':
                    sl_pct, tp_pct = 0.02, 0.04   # %2 SL, %4 TP
                else:  # position
                    sl_pct, tp_pct = 0.03, 0.06   # %3 SL, %6 TP
                
                if side_txt == 'LONG':
                    stop_loss = avg_price * (1 - sl_pct)
                    take_profit = avg_price * (1 + tp_pct)
                else:
                    stop_loss = avg_price * (1 + sl_pct)
                    take_profit = avg_price * (1 - tp_pct)
                
                position_data = {
                    'symbol': symbol,
                    'side': side_txt,
                    'quantity': float(order["executedQty"]),
                    'entry_price': avg_price,
                    'invested_amount': float(order["executedQty"]) * avg_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now(LOCAL_TZ),
                    'signal_data': signal,
                    'order_id': order["orderId"],
                    'strategy': self.current_strategy,
                    'timeframe': self.current_timeframe,
                    'leverage': leverage
                }
                
                self.positions[symbol] = position_data
                
                logger.info(f"✅ LIVE POZİSYON AÇILDI: {symbol} {side_txt} {quantity} @ ${avg_price:.6f}")
                logger.info(f"💰 Yatırılan: ${position_data['invested_amount']:.2f} | Leverage: {leverage}x")
                logger.info(f"📊 SL: ${stop_loss:.6f} | TP: ${take_profit:.6f} | AI: {signal['ai_score']:.0f}%")
                
                return True
            
            logger.error(f"❌ Live order başarısız: {order}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Live pozisyon açma hatası {symbol}: {e}")
            return False
    
    def _close_position(self, symbol: str, close_reason: str) -> bool:
        """Gerçek pozisyon kapat"""
        try:
            if symbol not in self.positions:
                logger.warning(f"⚠️ {symbol} live pozisyonu bulunamadı")
                return False
            
            position = self.positions[symbol]
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
                
                # İstatistikleri güncelle
                self.stats['total_trades'] += 1
                self.stats['total_pnl'] += pnl
                self.daily_pnl += pnl
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                    self.stats['best_trade'] = max(self.stats['best_trade'], pnl)
                else:
                    self.stats['worst_trade'] = min(self.stats['worst_trade'], pnl)
                
                logger.info(f"✅ LIVE POZİSYON KAPANDI: {symbol} {position['side']} | Sebep: {close_reason}")
                logger.info(f"💲 Giriş: ${position['entry_price']:.6f} → Çıkış: ${exit_price:.6f} | P&L: ${pnl:.4f}")
                
                del self.positions[symbol]
                return True
            
            logger.error(f"❌ Live pozisyon kapatma başarısız: {order}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Live pozisyon kapatma hatası {symbol}: {e}")
            return False
    
    def _monitor_positions(self):
        """Gerçek pozisyonları izle"""
        try:
            if not self.positions:
                return
            
            logger.debug(f"👀 {len(self.positions)} live pozisyon izleniyor...")
            to_close = []
            
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                
                current_price = get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"⚠️ {symbol} live fiyat alınamadı")
                    continue
                
                should_close = False
                reason = ""
                
                # SL/TP kontrolü
                if position['side'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price >= position['take_profit']:
                        should_close = True
                        reason = "Take Profit"
                else:  # SHORT
                    if current_price >= position['stop_loss']:
                        should_close = True
                        reason = "Stop Loss"
                    elif current_price <= position['take_profit']:
                        should_close = True
                        reason = "Take Profit"
                
                if should_close:
                    if self._close_position(symbol, reason):
                        to_close.append(symbol)
            
            if to_close:
                logger.info(f"🔄 Live kapanan pozisyonlar: {to_close}")
                
        except Exception as e:
            logger.error(f"❌ Live pozisyon izleme hatası: {e}")
    
    def _update_stats(self):
        """İstatistikleri güncelle"""
        try:
            # Timeframe bazlı istatistikler
            tf_key = f"{self.current_timeframe}_{self.current_strategy}"
            if tf_key not in self.stats['timeframe_stats']:
                self.stats['timeframe_stats'][tf_key] = {
                    'trades': 0,
                    'wins': 0,
                    'pnl': 0.0
                }
            
        except Exception as e:
            logger.debug(f"Live istatistik güncelleme hatası: {e}")
    
    def get_status(self) -> Dict:
        """Live bot durumunu al"""
        return {
            'is_active': self.is_active,
            'futures_balance': self.futures_balance,
            'current_timeframe': self.current_timeframe,
            'current_strategy': self.current_strategy,
            'open_positions': len(self.positions),
            'max_positions': self.max_trades,
            'daily_pnl': self.daily_pnl,
            'stats': self.stats,
            'last_top3': self.last_top3_selection
        }
    
    def get_last_top3_selection(self) -> List[Dict]:
        """Son en iyi 3 seçimini al"""
        return self.last_top3_selection
    
    def get_timeframe_stats(self) -> Dict:
        """Timeframe bazlı istatistikleri al"""
        return self.stats.get('timeframe_stats', {})