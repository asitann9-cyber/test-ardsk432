"""
🧪 Deviso Live Test - Demo Trading Bot
Timeframe adaptive demo trading sistemi (gerçek para kullanmaz)
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional

from config import (
    LOCAL_TZ, DEVISO_LIVE_TEST, DEVISO_TIMEFRAME_SIGNALS,
    get_strategy_from_timeframe, get_adaptive_ai_threshold, get_scan_interval_for_strategy,
    deviso_live_test_positions, current_data, current_settings
)
from data.fetch_data import get_current_price, get_usdt_perp_symbols
from trading.deviso_signals import (
    calculate_timeframe_adaptive_signals, select_top3_adaptive_signals,
    validate_signal_quality
)

logger = logging.getLogger("crypto-analytics")


class DevisoLiveTest:
    """🧪 Deviso Demo Trading Bot - Timeframe Adaptive"""
    
    def __init__(self):
        self.demo_balance = DEVISO_LIVE_TEST['demo_balance']
        self.max_trades = DEVISO_LIVE_TEST['max_trades']  # En iyi 3 kuralı
        self.positions = {}  # Demo pozisyonlar
        
        # Timeframe ayarları
        self.current_timeframe = '15m'
        self.current_strategy = 'swing'
        self.scan_interval = 60
        self.min_ai_score = 90
        
        # Bot durumu
        self.is_active = False
        self.trading_thread = None
        self.last_top3_selection = []
        
        # İstatistikler
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'timeframe_stats': {}
        }
        
        logger.info("🧪 Deviso Live Test bot oluşturuldu")
    
    def start(self) -> bool:
        """Demo trading bot'unu başlat"""
        try:
            if self.is_active:
                logger.warning("⚠️ Deviso Live Test zaten aktif")
                return False
            
            logger.info("🧪 Deviso Live Test başlatılıyor...")
            logger.info(f"💰 Demo bakiye: ${self.demo_balance:.2f}")
            logger.info(f"📊 Maksimum pozisyon: {self.max_trades} (En iyi 3)")
            logger.info(f"🎯 Timeframe: {self.current_timeframe} | Strateji: {self.current_strategy}")
            
            self.is_active = True
            
            # Trading thread'ini başlat
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info("✅ Deviso Live Test başarıyla başlatıldı")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deviso Live Test başlatma hatası: {e}")
            return False
    
    def stop(self):
        """Demo trading bot'unu durdur"""
        try:
            if not self.is_active:
                logger.info("ℹ️ Deviso Live Test zaten durdurulmuş")
                return
            
            logger.info("🛑 Deviso Live Test durduruluyor...")
            self.is_active = False
            
            # Açık pozisyonları kapat
            if self.positions:
                logger.info(f"📚 {len(self.positions)} demo pozisyon kapatılıyor...")
                for symbol in list(self.positions.keys()):
                    self._close_position(symbol, "Bot Durduruldu")
            
            logger.info("✅ Deviso Live Test durduruldu")
            
        except Exception as e:
            logger.error(f"❌ Deviso Live Test durdurma hatası: {e}")
    
    def update_timeframe_settings(self, timeframe: str, min_ai_score: float):
        """Timeframe ayarlarını güncelle"""
        try:
            old_strategy = self.current_strategy
            
            self.current_timeframe = timeframe
            self.current_strategy = get_strategy_from_timeframe(timeframe)
            self.scan_interval = get_scan_interval_for_strategy(self.current_strategy)
            self.min_ai_score = min_ai_score or get_adaptive_ai_threshold(self.current_strategy)
            
            logger.info(f"🎯 Deviso Live Test timeframe güncellendi:")
            logger.info(f"   📊 Timeframe: {timeframe}")
            logger.info(f"   🔄 Strateji: {old_strategy} → {self.current_strategy}")
            logger.info(f"   🤖 AI Eşiği: {self.min_ai_score}%")
            logger.info(f"   ⏱️ Tarama: {self.scan_interval}s")
            
        except Exception as e:
            logger.error(f"❌ Timeframe güncelleme hatası: {e}")
    
    def _trading_loop(self):
        """Ana demo trading döngüsü"""
        logger.info("🔄 Deviso Live Test döngüsü başlatıldı")
        loop_count = 0
        
        while self.is_active:
            try:
                loop_count += 1
                loop_start = time.time()
                
                logger.info(f"🧪 Demo döngü #{loop_count} - Pozisyon: {len(self.positions)}/{self.max_trades}")
                
                # Pozisyonları izle
                self._monitor_positions()
                
                # Boş slotları doldur
                self._fill_empty_positions()
                
                # İstatistikleri güncelle
                self._update_stats()
                
                loop_time = time.time() - loop_start
                logger.info(f"⏱️ Demo döngü #{loop_count}: {loop_time:.2f}s tamamlandı")
                
                if self.positions:
                    positions_summary = ", ".join(self.positions.keys())
                    logger.info(f"🧪 Demo pozisyonlar: {positions_summary}")
                
                # Scan interval kadar bekle
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"❌ Demo trading döngüsü hatası: {e}")
                time.sleep(30)
        
        logger.info("ℹ️ Deviso Live Test döngüsü sonlandırıldı")
    
    def _fill_empty_positions(self):
        """Boş pozisyon slotlarını doldur - En iyi 3 kuralı"""
        try:
            if not self.is_active:
                return
            
            current_positions = len(self.positions)
            if current_positions >= self.max_trades:
                return
            
            needed_slots = self.max_trades - current_positions
            
            # Mevcut sinyalleri kullan (current_data)
            if current_data is None or current_data.empty:
                logger.debug("📊 Demo: Sinyal verisi yok")
                return
            
            # Filtreleme ve adaptive analiz
            df = current_data.copy()
            
            # AI skoru 0-1 ise 0-100'e çevir
            try:
                if df["ai_score"].max() <= 1.0:
                    df["ai_score"] = (df["ai_score"] * 100.0).clip(0, 100)
            except Exception:
                pass
            
            # Demo için daha gevşek filtreler
            min_ai = max(70, self.min_ai_score - 10)  # Demo için %10 daha düşük eşik
            min_streak = int(current_settings.get('min_streak', 3))
            min_move = float(current_settings.get('min_pct', 0.5))
            min_volr = float(current_settings.get('min_volr', 1.5))
            
            logger.info(f"🧮 Demo filtre: AI≥{min_ai}%, streak≥{min_streak}, move≥{min_move}%")
            
            df = df[
                (df['ai_score'] >= min_ai) &
                (df['run_count'] >= min_streak) &
                (df['run_perc'] >= min_move)
            ]
            
            if 'vol_ratio' in df.columns:
                df = df[df['vol_ratio'].fillna(0) >= min_volr]
            
            # Zaten açık pozisyonları hariç tut
            exclude_symbols = set(self.positions.keys())
            df = df[~df['symbol'].isin(exclude_symbols)]
            
            if df.empty:
                logger.debug("🧪 Demo: Uygun sinyal yok")
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
            
            logger.info("🎯 Demo En iyi 3 aday:")
            for _, r in top3.iterrows():
                logger.info(f"   🧪 {r['symbol']} | AI={r['ai_score']:.0f}% | {r['run_type']} | move={r['run_perc']:.2f}%")
            
            # Pozisyonları aç
            opened = 0
            for _, row in top3.iterrows():
                if opened >= needed_slots:
                    break
                if self._open_position(row.to_dict()):
                    opened += 1
            
            if opened > 0:
                logger.info(f"🧪 {opened} yeni demo pozisyon açıldı")
            
        except Exception as e:
            logger.error(f"❌ Demo pozisyon doldurma hatası: {e}")
    
    def _open_position(self, signal: Dict) -> bool:
        """Demo pozisyon aç"""
        try:
            symbol = signal['symbol']
            side = signal['run_type'].upper()
            
            if symbol in self.positions:
                logger.warning(f"⚠️ {symbol} için zaten demo pozisyon var")
                return False
            
            # Mevcut fiyat
            current_price = get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ {symbol} demo fiyat alınamadı")
                return False
            
            # Pozisyon büyüklüğü (strateji adaptive)
            position_size_pct = DEVISO_LIVE_TEST['position_size_adaptive'][self.current_strategy]
            position_value = self.demo_balance * position_size_pct
            quantity = position_value / current_price
            
            # SL/TP hesapla (strateji adaptive)
            if self.current_strategy == 'scalping':
                sl_pct, tp_pct = 0.015, 0.03  # %1.5 SL, %3 TP
            elif self.current_strategy == 'swing':
                sl_pct, tp_pct = 0.02, 0.04   # %2 SL, %4 TP
            else:  # position
                sl_pct, tp_pct = 0.03, 0.06   # %3 SL, %6 TP
            
            if side == 'LONG':
                stop_loss = current_price * (1 - sl_pct)
                take_profit = current_price * (1 + tp_pct)
            else:
                stop_loss = current_price * (1 + sl_pct)
                take_profit = current_price * (1 - tp_pct)
            
            # Pozisyon verisi
            position_data = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': current_price,
                'invested_amount': position_value,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_time': datetime.now(LOCAL_TZ),
                'signal_data': signal,
                'strategy': self.current_strategy,
                'timeframe': self.current_timeframe
            }
            
            self.positions[symbol] = position_data
            
            logger.info(f"✅ DEMO POZİSYON AÇILDI: {symbol} {side}")
            logger.info(f"💰 Demo yatırım: ${position_value:.2f} @ ${current_price:.6f}")
            logger.info(f"📊 SL: ${stop_loss:.6f} | TP: ${take_profit:.6f} | AI: {signal['ai_score']:.0f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo pozisyon açma hatası {symbol}: {e}")
            return False
    
    def _close_position(self, symbol: str, close_reason: str) -> bool:
        """Demo pozisyon kapat"""
        try:
            if symbol not in self.positions:
                logger.warning(f"⚠️ {symbol} demo pozisyonu bulunamadı")
                return False
            
            position = self.positions[symbol]
            
            # Çıkış fiyatı
            current_price = get_current_price(symbol)
            if not current_price:
                logger.error(f"❌ {symbol} demo çıkış fiyatı alınamadı")
                return False
            
            # P&L hesapla
            if position['side'] == 'LONG':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # Demo bakiyeye ekle/çıkar
            self.demo_balance += pnl
            
            # İstatistikleri güncelle
            self.stats['total_trades'] += 1
            self.stats['total_pnl'] += pnl
            
            if pnl > 0:
                self.stats['winning_trades'] += 1
                self.stats['best_trade'] = max(self.stats['best_trade'], pnl)
            else:
                self.stats['worst_trade'] = min(self.stats['worst_trade'], pnl)
            
            logger.info(f"✅ DEMO POZİSYON KAPANDI: {symbol} {position['side']}")
            logger.info(f"💲 Demo giriş: ${position['entry_price']:.6f} → çıkış: ${current_price:.6f}")
            logger.info(f"💰 Demo P&L: ${pnl:.4f} | Yeni bakiye: ${self.demo_balance:.2f} | Sebep: {close_reason}")
            
            # Pozisyonu sil
            del self.positions[symbol]
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo pozisyon kapatma hatası {symbol}: {e}")
            return False
    
    def _monitor_positions(self):
        """Demo pozisyonları izle"""
        try:
            if not self.positions:
                return
            
            logger.debug(f"👀 {len(self.positions)} demo pozisyon izleniyor...")
            to_close = []
            
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                
                current_price = get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"⚠️ {symbol} demo fiyat alınamadı")
                    continue
                
                should_close = False
                reason = ""
                
                # SL/TP kontrolü
                if position['side'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        reason = "Demo Stop Loss"
                    elif current_price >= position['take_profit']:
                        should_close = True
                        reason = "Demo Take Profit"
                else:  # SHORT
                    if current_price >= position['stop_loss']:
                        should_close = True
                        reason = "Demo Stop Loss"
                    elif current_price <= position['take_profit']:
                        should_close = True
                        reason = "Demo Take Profit"
                
                if should_close:
                    if self._close_position(symbol, reason):
                        to_close.append(symbol)
            
            if to_close:
                logger.info(f"🔄 Demo kapanan pozisyonlar: {to_close}")
                
        except Exception as e:
            logger.error(f"❌ Demo pozisyon izleme hatası: {e}")
    
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
            logger.debug(f"Demo istatistik güncelleme hatası: {e}")
    
    def get_status(self) -> Dict:
        """Demo bot durumunu al"""
        return {
            'is_active': self.is_active,
            'demo_balance': self.demo_balance,
            'current_timeframe': self.current_timeframe,
            'current_strategy': self.current_strategy,
            'open_positions': len(self.positions),
            'max_positions': self.max_trades,
            'stats': self.stats,
            'last_top3': self.last_top3_selection
        }
    
    def get_last_top3_selection(self) -> List[Dict]:
        """Son en iyi 3 seçimini al"""
        return self.last_top3_selection
    
    def get_timeframe_stats(self) -> Dict:
        """Timeframe bazlı istatistikleri al"""
        return self.stats.get('timeframe_stats', {})