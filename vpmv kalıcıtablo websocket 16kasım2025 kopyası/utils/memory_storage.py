"""
Memory Storage Service
Bellekte veri depolama ve yönetim işlemleri
🆕 YENİ: VPMV (Volume-Price-Momentum-Volatility) NET POWER desteği
✅ YENİ: TETİKLEYİCİ SİSTEMİ - Momentum, Hacim, Volatilite tetikleyicileri
✅ Dinamik C-Signal ±X Threshold - Panel'den Ayarlanabilir L/S Sinyal Tespiti
✅ FIX: max_ratio_percent → ratio_percent isim değişikliği
🐛 BUG FIX: last_c_signal_alert_time spam önleme sorunu düzeltildi
✅ CRITICAL FIX: Thread-safe locks eklendi - Race condition koruması
"""

import logging
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryStorage:
    """
    Bellekte veri depolama sınıfı
    ✅ Thread-safe - Tüm okuma/yazma işlemleri korunuyor
    """
    
    def __init__(self):
        """Memory storage'ı başlat"""
        self.selected_symbols: List[str] = []
        self.permanent_high_ratio: List[Dict[str, Any]] = []
        self.analysis_cache: Dict[str, Any] = {}
        self.system_stats: Dict[str, Any] = {
            'total_analyses': 0,
            'last_analysis_time': None,
            'telegram_alerts_sent': 0,
            'c_signal_alerts_sent': 0
        }
        
        # ✅ YENİ: Thread-safety için locks
        self._selected_symbols_lock = threading.Lock()
        self._permanent_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._stats_lock = threading.Lock()
    
    # =====================================================
    # SELECTED SYMBOLS MANAGEMENT (Thread-safe)
    # =====================================================
    
    def get_selected_symbols(self) -> List[str]:
        """Seçili sembolleri getir (Thread-safe)"""
        with self._selected_symbols_lock:
            return self.selected_symbols.copy()
    
    def save_selected_symbols(self, symbols: List[str]) -> None:
        """Seçili sembolleri belleğe kaydet (Thread-safe)"""
        with self._selected_symbols_lock:
            self.selected_symbols = symbols.copy()
            logger.info(f"{len(symbols)} sembol belleğe kaydedildi")
    
    def add_selected_symbols(self, symbols: List[str]) -> List[str]:
        """Yeni semboller ekle (Thread-safe)"""
        with self._selected_symbols_lock:
            current_symbols = set(self.selected_symbols)
            new_symbols = set(symbols)
            all_symbols = list(current_symbols | new_symbols)
            self.selected_symbols = all_symbols
            
            added_count = len(new_symbols - current_symbols)
            logger.info(f"{added_count} yeni sembol eklendi, toplam: {len(all_symbols)}")
            
            return all_symbols
    
    def remove_selected_symbol(self, symbol: str) -> List[str]:
        """Belirli sembolü sil (Thread-safe)"""
        with self._selected_symbols_lock:
            if symbol in self.selected_symbols:
                self.selected_symbols.remove(symbol)
                logger.info(f"{symbol} sembolü silindi")
            
            return self.selected_symbols.copy()
    
    def clear_selected_symbols(self) -> None:
        """Tüm seçili sembolleri temizle (Thread-safe)"""
        with self._selected_symbols_lock:
            count = len(self.selected_symbols)
            self.selected_symbols = []
            logger.info(f"{count} sembol temizlendi")
    
    def is_symbol_selected(self, symbol: str) -> bool:
        """Sembol seçili mi kontrol et (Thread-safe)"""
        with self._selected_symbols_lock:
            return symbol in self.selected_symbols
    
    # =====================================================
    # PERMANENT HIGH RATIO MANAGEMENT (Thread-safe) - ✅ VPMV + TETİKLEYİCİ ENTEGRASYONU
    # =====================================================
    
    def add_permanent_symbol(self, symbol_data: Dict[str, Any]) -> None:
        """Kalıcı listeye sembol ekle (VPMV + Tetikleyici dahil) (Thread-safe)"""
        symbol = symbol_data.get('symbol')
        if not symbol:
            logger.warning("Sembol adı bulunamadı, kalıcı listeye eklenemedi")
            return
        
        with self._permanent_lock:
            # Mevcut sembol kontrolü
            existing_symbol = self._get_permanent_symbol_unsafe(symbol)
            
            if not existing_symbol:
                # Yeni sembol - kalıcı listeye ekle
                ratio_percent = abs(symbol_data.get('ratio_percent', 0))
                z_score = abs(symbol_data.get('z_score', 0))
                
                # Ekleme nedenini belirle
                add_reasons = []
                if ratio_percent >= 100.0:
                    add_reasons.append(f"{ratio_percent:.2f}% ratio")
                if z_score >= 2.0:
                    add_reasons.append(f"Z-Score: {z_score:.2f}")
                
                permanent_entry = {
                    'symbol': symbol,
                    'first_high_ratio_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    # ✅ Supertrend verileri
                    'ratio_percent': symbol_data.get('ratio_percent', 0),
                    'supertrend_type': symbol_data.get('trend_direction', 'None'),
                    'z_score': symbol_data.get('z_score', 0),
                    'abs_ratio_percent': abs(symbol_data.get('ratio_percent', 0)),
                    'timeframe': symbol_data.get('timeframe', '4h'),
                    'tradingview_link': symbol_data.get('tradingview_link', '#'),
                    # C-Signal verileri
                    'c_signal': None,
                    'c_signal_update_time': None,
                    # 🆕 VPMV verileri
                    'vpmv_net_power': symbol_data.get('vpmv_net_power', 0),
                    'vpmv_signal': symbol_data.get('vpmv_signal', 'NEUTRAL'),
                    'vpmv_update_time': datetime.now().strftime('%H:%M'),
                    # ✅ YENİ: TETİKLEYİCİ ALANLARI
                    'vpmv_trigger_name': symbol_data.get('vpmv_trigger_name', 'Yok'),
                    'vpmv_trigger_active': symbol_data.get('vpmv_trigger_active', False),
                    # Diğer alanlar
                    'add_reason': " + ".join(add_reasons) if add_reasons else "Manuel ekleme",
                    'last_telegram_alert': None,
                    # MANUEL TÜR DEĞİŞTİRME ALANLARI
                    'manual_type_override': False,
                    'manual_type_value': None,
                    'manual_override_date': None,
                    # C-SIGNAL TARİHÇE ALANLARI
                    'last_c_signal_value': None,
                    'last_c_signal_type': None,
                    'last_c_signal_alert_time': None,
                    'c_signal_history': []
                }
                
                self.permanent_high_ratio.append(permanent_entry)
                logger.info(f"🎯 {symbol} kalıcı listeye eklendi ({permanent_entry['add_reason']}) - VPMV: {symbol_data.get('vpmv_net_power', 0)} - Tetikleyici: {symbol_data.get('vpmv_trigger_name', 'Yok')}")
            else:
                # ✅ MEVCUT SEMBOL - Güncel değerleri güncelle (VPMV + Tetikleyici dahil)
                existing_symbol['ratio_percent'] = symbol_data.get('ratio_percent', 0)
                existing_symbol['abs_ratio_percent'] = abs(symbol_data.get('ratio_percent', 0))
                existing_symbol['z_score'] = symbol_data.get('z_score', 0)
                existing_symbol['timeframe'] = symbol_data.get('timeframe', '4h')
                
                # 🆕 VPMV güncelle
                existing_symbol['vpmv_net_power'] = symbol_data.get('vpmv_net_power', 0)
                existing_symbol['vpmv_signal'] = symbol_data.get('vpmv_signal', 'NEUTRAL')
                existing_symbol['vpmv_update_time'] = datetime.now().strftime('%H:%M')
                
                # ✅ YENİ: Tetikleyici güncelle
                existing_symbol['vpmv_trigger_name'] = symbol_data.get('vpmv_trigger_name', 'Yok')
                existing_symbol['vpmv_trigger_active'] = symbol_data.get('vpmv_trigger_active', False)
                
                # SADECE manuel override yoksa türü güncelle
                if not existing_symbol.get('manual_type_override', False):
                    existing_symbol['supertrend_type'] = symbol_data.get('trend_direction', 'None')
                
                # Ekleme nedenini güncelle
                ratio_percent = abs(symbol_data.get('ratio_percent', 0))
                z_score = abs(symbol_data.get('z_score', 0))
                add_reasons = []
                if ratio_percent >= 100.0:
                    add_reasons.append(f"{ratio_percent:.2f}% ratio")
                if z_score >= 2.0:
                    add_reasons.append(f"Z-Score: {z_score:.2f}")
                existing_symbol['add_reason'] = " + ".join(add_reasons) if add_reasons else "Manuel ekleme"
                
                logger.info(f"🔄 {symbol} güncellendi: Ratio {ratio_percent:.2f}%, VPMV: {symbol_data.get('vpmv_net_power', 0)}, Tetik: {symbol_data.get('vpmv_trigger_name', 'Yok')}")
    
    def get_permanent_symbols(self) -> List[Dict[str, Any]]:
        """✅ Kalıcı sembolleri getir (Ratio %'ye göre sıralı) (Thread-safe)"""
        with self._permanent_lock:
            return sorted(
                self.permanent_high_ratio, 
                key=lambda x: (x.get('abs_ratio_percent', 0), abs(x.get('z_score', 0))), 
                reverse=True
            )
    
    def _get_permanent_symbol_unsafe(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Belirli kalıcı sembolü getir (UNSAFE - lock içinden çağrılmalı)
        ⚠️ Bu metod lock içinden çağrılır, direkt kullanma!
        """
        perm_symbol = next((s for s in self.permanent_high_ratio if s['symbol'] == symbol), None)
        
        # ✅ Eksik alanları otomatik ekle (backward compatibility)
        if perm_symbol:
            # C-Signal alanları
            if 'last_c_signal_alert_time' not in perm_symbol:
                perm_symbol['last_c_signal_alert_time'] = None
            if 'last_c_signal_value' not in perm_symbol:
                perm_symbol['last_c_signal_value'] = None
            if 'last_c_signal_type' not in perm_symbol:
                perm_symbol['last_c_signal_type'] = None
            if 'c_signal_history' not in perm_symbol:
                perm_symbol['c_signal_history'] = []
            
            # 🆕 VPMV alanları
            if 'vpmv_net_power' not in perm_symbol:
                perm_symbol['vpmv_net_power'] = 0
            if 'vpmv_signal' not in perm_symbol:
                perm_symbol['vpmv_signal'] = 'NEUTRAL'
            if 'vpmv_update_time' not in perm_symbol:
                perm_symbol['vpmv_update_time'] = None
            
            # ✅ YENİ: Tetikleyici alanları
            if 'vpmv_trigger_name' not in perm_symbol:
                perm_symbol['vpmv_trigger_name'] = 'Yok'
            if 'vpmv_trigger_active' not in perm_symbol:
                perm_symbol['vpmv_trigger_active'] = False
                
        return perm_symbol
    
    def get_permanent_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Belirli kalıcı sembolü getir (VPMV + Tetikleyici alanları dahil) (Thread-safe)"""
        with self._permanent_lock:
            return self._get_permanent_symbol_unsafe(symbol)
    
    def update_permanent_symbol(self, symbol: str, update_data: Dict[str, Any]) -> bool:
        """Kalıcı sembol verisini güncelle (VPMV + Tetikleyici dahil) (Thread-safe)"""
        with self._permanent_lock:
            existing_symbol = self._get_permanent_symbol_unsafe(symbol)
            if existing_symbol:
                existing_symbol.update(update_data)
                
                # 🆕 VPMV veya Tetikleyici güncellemesi varsa zamanı işaretle
                if 'vpmv_net_power' in update_data or 'vpmv_signal' in update_data or 'vpmv_trigger_name' in update_data:
                    existing_symbol['vpmv_update_time'] = datetime.now().strftime('%H:%M')
                
                return True
            return False
    
    # =====================================================
    # 🆕 VPMV SPESİFİK FONKSİYONLAR (Thread-safe)
    # =====================================================
    
    def get_vpmv_statistics(self) -> Dict[str, Any]:
        """VPMV istatistiklerini getir (Thread-safe)"""
        with self._permanent_lock:
            if not self.permanent_high_ratio:
                return {
                    'total_symbols': 0,
                    'strong_long_count': 0,
                    'long_count': 0,
                    'short_count': 0,
                    'strong_short_count': 0,
                    'neutral_count': 0,
                    'avg_vpmv': 0,
                    'max_vpmv': 0,
                    'min_vpmv': 0,
                    'trigger_active_count': 0,
                    'momentum_trigger_count': 0,
                    'volume_trigger_count': 0,
                    'volatility_trigger_count': 0
                }
            
            # Sinyal sayıları
            strong_long = sum(1 for s in self.permanent_high_ratio if s.get('vpmv_signal') == 'STRONG LONG')
            long_count = sum(1 for s in self.permanent_high_ratio if s.get('vpmv_signal') == 'LONG')
            short_count = sum(1 for s in self.permanent_high_ratio if s.get('vpmv_signal') == 'SHORT')
            strong_short = sum(1 for s in self.permanent_high_ratio if s.get('vpmv_signal') == 'STRONG SHORT')
            neutral_count = sum(1 for s in self.permanent_high_ratio if s.get('vpmv_signal') == 'NEUTRAL')
            
            # VPMV değerleri
            vpmv_values = [s.get('vpmv_net_power', 0) for s in self.permanent_high_ratio]
            avg_vpmv = sum(vpmv_values) / len(vpmv_values) if vpmv_values else 0
            max_vpmv = max(vpmv_values) if vpmv_values else 0
            min_vpmv = min(vpmv_values) if vpmv_values else 0
            
            # Tetikleyici istatistikleri
            trigger_active_count = sum(1 for s in self.permanent_high_ratio if s.get('vpmv_trigger_active', False))
            momentum_trigger_count = sum(1 for s in self.permanent_high_ratio 
                                        if s.get('vpmv_trigger_active', False) and s.get('vpmv_trigger_name') == 'Momentum')
            volume_trigger_count = sum(1 for s in self.permanent_high_ratio 
                                      if s.get('vpmv_trigger_active', False) and s.get('vpmv_trigger_name') == 'Hacim')
            volatility_trigger_count = sum(1 for s in self.permanent_high_ratio 
                                          if s.get('vpmv_trigger_active', False) and s.get('vpmv_trigger_name') == 'Volatilite')
            
            return {
                'total_symbols': len(self.permanent_high_ratio),
                'strong_long_count': strong_long,
                'long_count': long_count,
                'short_count': short_count,
                'strong_short_count': strong_short,
                'neutral_count': neutral_count,
                'avg_vpmv': round(avg_vpmv, 2),
                'max_vpmv': round(max_vpmv, 2),
                'min_vpmv': round(min_vpmv, 2),
                'trigger_active_count': trigger_active_count,
                'momentum_trigger_count': momentum_trigger_count,
                'volume_trigger_count': volume_trigger_count,
                'volatility_trigger_count': volatility_trigger_count
            }
    
    def get_symbols_by_vpmv_signal(self, signal_type: str) -> List[Dict[str, Any]]:
        """Belirli VPMV sinyaline sahip sembolleri getir (Thread-safe)"""
        with self._permanent_lock:
            return [
                symbol for symbol in self.permanent_high_ratio 
                if symbol.get('vpmv_signal') == signal_type
            ]
    
    def get_top_vpmv_symbols(self, limit: int = 10, sort_by: str = 'highest') -> List[Dict[str, Any]]:
        """En yüksek/düşük VPMV değerine sahip sembolleri getir (Thread-safe)"""
        with self._permanent_lock:
            sorted_symbols = sorted(
                self.permanent_high_ratio,
                key=lambda x: x.get('vpmv_net_power', 0),
                reverse=(sort_by == 'highest')
            )
            
            return sorted_symbols[:limit]
    
    def get_active_triggers(self) -> List[Dict[str, Any]]:
        """Aktif tetikleyicileri olan sembolleri getir (Thread-safe)"""
        with self._permanent_lock:
            return [
                {
                    'symbol': s['symbol'],
                    'trigger_name': s.get('vpmv_trigger_name', 'Yok'),
                    'vpmv_net_power': s.get('vpmv_net_power', 0),
                    'vpmv_signal': s.get('vpmv_signal', 'NEUTRAL'),
                    'tradingview_link': s.get('tradingview_link', '#')
                }
                for s in self.permanent_high_ratio 
                if s.get('vpmv_trigger_active', False)
            ]
    
    def get_symbols_by_trigger_type(self, trigger_type: str) -> List[Dict[str, Any]]:
        """Belirli tetikleyici tipine sahip sembolleri getir (Thread-safe)"""
        with self._permanent_lock:
            return [
                symbol for symbol in self.permanent_high_ratio 
                if symbol.get('vpmv_trigger_active', False) and 
                   symbol.get('vpmv_trigger_name') == trigger_type
            ]
    
    # =====================================================
    # DİNAMİK C-SIGNAL ±X MANAGEMENT (Thread-safe)
    # =====================================================
    
    def update_c_signal(self, symbol: str, c_signal_value: Optional[float]) -> Dict[str, Any]:
        """C-Signal değerini güncelle ve DİNAMİK THRESHOLD kontrolü yap (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if not permanent_symbol:
                return {
                    'signal_triggered': False,
                    'signal_type': None,
                    'reason': 'Symbol not in permanent list'
                }
            
            # C-Signal değerini güncelle
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            permanent_symbol['c_signal'] = c_signal_value
            permanent_symbol['c_signal_update_time'] = current_time
            
            if c_signal_value is None:
                return {
                    'signal_triggered': False,
                    'signal_type': None,
                    'reason': 'C-Signal value is None'
                }
            
            # Önceki değeri al
            last_c_signal = permanent_symbol.get('last_c_signal_value')
            
            # C-Signal tarihçesine ekle
            if 'c_signal_history' not in permanent_symbol:
                permanent_symbol['c_signal_history'] = []
            
            permanent_symbol['c_signal_history'].append({
                'value': c_signal_value,
                'timestamp': current_time
            })
            
            # Son 10 değeri tut
            if len(permanent_symbol['c_signal_history']) > 10:
                permanent_symbol['c_signal_history'] = permanent_symbol['c_signal_history'][-10:]
            
            # DİNAMİK THRESHOLD kontrolü yap
            signal_result = self._check_c_signal_threshold(symbol, c_signal_value, last_c_signal)
            
            # Son C-Signal değerini kaydet
            permanent_symbol['last_c_signal_value'] = c_signal_value
            
            if signal_result['signal_triggered']:
                permanent_symbol['last_c_signal_type'] = signal_result['signal_type']
                logger.info(f"🔔 {symbol} C-Signal ALERT: {signal_result['signal_type']} - Değer: {c_signal_value:.2f}")
            
            return signal_result
    
    def _check_c_signal_threshold(self, symbol: str, current_value: float, 
                                  previous_value: Optional[float]) -> Dict[str, Any]:
        """DİNAMİK C-Signal ±X threshold kontrolü - Config'den threshold alır (UNSAFE - lock içinden çağrılır)"""
        from config import Config
        
        LONG_THRESHOLD = Config.get_c_signal_long_threshold()
        SHORT_THRESHOLD = Config.get_c_signal_short_threshold()
        
        signal_triggered = False
        signal_type = None
        reason = "No signal"
        
        # LONG sinyali kontrolü (>= +X)
        if current_value >= LONG_THRESHOLD:
            if previous_value is None or previous_value < LONG_THRESHOLD:
                signal_triggered = True
                signal_type = 'L'
                reason = f"C-Signal crossed +{LONG_THRESHOLD} threshold: {current_value:.2f}"
        
        # SHORT sinyali kontrolü (<= -X)
        elif current_value <= SHORT_THRESHOLD:
            if previous_value is None or previous_value > SHORT_THRESHOLD:
                signal_triggered = True
                signal_type = 'S'
                reason = f"C-Signal crossed {SHORT_THRESHOLD} threshold: {current_value:.2f}"
        
        return {
            'signal_triggered': signal_triggered,
            'signal_type': signal_type,
            'current_value': current_value,
            'previous_value': previous_value,
            'reason': reason,
            'threshold_used': abs(LONG_THRESHOLD)
        }
    
    def get_c_signal_status(self, symbol: str) -> Dict[str, Any]:
        """Sembolün C-Signal durumunu getir (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if not permanent_symbol:
                return {
                    'has_signal': False,
                    'signal_type': None,
                    'current_value': None,
                    'last_alert_time': None
                }
            
            return {
                'has_signal': permanent_symbol.get('last_c_signal_type') is not None,
                'signal_type': permanent_symbol.get('last_c_signal_type'),
                'current_value': permanent_symbol.get('c_signal'),
                'last_alert_time': permanent_symbol.get('last_c_signal_alert_time'),
                'c_signal_history': permanent_symbol.get('c_signal_history', [])
            }
    
    def get_all_active_c_signals(self) -> List[Dict[str, Any]]:
        """Aktif C-Signal'leri olan tüm sembolleri getir (Thread-safe)"""
        with self._permanent_lock:
            active_signals = []
            
            for symbol_data in self.permanent_high_ratio:
                signal_type = symbol_data.get('last_c_signal_type')
                if signal_type:
                    active_signals.append({
                        'symbol': symbol_data['symbol'],
                        'signal_type': signal_type,
                        'c_signal_value': symbol_data.get('c_signal'),
                        'alert_time': symbol_data.get('last_c_signal_alert_time'),
                        'tradingview_link': symbol_data.get('tradingview_link', '#')
                    })
            
            return active_signals
    
    def clear_c_signal_alert(self, symbol: str) -> bool:
        """C-Signal alert'ini temizle (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if permanent_symbol:
                permanent_symbol['last_c_signal_type'] = None
                permanent_symbol['last_c_signal_alert_time'] = None
                logger.info(f"🧹 {symbol} C-Signal alert temizlendi")
                return True
            return False
    
    # =====================================================
    # MANUEL TÜR DEĞİŞTİRME FONKSİYONLARI (Thread-safe)
    # =====================================================
    
    def set_manual_type_override(self, symbol: str, new_type: str) -> bool:
        """Manuel tür değişikliği işaretleme ve kaydetme (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if permanent_symbol:
                old_type = permanent_symbol.get('supertrend_type', 'None')
                
                permanent_symbol['manual_type_override'] = True
                permanent_symbol['manual_type_value'] = new_type
                permanent_symbol['supertrend_type'] = new_type
                permanent_symbol['manual_override_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                permanent_symbol['last_telegram_alert'] = None
                
                logger.info(f"🔒 {symbol} türü manuel olarak {old_type} -> {new_type} DEĞİŞTİRİLDİ ve KİLİTLENDİ")
                return True
            
            logger.warning(f"⚠️ {symbol} kalıcı listede bulunamadı")
            return False
    
    def clear_manual_type_override(self, symbol: str) -> bool:
        """Manuel tür değişikliği kilidi kaldır (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if permanent_symbol:
                permanent_symbol['manual_type_override'] = False
                permanent_symbol['manual_type_value'] = None
                permanent_symbol['manual_override_date'] = None
                
                logger.info(f"🔓 {symbol} manuel tür kilidi kaldırıldı")
                return True
            
            return False
    
    def is_manual_type_overridden(self, symbol: str) -> bool:
        """Sembol manuel olarak değiştirilmiş mi kontrol et (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if permanent_symbol:
                return permanent_symbol.get('manual_type_override', False)
            return False
    
    def get_manual_type_info(self, symbol: str) -> Dict[str, Any]:
        """Manuel tür değişikliği bilgilerini getir (Thread-safe)"""
        with self._permanent_lock:
            permanent_symbol = self._get_permanent_symbol_unsafe(symbol)
            if permanent_symbol:
                return {
                    'is_manual': permanent_symbol.get('manual_type_override', False),
                    'manual_type': permanent_symbol.get('manual_type_value'),
                    'override_date': permanent_symbol.get('manual_override_date'),
                    'current_type': permanent_symbol.get('supertrend_type')
                }
            return {
                'is_manual': False,
                'manual_type': None,
                'override_date': None,
                'current_type': None
            }
    
    def clear_permanent_symbols(self) -> int:
        """Tüm kalıcı sembolleri temizle (Thread-safe)"""
        with self._permanent_lock:
            count = len(self.permanent_high_ratio)
            self.permanent_high_ratio = []
            logger.info(f"{count} kalıcı sembol temizlendi")
            return count
    
    def remove_permanent_symbol(self, symbol: str) -> bool:
        """Kalıcı listeden belirli sembolü çıkar (Thread-safe)"""
        with self._permanent_lock:
            try:
                initial_count = len(self.permanent_high_ratio)
                self.permanent_high_ratio = [
                    s for s in self.permanent_high_ratio 
                    if s['symbol'] != symbol
                ]
                removed = len(self.permanent_high_ratio) < initial_count
                
                if removed:
                    logger.info(f"{symbol} kalıcı listeden çıkarıldı")
                
                return removed
                
            except Exception as e:
                logger.error(f"Kalıcı listeden çıkarma hatası: {e}")
                return False
    
    # =====================================================
    # SUPERTREND SPESİFİK FONKSİYONLAR (Thread-safe)
    # =====================================================
    
    def is_high_priority_symbol(self, symbol_data: Dict[str, Any]) -> bool:
        """Yüksek öncelikli sembol mu kontrol et"""
        ratio_percent = abs(symbol_data.get('ratio_percent', 0))
        return ratio_percent >= 100.0
    
    def get_high_ratio_symbols(self, min_ratio: float = 100.0) -> List[Dict[str, Any]]:
        """Belirli ratio üzerindeki sembolleri getir (Thread-safe)"""
        with self._permanent_lock:
            return [
                symbol for symbol in self.permanent_high_ratio 
                if symbol.get('abs_ratio_percent', 0) >= min_ratio
            ]
    
    def get_supertrend_statistics(self) -> Dict[str, Any]:
        """Supertrend + VPMV + Tetikleyici sistemi istatistikleri (Thread-safe)"""
        with self._permanent_lock:
            if not self.permanent_high_ratio:
                return {
                    'total_symbols': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'avg_ratio': 0,
                    'max_ratio': 0,
                    'high_z_score_count': 0,
                    'active_c_signal_count': 0,
                    'vpmv_strong_long': 0,
                    'vpmv_long': 0,
                    'vpmv_short': 0,
                    'vpmv_strong_short': 0,
                    'avg_vpmv': 0,
                    'trigger_active_count': 0,
                    'momentum_trigger_count': 0,
                    'volume_trigger_count': 0,
                    'volatility_trigger_count': 0
                }
            
            bullish_count = sum(1 for s in self.permanent_high_ratio 
                               if s.get('supertrend_type') == 'Bullish')
            bearish_count = sum(1 for s in self.permanent_high_ratio 
                               if s.get('supertrend_type') == 'Bearish')
            
            ratios = [s.get('abs_ratio_percent', 0) for s in self.permanent_high_ratio]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0
            max_ratio = max(ratios) if ratios else 0
            
            high_z_score_count = sum(1 for s in self.permanent_high_ratio 
                                   if abs(s.get('z_score', 0)) >= 2.0)
            
            active_c_signal_count = sum(1 for s in self.permanent_high_ratio 
                                       if s.get('last_c_signal_type') is not None)
            
            vpmv_stats = self.get_vpmv_statistics()
            
            return {
                'total_symbols': len(self.permanent_high_ratio),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'avg_ratio': round(avg_ratio, 2),
                'max_ratio': round(max_ratio, 2),
                'high_z_score_count': high_z_score_count,
                'active_c_signal_count': active_c_signal_count,
                'vpmv_strong_long': vpmv_stats['strong_long_count'],
                'vpmv_long': vpmv_stats['long_count'],
                'vpmv_short': vpmv_stats['short_count'],
                'vpmv_strong_short': vpmv_stats['strong_short_count'],
                'avg_vpmv': vpmv_stats['avg_vpmv'],
                'trigger_active_count': vpmv_stats['trigger_active_count'],
                'momentum_trigger_count': vpmv_stats['momentum_trigger_count'],
                'volume_trigger_count': vpmv_stats['volume_trigger_count'],
                'volatility_trigger_count': vpmv_stats['volatility_trigger_count']
            }
    
    # =====================================================
    # CACHE MANAGEMENT (Thread-safe)
    # =====================================================
    
    def set_cache(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Cache'e veri ekle (Thread-safe)"""
        with self._cache_lock:
            self.analysis_cache[key] = {
                'value': value,
                'timestamp': datetime.now(),
                'ttl': ttl_seconds
            }
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Cache'den veri getir (Thread-safe)"""
        with self._cache_lock:
            if key not in self.analysis_cache:
                return None
            
            cached_item = self.analysis_cache[key]
            
            # TTL kontrolü
            elapsed = (datetime.now() - cached_item['timestamp']).total_seconds()
            if elapsed > cached_item['ttl']:
                del self.analysis_cache[key]
                return None
            
            return cached_item['value']
    
    def clear_cache(self) -> None:
        """Tüm cache'i temizle (Thread-safe)"""
        with self._cache_lock:
            count = len(self.analysis_cache)
            self.analysis_cache = {}
            logger.info(f"{count} cache entry temizlendi")
    
    def cleanup_expired_cache(self) -> int:
        """Süresi dolmuş cache'leri temizle (Thread-safe)"""
        with self._cache_lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, item in self.analysis_cache.items():
                elapsed = (current_time - item['timestamp']).total_seconds()
                if elapsed > item['ttl']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.analysis_cache[key]
            
            if expired_keys:
                logger.info(f"{len(expired_keys)} expired cache entry temizlendi")
            
            return len(expired_keys)
    
    # =====================================================
    # SYSTEM STATISTICS (Thread-safe)
    # =====================================================
    
    def increment_analysis_count(self) -> None:
        """Analiz sayacını artır (Thread-safe)"""
        with self._stats_lock:
            self.system_stats['total_analyses'] += 1
            self.system_stats['last_analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def increment_telegram_alerts(self) -> None:
        """Telegram bildirim sayacını artır (Thread-safe)"""
        with self._stats_lock:
            self.system_stats['telegram_alerts_sent'] += 1
    
    def increment_c_signal_alerts(self) -> None:
        """C-Signal alert sayacını artır (Thread-safe)"""
        with self._stats_lock:
            self.system_stats['c_signal_alerts_sent'] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Sistem istatistiklerini getir (Thread-safe)"""
        with self._stats_lock:
            with self._selected_symbols_lock:
                with self._permanent_lock:
                    with self._cache_lock:
                        base_stats = {
                            **self.system_stats,
                            'selected_symbols_count': len(self.selected_symbols),
                            'permanent_symbols_count': len(self.permanent_high_ratio),
                            'cache_entries_count': len(self.analysis_cache)
                        }
        
        combined_stats = self.get_supertrend_statistics()
        base_stats.update({f'supertrend_{k}': v for k, v in combined_stats.items()})
        
        return base_stats
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırla (Thread-safe)"""
        with self._stats_lock:
            self.system_stats = {
                'total_analyses': 0,
                'last_analysis_time': None,
                'telegram_alerts_sent': 0,
                'c_signal_alerts_sent': 0
            }
            logger.info("Sistem istatistikleri sıfırlandı")
    
    # =====================================================
    # UTILITY METHODS (Thread-safe)
    # =====================================================
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Bellek kullanım özetini getir (Thread-safe)"""
        with self._selected_symbols_lock:
            with self._permanent_lock:
                with self._cache_lock:
                    return {
                        'selected_symbols_count': len(self.selected_symbols),
                        'permanent_symbols_count': len(self.permanent_high_ratio),
                        'cache_entries_count': len(self.analysis_cache),
                        'total_memory_objects': (
                            len(self.selected_symbols) + 
                            len(self.permanent_high_ratio) + 
                            len(self.analysis_cache)
                        )
                    }
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Veri bütünlüğünü kontrol et (Thread-safe)"""
        with self._permanent_lock:
            with self._selected_symbols_lock:
                issues = []
                
                for symbol_data in self.permanent_high_ratio:
                    if not symbol_data.get('symbol'):
                        issues.append("Sembol adı eksik permanent symbol bulundu")
                    
                    if symbol_data.get('abs_ratio_percent', 0) < 0:
                        issues.append(f"Negatif ratio: {symbol_data.get('symbol')}")
                    
                    supertrend_type = symbol_data.get('supertrend_type')
                    if supertrend_type not in ['Bullish', 'Bearish', 'None', None]:
                        issues.append(f"Geçersiz supertrend türü: {symbol_data.get('symbol')} - {supertrend_type}")
                    
                    vpmv_value = symbol_data.get('vpmv_net_power', 0)
                    if vpmv_value < -100 or vpmv_value > 100:
                        issues.append(f"VPMV değeri aralık dışı: {symbol_data.get('symbol')} - {vpmv_value}")
                    
                    vpmv_signal = symbol_data.get('vpmv_signal')
                    valid_signals = ['STRONG LONG', 'LONG', 'SHORT', 'STRONG SHORT', 'NEUTRAL']
                    if vpmv_signal not in valid_signals:
                        issues.append(f"Geçersiz VPMV sinyali: {symbol_data.get('symbol')} - {vpmv_signal}")
                    
                    trigger_name = symbol_data.get('vpmv_trigger_name')
                    valid_triggers = ['Yok', 'Momentum', 'Hacim', 'Volatilite']
                    if trigger_name not in valid_triggers:
                        issues.append(f"Geçersiz tetikleyici: {symbol_data.get('symbol')} - {trigger_name}")
                
                for symbol in self.selected_symbols:
                    if not isinstance(symbol, str) or len(symbol) < 3:
                        issues.append(f"Geçersiz sembol formatı: {symbol}")
                
                return {
                    'is_valid': len(issues) == 0,
                    'issues': issues,
                    'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
    
    def export_data(self) -> Dict[str, Any]:
        """Tüm veriyi export et (Thread-safe)"""
        with self._selected_symbols_lock:
            with self._permanent_lock:
                return {
                    'selected_symbols': self.selected_symbols.copy(),
                    'permanent_high_ratio': [s.copy() for s in self.permanent_high_ratio],
                    'system_stats': self.get_system_stats(),
                    'supertrend_stats': self.get_supertrend_statistics(),
                    'vpmv_stats': self.get_vpmv_statistics(),
                    'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
    
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Veriyi import et (Thread-safe)"""
        with self._selected_symbols_lock:
            with self._permanent_lock:
                with self._stats_lock:
                    try:
                        if 'selected_symbols' in data:
                            self.selected_symbols = data['selected_symbols']
                        
                        if 'permanent_high_ratio' in data:
                            self.permanent_high_ratio = []
                            for symbol_data in data['permanent_high_ratio']:
                                
                                # Backward compatibility dönüşümleri
                                if 'max_ratio_percent' in symbol_data and 'ratio_percent' not in symbol_data:
                                    symbol_data['ratio_percent'] = symbol_data['max_ratio_percent']
                                    del symbol_data['max_ratio_percent']
                                
                                if 'max_abs_ratio_percent' in symbol_data and 'abs_ratio_percent' not in symbol_data:
                                    symbol_data['abs_ratio_percent'] = symbol_data['max_abs_ratio_percent']
                                    del symbol_data['max_abs_ratio_percent']
                                
                                if 'max_supertrend_type' in symbol_data and 'supertrend_type' not in symbol_data:
                                    symbol_data['supertrend_type'] = symbol_data['max_supertrend_type']
                                    del symbol_data['max_supertrend_type']
                                
                                if 'max_z_score' in symbol_data and 'z_score' not in symbol_data:
                                    symbol_data['z_score'] = symbol_data['max_z_score']
                                    del symbol_data['max_z_score']
                                
                                # Eksik alanları ekle
                                for field, default in [
                                    ('last_c_signal_value', None),
                                    ('last_c_signal_type', None),
                                    ('last_c_signal_alert_time', None),
                                    ('c_signal_history', []),
                                    ('vpmv_net_power', 0),
                                    ('vpmv_signal', 'NEUTRAL'),
                                    ('vpmv_update_time', None),
                                    ('vpmv_trigger_name', 'Yok'),
                                    ('vpmv_trigger_active', False)
                                ]:
                                    if field not in symbol_data:
                                        symbol_data[field] = default
                                
                                # Eski alanları temizle
                                for old_field in ['deviso_lines', 'deviso_status', 'deviso_contact_history', 'reverse_momentum']:
                                    if old_field in symbol_data:
                                        del symbol_data[old_field]
                                
                                self.permanent_high_ratio.append(symbol_data)
                        
                        if 'system_stats' in data:
                            old_stats = data['system_stats']
                            
                            for old_field in ['deviso_line_contacts', 'reverse_momentum_detected']:
                                if old_field in old_stats:
                                    del old_stats[old_field]
                            
                            if 'c_signal_alerts_sent' not in old_stats:
                                old_stats['c_signal_alerts_sent'] = 0
                            
                            self.system_stats.update(old_stats)
                        
                        logger.info("✅ Veri başarıyla import edildi (VPMV + Tetikleyici + Thread-safe)")
                        return True
                        
                    except Exception as e:
                        logger.error(f"❌ Veri import hatası: {e}")
                        return False

# Global instance
memory_storage_instance = MemoryStorage()