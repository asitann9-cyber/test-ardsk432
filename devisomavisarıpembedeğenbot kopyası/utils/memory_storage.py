"""
Memory Storage Service
Bellekte veri depolama ve yönetim işlemleri - Devis'So Trend için güncellenmiş
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryStorage:
    """Bellekte veri depolama sınıfı - Devis'So Trend sistemi için"""
    
    def __init__(self):
        """Memory storage'ı başlat"""
        self.selected_symbols: List[str] = []
        self.permanent_high_ratio: List[Dict[str, Any]] = []  # RENAMED: high_consecutive -> high_ratio
        self.analysis_cache: Dict[str, Any] = {}
        self.system_stats: Dict[str, Any] = {
            'total_analyses': 0,
            'last_analysis_time': None,
            'telegram_alerts_sent': 0,
            'deviso_line_contacts': 0  # UPDATED: reverse_momentum -> deviso_line_contacts
        }
    
    # =====================================================
    # SELECTED SYMBOLS MANAGEMENT
    # =====================================================
    
    def get_selected_symbols(self) -> List[str]:
        """Seçili sembolleri getir"""
        return self.selected_symbols.copy()
    
    def save_selected_symbols(self, symbols: List[str]) -> None:
        """Seçili sembolleri belleğe kaydet"""
        self.selected_symbols = symbols.copy()
        logger.info(f"{len(symbols)} sembol belleğe kaydedildi")
    
    def add_selected_symbols(self, symbols: List[str]) -> List[str]:
        """Yeni semboller ekle"""
        current_symbols = set(self.selected_symbols)
        new_symbols = set(symbols)
        all_symbols = list(current_symbols | new_symbols)
        self.selected_symbols = all_symbols
        
        added_count = len(new_symbols - current_symbols)
        logger.info(f"{added_count} yeni sembol eklendi, toplam: {len(all_symbols)}")
        
        return all_symbols
    
    def remove_selected_symbol(self, symbol: str) -> List[str]:
        """Belirli sembolü sil"""
        if symbol in self.selected_symbols:
            self.selected_symbols.remove(symbol)
            logger.info(f"{symbol} sembolü silindi")
        
        return self.selected_symbols
    
    def clear_selected_symbols(self) -> None:
        """Tüm seçili sembolleri temizle"""
        count = len(self.selected_symbols)
        self.selected_symbols = []
        logger.info(f"{count} sembol temizlendi")
    
    def is_symbol_selected(self, symbol: str) -> bool:
        """Sembol seçili mi kontrol et"""
        return symbol in self.selected_symbols
    
    # =====================================================
    # PERMANENT HIGH RATIO MANAGEMENT - DEVİS'SO TREND ENTEGRE
    # =====================================================
    
    def add_permanent_symbol(self, symbol_data: Dict[str, Any]) -> None:
        """Kalıcı listeye sembol ekle - Devis'So çizgi değerleri ile"""
        symbol = symbol_data.get('symbol')
        if not symbol:
            logger.warning("Sembol adı bulunamadı, kalıcı listeye eklenemedi")
            return
        
        # Mevcut sembol kontrolü
        existing_symbol = self.get_permanent_symbol(symbol)
        
        if not existing_symbol:
            # Yeni sembol - kalıcı listeye ekle
            ratio_percent = abs(symbol_data.get('ratio_percent', 0))
            z_score = abs(symbol_data.get('z_score', 0))
            
            # Ekleme nedenini belirle - SUPERTREND KRİTERLERİ
            add_reasons = []
            if ratio_percent >= 100.0:
                add_reasons.append(f"{ratio_percent:.2f}% ratio")
            if z_score >= 2.0:
                add_reasons.append(f"Z-Score: {z_score:.2f}")
            
            # Devis'So çizgi değerleri - boş başlangıç
            deviso_lines = symbol_data.get('deviso_lines', {})
            default_deviso_lines = {
                'ma_blue': deviso_lines.get('ma_blue', 0.0),
                'ma_yellow': deviso_lines.get('ma_yellow', 0.0),
                'ma_purple': deviso_lines.get('ma_purple', 0.0),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            permanent_entry = {
                'symbol': symbol,
                'first_high_ratio_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'max_ratio_percent': symbol_data.get('ratio_percent', 0),  # GÜNCEL DEĞER
                'max_supertrend_type': symbol_data.get('trend_direction', 'None'),  # GÜNCEL DEĞER
                'max_z_score': symbol_data.get('z_score', 0),  # GÜNCEL DEĞER
                'max_abs_ratio_percent': abs(symbol_data.get('ratio_percent', 0)),  # GÜNCEL DEĞER
                'timeframe': symbol_data.get('timeframe', '4h'),
                'tradingview_link': symbol_data.get('tradingview_link', '#'),
                'c_signal': None,
                'c_signal_update_time': None,
                'add_reason': " + ".join(add_reasons) if add_reasons else "Manuel ekleme",
                
                # DEVİS'SO TREND YENİ ALANLAR
                'deviso_lines': default_deviso_lines,  # Çizgi fiyatları (USDT)
                'deviso_status': {  # Çizgi temas durumu
                    'has_line_contact': False,
                    'contact_level': 'None',
                    'signal_strength': 'None',
                    'alert_message': '',
                    'line_values': {}
                },
                'deviso_contact_history': [],  # Temas geçmişi
                
                'last_telegram_alert': None,
                # MANUEL TÜR DEĞİŞTİRME ALANLARI
                'manual_type_override': False,
                'manual_type_value': None,
                'manual_override_date': None
            }
            
            self.permanent_high_ratio.append(permanent_entry)
            logger.info(f"🎯 {symbol} kalıcı listeye eklendi ({permanent_entry['add_reason']})")
        else:
            # Mevcut sembol - HER ZAMAN GÜNCEL DEĞERLERİ GÜNCELLE (rekor kontrolü yok)
            existing_symbol['max_ratio_percent'] = symbol_data.get('ratio_percent', 0)
            existing_symbol['max_abs_ratio_percent'] = abs(symbol_data.get('ratio_percent', 0))
            existing_symbol['max_z_score'] = symbol_data.get('z_score', 0)
            existing_symbol['timeframe'] = symbol_data.get('timeframe', '4h')
            
            # SADECE manuel override yoksa türü güncelle
            if not existing_symbol.get('manual_type_override', False):
                existing_symbol['max_supertrend_type'] = symbol_data.get('trend_direction', 'None')
            
            # Devis'So çizgi değerlerini güncelle
            deviso_lines = symbol_data.get('deviso_lines', {})
            if deviso_lines:
                existing_symbol['deviso_lines'] = {
                    'ma_blue': deviso_lines.get('ma_blue', existing_symbol.get('deviso_lines', {}).get('ma_blue', 0.0)),
                    'ma_yellow': deviso_lines.get('ma_yellow', existing_symbol.get('deviso_lines', {}).get('ma_yellow', 0.0)),
                    'ma_purple': deviso_lines.get('ma_purple', existing_symbol.get('deviso_lines', {}).get('ma_purple', 0.0)),
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # Ekleme nedenini güncelle
            ratio_percent = abs(symbol_data.get('ratio_percent', 0))
            z_score = abs(symbol_data.get('z_score', 0))
            add_reasons = []
            if ratio_percent >= 100.0:
                add_reasons.append(f"{ratio_percent:.2f}% ratio")
            if z_score >= 2.0:
                add_reasons.append(f"Z-Score: {z_score:.2f}")
            existing_symbol['add_reason'] = " + ".join(add_reasons) if add_reasons else "Manuel ekleme"
            
            logger.info(f"🔄 {symbol} güncel değerlerle güncellendi: Ratio {ratio_percent:.2f}%, Z-Score {z_score:.2f}")
    
    def get_permanent_symbols(self) -> List[Dict[str, Any]]:
        """Kalıcı sembolleri getir (Ratio %'ye göre sıralı)"""
        return sorted(
            self.permanent_high_ratio, 
            key=lambda x: (x.get('max_abs_ratio_percent', 0), abs(x.get('max_z_score', 0))), 
            reverse=True
        )
    
    def get_permanent_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Belirli kalıcı sembolü getir"""
        return next((s for s in self.permanent_high_ratio if s['symbol'] == symbol), None)
    
    def update_permanent_symbol(self, symbol: str, update_data: Dict[str, Any]) -> bool:
        """Kalıcı sembol verisini güncelle"""
        existing_symbol = self.get_permanent_symbol(symbol)
        if existing_symbol:
            existing_symbol.update(update_data)
            return True
        return False
    
    # =====================================================
    # DEVİS'SO TREND SPESİFİK FONKSİYONLAR
    # =====================================================
    
    def update_deviso_line_contact(self, symbol: str, contact_data: Dict[str, Any]) -> bool:
        """
        Devis'So çizgi temas durumunu güncelle
        
        Args:
            symbol (str): Sembol adı
            contact_data (Dict[str, Any]): Temas verileri
            
        Returns:
            bool: Güncelleme başarılı ise True
        """
        permanent_symbol = self.get_permanent_symbol(symbol)
        if permanent_symbol:
            # Önceki durumu kaydet
            previous_contact = permanent_symbol.get('deviso_status', {}).get('contact_level', 'None')
            new_contact = contact_data.get('contact_level', 'None')
            
            # Deviso durumunu güncelle
            permanent_symbol['deviso_status'] = contact_data
            
            # Temas seviyesi değiştiyse geçmişe ekle
            if new_contact != 'None' and new_contact != previous_contact:
                contact_history = permanent_symbol.get('deviso_contact_history', [])
                contact_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'contact_level': new_contact,
                    'signal_strength': contact_data.get('signal_strength', 'None'),
                    'line_values': contact_data.get('line_values', {})
                })
                
                # Son 10 temas kaydını tut
                permanent_symbol['deviso_contact_history'] = contact_history[-10:]
                
                logger.info(f"📊 {symbol} Devis'So temas: {previous_contact} -> {new_contact}")
            
            return True
        return False
    
    def get_deviso_contact_symbols(self) -> List[Dict[str, Any]]:
        """Devis'So çizgi teması olan sembolleri getir"""
        return [
            symbol for symbol in self.permanent_high_ratio 
            if symbol.get('deviso_status', {}).get('has_line_contact', False)
        ]
    
    def clear_deviso_contact_history(self, symbol: str) -> bool:
        """Belirli sembolün Devis'So temas geçmişini temizle"""
        permanent_symbol = self.get_permanent_symbol(symbol)
        if permanent_symbol:
            permanent_symbol['deviso_contact_history'] = []
            logger.info(f"🗑️ {symbol} Devis'So temas geçmişi temizlendi")
            return True
        return False
    
    # MANUEL TÜR DEĞİŞTİRME FONKSİYONU
    def set_manual_type_override(self, symbol: str, new_type: str) -> bool:
        """
        Manuel tür değişikliği işaretleme ve kaydetme
        
        Args:
            symbol (str): Sembol adı
            new_type (str): Yeni tür (Bullish/Bearish)
            
        Returns:
            bool: İşlem başarılı ise True
        """
        permanent_symbol = self.get_permanent_symbol(symbol)
        if permanent_symbol:
            # Eski türü kaydet
            old_type = permanent_symbol.get('max_supertrend_type', 'None')
            
            # Manuel değişiklik bilgilerini güncelle
            permanent_symbol['manual_type_override'] = True
            permanent_symbol['manual_type_value'] = new_type
            permanent_symbol['max_supertrend_type'] = new_type  # Ana türü de güncelle
            permanent_symbol['manual_override_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Manuel değişiklik = Devis'So çizgi kontrol sıfırlama
            permanent_symbol['deviso_status'] = {
                'has_line_contact': False,
                'contact_level': 'Manual_Reset',
                'signal_strength': 'Strong',
                'alert_message': f'MANUEL DEĞİŞİKLİK: {old_type} → {new_type} trend değişikliği yapıldı!',
                'line_values': permanent_symbol.get('deviso_lines', {})
            }
            permanent_symbol['last_telegram_alert'] = None  # Yeni bildirim için sıfırla
            
            logger.info(f"🔒 {symbol} türü manuel olarak {old_type} -> {new_type} DEĞİŞTİRİLDİ ve KİLİTLENDİ")
            return True
        
        logger.warning(f"⚠️ {symbol} kalıcı listede bulunamadı, manuel tür değiştirilemedi")
        return False
    
    def clear_manual_type_override(self, symbol: str) -> bool:
        """
        Manuel tür değişikliği kilidi kaldır
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            bool: İşlem başarılı ise True
        """
        permanent_symbol = self.get_permanent_symbol(symbol)
        if permanent_symbol:
            permanent_symbol['manual_type_override'] = False
            permanent_symbol['manual_type_value'] = None
            permanent_symbol['manual_override_date'] = None
            
            # Devis'So durumu sıfırla
            permanent_symbol['deviso_status'] = {
                'has_line_contact': False,
                'contact_level': 'None',
                'signal_strength': 'None',
                'alert_message': '',
                'line_values': {}
            }
            
            logger.info(f"🔓 {symbol} manuel tür kilidi kaldırıldı - gerçek veriye dönecek")
            return True
        
        return False
    
    def is_manual_type_overridden(self, symbol: str) -> bool:
        """Sembol manuel olarak değiştirilmiş mi kontrol et"""
        permanent_symbol = self.get_permanent_symbol(symbol)
        if permanent_symbol:
            return permanent_symbol.get('manual_type_override', False)
        return False
    
    def get_manual_type_info(self, symbol: str) -> Dict[str, Any]:
        """Manuel tür değişikliği bilgilerini getir"""
        permanent_symbol = self.get_permanent_symbol(symbol)
        if permanent_symbol:
            return {
                'is_manual': permanent_symbol.get('manual_type_override', False),
                'manual_type': permanent_symbol.get('manual_type_value'),
                'override_date': permanent_symbol.get('manual_override_date'),
                'current_type': permanent_symbol.get('max_supertrend_type')
            }
        return {
            'is_manual': False,
            'manual_type': None,
            'override_date': None,
            'current_type': None
        }
    
    def clear_permanent_symbols(self) -> int:
        """Tüm kalıcı sembolleri temizle"""
        count = len(self.permanent_high_ratio)
        self.permanent_high_ratio = []
        logger.info(f"{count} kalıcı sembol temizlendi")
        return count
    
    def remove_permanent_symbol(self, symbol: str) -> bool:
        """Kalıcı listeden belirli sembolü çıkar"""
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
    # SUPERTREND SPESİFİK FONKSİYONLAR
    # =====================================================
    
    def is_high_priority_symbol(self, symbol_data: Dict[str, Any]) -> bool:
        """Yüksek öncelikli sembol mu kontrol et"""
        ratio_percent = abs(symbol_data.get('ratio_percent', 0))
        return ratio_percent >= 100.0
    
    def get_high_ratio_symbols(self, min_ratio: float = 100.0) -> List[Dict[str, Any]]:
        """Belirli ratio üzerindeki sembolleri getir"""
        return [
            symbol for symbol in self.permanent_high_ratio 
            if symbol.get('max_abs_ratio_percent', 0) >= min_ratio
        ]
    
    def get_supertrend_statistics(self) -> Dict[str, Any]:
        """Supertrend sistemi istatistikleri"""
        if not self.permanent_high_ratio:
            return {
                'total_symbols': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'avg_ratio': 0,
                'max_ratio': 0,
                'high_z_score_count': 0,
                'deviso_contact_count': 0
            }
        
        bullish_count = sum(1 for s in self.permanent_high_ratio 
                           if s.get('max_supertrend_type') == 'Bullish')
        bearish_count = sum(1 for s in self.permanent_high_ratio 
                           if s.get('max_supertrend_type') == 'Bearish')
        
        ratios = [s.get('max_abs_ratio_percent', 0) for s in self.permanent_high_ratio]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        max_ratio = max(ratios) if ratios else 0
        
        high_z_score_count = sum(1 for s in self.permanent_high_ratio 
                               if abs(s.get('max_z_score', 0)) >= 2.0)
        
        # Devis'So temas sayısı
        deviso_contact_count = sum(1 for s in self.permanent_high_ratio 
                                 if s.get('deviso_status', {}).get('has_line_contact', False))
        
        return {
            'total_symbols': len(self.permanent_high_ratio),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'avg_ratio': round(avg_ratio, 2),
            'max_ratio': round(max_ratio, 2),
            'high_z_score_count': high_z_score_count,
            'deviso_contact_count': deviso_contact_count  # YENİ
        }
    
    # =====================================================
    # CACHE MANAGEMENT
    # =====================================================
    
    def set_cache(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Cache'e veri ekle"""
        self.analysis_cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'ttl': ttl_seconds
        }
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Cache'den veri getir"""
        if key not in self.analysis_cache:
            return None
        
        cached_item = self.analysis_cache[key]
        
        # TTL kontrolü
        elapsed = (datetime.now() - cached_item['timestamp']).total_seconds()
        if elapsed > cached_item['ttl']:
            # Cache expired
            del self.analysis_cache[key]
            return None
        
        return cached_item['value']
    
    def clear_cache(self) -> None:
        """Tüm cache'i temizle"""
        count = len(self.analysis_cache)
        self.analysis_cache = {}
        logger.info(f"{count} cache entry temizlendi")
    
    def cleanup_expired_cache(self) -> int:
        """Süresi dolmuş cache'leri temizle"""
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
    # SYSTEM STATISTICS
    # =====================================================
    
    def increment_analysis_count(self) -> None:
        """Analiz sayacını artır"""
        self.system_stats['total_analyses'] += 1
        self.system_stats['last_analysis_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def increment_telegram_alerts(self) -> None:
        """Telegram bildirim sayacını artır"""
        self.system_stats['telegram_alerts_sent'] += 1
    
    def increment_deviso_contact_count(self) -> None:
        """Devis'So çizgi temas sayacını artır"""
        self.system_stats['deviso_line_contacts'] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Sistem istatistiklerini getir"""
        base_stats = {
            **self.system_stats,
            'selected_symbols_count': len(self.selected_symbols),
            'permanent_symbols_count': len(self.permanent_high_ratio),
            'cache_entries_count': len(self.analysis_cache),
            'deviso_contact_symbols_count': len(self.get_deviso_contact_symbols())
        }
        
        # Supertrend istatistikleri ekle
        supertrend_stats = self.get_supertrend_statistics()
        base_stats.update({f'supertrend_{k}': v for k, v in supertrend_stats.items()})
        
        return base_stats
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırla"""
        self.system_stats = {
            'total_analyses': 0,
            'last_analysis_time': None,
            'telegram_alerts_sent': 0,
            'deviso_line_contacts': 0
        }
        logger.info("Sistem istatistikleri sıfırlandı")
    
    # =====================================================
    # UTILITY METHODS
    # =====================================================
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Bellek kullanım özetini getir"""
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
        """Veri bütünlüğünü kontrol et"""
        issues = []
        
        # Permanent symbols validation
        for symbol_data in self.permanent_high_ratio:
            if not symbol_data.get('symbol'):
                issues.append("Sembol adı eksik permanent symbol bulundu")
            
            # Supertrend validasyonları
            if symbol_data.get('max_abs_ratio_percent', 0) < 0:
                issues.append(f"Negatif ratio: {symbol_data.get('symbol')}")
            
            supertrend_type = symbol_data.get('max_supertrend_type')
            if supertrend_type not in ['Bullish', 'Bearish', 'None', None]:
                issues.append(f"Geçersiz supertrend türü: {symbol_data.get('symbol')} - {supertrend_type}")
            
            # Devis'So validasyonları
            deviso_lines = symbol_data.get('deviso_lines', {})
            if not isinstance(deviso_lines, dict):
                issues.append(f"Geçersiz deviso_lines formatı: {symbol_data.get('symbol')}")
            
            deviso_status = symbol_data.get('deviso_status', {})
            if not isinstance(deviso_status, dict):
                issues.append(f"Geçersiz deviso_status formatı: {symbol_data.get('symbol')}")
        
        # Selected symbols validation
        for symbol in self.selected_symbols:
            if not isinstance(symbol, str) or len(symbol) < 3:
                issues.append(f"Geçersiz sembol formatı: {symbol}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def export_data(self) -> Dict[str, Any]:
        """Tüm veriyi export et"""
        return {
            'selected_symbols': self.selected_symbols,
            'permanent_high_ratio': self.permanent_high_ratio,
            'system_stats': self.get_system_stats(),
            'supertrend_stats': self.get_supertrend_statistics(),
            'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Veriyi import et - Backward compatibility"""
        try:
            if 'selected_symbols' in data:
                self.selected_symbols = data['selected_symbols']
            
            # Backward compatibility: permanent_high_consecutive -> permanent_high_ratio
            if 'permanent_high_ratio' in data:
                # Yeni formatı yükle ve eksik Devis'So alanları ekle
                self.permanent_high_ratio = []
                for symbol_data in data['permanent_high_ratio']:
                    # Eksik Devis'So alanlarını ekle
                    if 'deviso_lines' not in symbol_data:
                        symbol_data['deviso_lines'] = {
                            'ma_blue': 0.0,
                            'ma_yellow': 0.0,
                            'ma_purple': 0.0,
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                    
                    if 'deviso_status' not in symbol_data:
                        symbol_data['deviso_status'] = {
                            'has_line_contact': False,
                            'contact_level': 'None',
                            'signal_strength': 'None',
                            'alert_message': '',
                            'line_values': {}
                        }
                    
                    if 'deviso_contact_history' not in symbol_data:
                        symbol_data['deviso_contact_history'] = []
                    
                    # Eski reverse_momentum'u temizle
                    if 'reverse_momentum' in symbol_data:
                        del symbol_data['reverse_momentum']
                    
                    self.permanent_high_ratio.append(symbol_data)
                    
            elif 'permanent_high_consecutive' in data:
                # Eski formatı yeni formata dönüştür
                old_data = data['permanent_high_consecutive']
                self.permanent_high_ratio = []
                for old_symbol in old_data:
                    # Eski field'ları yeni field'lara map et
                    new_symbol = old_symbol.copy()
                    if 'max_consecutive_count' in old_symbol:
                        new_symbol['max_ratio_percent'] = old_symbol.get('max_percentage_change', 0)
                    if 'max_consecutive_type' in old_symbol:
                        old_type = old_symbol['max_consecutive_type']
                        new_symbol['max_supertrend_type'] = 'Bullish' if old_type == 'Long' else 'Bearish' if old_type == 'Short' else 'None'
                    if 'first_high_consecutive_date' in old_symbol:
                        new_symbol['first_high_ratio_date'] = old_symbol['first_high_consecutive_date']
                    
                    # Yeni Devis'So alanları ekle
                    new_symbol['deviso_lines'] = {
                        'ma_blue': 0.0,
                        'ma_yellow': 0.0,
                        'ma_purple': 0.0,
                        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    new_symbol['deviso_status'] = {
                        'has_line_contact': False,
                        'contact_level': 'None',
                        'signal_strength': 'None',
                        'alert_message': '',
                        'line_values': {}
                    }
                    new_symbol['deviso_contact_history'] = []
                    
                    # Eski reverse_momentum alanını temizle
                    if 'reverse_momentum' in new_symbol:
                        del new_symbol['reverse_momentum']
                    
                    self.permanent_high_ratio.append(new_symbol)
                
                logger.info("Eski format başarıyla yeni Devis'So formata dönüştürüldü")
            
            if 'system_stats' in data:
                # Eski istatistikleri güncelle
                old_stats = data['system_stats']
                self.system_stats.update(old_stats)
                
                # Eski reverse_momentum_detected -> deviso_line_contacts
                if 'reverse_momentum_detected' in old_stats:
                    self.system_stats['deviso_line_contacts'] = old_stats['reverse_momentum_detected']
                    if 'reverse_momentum_detected' in self.system_stats:
                        del self.system_stats['reverse_momentum_detected']
            
            logger.info("Veri başarıyla import edildi - Devis'So entegrasyonu tamamlandı")
            return True
            
        except Exception as e:
            logger.error(f"Veri import hatası: {e}")
            return False
    
    # =====================================================
    # BACKWARD COMPATIBILITY METHODS
    # =====================================================
    
    def get_reverse_momentum_symbols(self) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Geriye uyumluluk için - Devis'So temas sembolleri döndür
        
        Returns:
            List[Dict[str, Any]]: Çizgi teması olan semboller
        """
        logger.warning("get_reverse_momentum_symbols() DEPRECATED - get_deviso_contact_symbols() kullanın")
        return self.get_deviso_contact_symbols()
    
    def increment_reverse_momentum_count(self) -> None:
        """
        DEPRECATED: Geriye uyumluluk için - Devis'So temas sayacını artır
        """
        logger.warning("increment_reverse_momentum_count() DEPRECATED - increment_deviso_contact_count() kullanın")
        self.increment_deviso_contact_count()