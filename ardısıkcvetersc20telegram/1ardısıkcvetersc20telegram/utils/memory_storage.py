"""
Memory Storage Service
Bellekte veri depolama ve yönetim işlemleri
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryStorage:
    """Bellekte veri depolama sınıfı"""
    
    def __init__(self):
        """Memory storage'ı başlat"""
        self.selected_symbols: List[str] = []
        self.permanent_high_consecutive: List[Dict[str, Any]] = []
        self.analysis_cache: Dict[str, Any] = {}
        self.system_stats: Dict[str, Any] = {
            'total_analyses': 0,
            'last_analysis_time': None,
            'telegram_alerts_sent': 0,
            'reverse_momentum_detected': 0
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
    # PERMANENT HIGH CONSECUTIVE MANAGEMENT
    # =====================================================
    
    def add_permanent_symbol(self, symbol_data: Dict[str, Any]) -> None:
        """Kalıcı listeye sembol ekle"""
        symbol = symbol_data.get('symbol')
        if not symbol:
            logger.warning("Sembol adı bulunamadı, kalıcı listeye eklenemedi")
            return
        
        # Mevcut sembol kontrolü
        existing_symbol = self.get_permanent_symbol(symbol)
        
        if not existing_symbol:
            # Yeni sembol - kalıcı listeye ekle
            consecutive_count = symbol_data.get('consecutive_count', 0)
            percentage_change = abs(symbol_data.get('percentage_change', 0))
            
            # Ekleme nedenini belirle
            add_reasons = []
            if consecutive_count >= 5:
                add_reasons.append(f"{consecutive_count} ardışık")
            if percentage_change >= 10.0:
                add_reasons.append(f"%{percentage_change:.2f} değişim")
            
            permanent_entry = {
                'symbol': symbol,
                'first_high_consecutive_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'max_consecutive_count': symbol_data.get('consecutive_count', 0),
                'max_consecutive_type': symbol_data.get('consecutive_type', 'None'),
                'max_percentage_change': symbol_data.get('percentage_change', 0),
                'max_abs_percentage_change': abs(symbol_data.get('percentage_change', 0)),
                'timeframe': symbol_data.get('timeframe', '4h'),
                'tradingview_link': symbol_data.get('tradingview_link', '#'),
                'c_signal': None,
                'c_signal_update_time': None,
                'add_reason': " + ".join(add_reasons) if add_reasons else "Manuel ekleme",
                'reverse_momentum': {
                    'has_reverse_momentum': False,
                    'reverse_type': 'None',
                    'signal_strength': 'None',
                    'alert_message': '',
                    'signal_value': None
                },
                'last_telegram_alert': None
            }
            
            self.permanent_high_consecutive.append(permanent_entry)
            logger.info(f"🎯 {symbol} kalıcı listeye eklendi ({permanent_entry['add_reason']})")
        else:
            # Mevcut sembol - rekor kontrolü
            updated = False
            consecutive_count = symbol_data.get('consecutive_count', 0)
            percentage_change = abs(symbol_data.get('percentage_change', 0))
            
            # Ardışık sayı rekoru
            if consecutive_count > existing_symbol['max_consecutive_count']:
                existing_symbol['max_consecutive_count'] = consecutive_count
                existing_symbol['max_consecutive_type'] = symbol_data.get('consecutive_type', 'None')
                existing_symbol['timeframe'] = symbol_data.get('timeframe', '4h')
                updated = True
                logger.info(f"🔥 {symbol} ardışık rekoru kırdı! Yeni: {consecutive_count}")
            
            # Yüzde değişim rekoru
            if percentage_change > existing_symbol['max_abs_percentage_change']:
                existing_symbol['max_percentage_change'] = symbol_data.get('percentage_change', 0)
                existing_symbol['max_abs_percentage_change'] = percentage_change
                updated = True
                logger.info(f"📈 {symbol} yüzde rekoru kırdı! Yeni: %{percentage_change:.2f}")
            
            # Ekleme nedenini güncelle
            if updated:
                add_reasons = []
                if existing_symbol['max_consecutive_count'] >= 5:
                    add_reasons.append(f"{existing_symbol['max_consecutive_count']} ardışık")
                if existing_symbol['max_abs_percentage_change'] >= 10.0:
                    add_reasons.append(f"%{existing_symbol['max_abs_percentage_change']:.2f} değişim")
                existing_symbol['add_reason'] = " + ".join(add_reasons) if add_reasons else "Manuel ekleme"
    
    def get_permanent_symbols(self) -> List[Dict[str, Any]]:
        """Kalıcı sembolleri getir (sıralı)"""
        return sorted(
            self.permanent_high_consecutive, 
            key=lambda x: (x.get('max_consecutive_count', 0), x.get('max_abs_percentage_change', 0)), 
            reverse=True
        )
    
    def get_permanent_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Belirli kalıcı sembolü getir"""
        return next((s for s in self.permanent_high_consecutive if s['symbol'] == symbol), None)
    
    def update_permanent_symbol(self, symbol: str, update_data: Dict[str, Any]) -> bool:
        """Kalıcı sembol verisini güncelle"""
        existing_symbol = self.get_permanent_symbol(symbol)
        if existing_symbol:
            existing_symbol.update(update_data)
            return True
        return False
    
    def clear_permanent_symbols(self) -> int:
        """Tüm kalıcı sembolleri temizle"""
        count = len(self.permanent_high_consecutive)
        self.permanent_high_consecutive = []
        logger.info(f"{count} kalıcı sembol temizlendi")
        return count
    
    def get_reverse_momentum_symbols(self) -> List[Dict[str, Any]]:
        """Ters momentum olan sembolleri getir"""
        return [
            symbol for symbol in self.permanent_high_consecutive 
            if symbol.get('reverse_momentum', {}).get('has_reverse_momentum', False)
        ]
    
    def remove_permanent_symbol(self, symbol: str) -> bool:
        """Kalıcı listeden belirli sembolü çıkar"""
        try:
            initial_count = len(self.permanent_high_consecutive)
            self.permanent_high_consecutive = [
                s for s in self.permanent_high_consecutive 
                if s['symbol'] != symbol
            ]
            removed = len(self.permanent_high_consecutive) < initial_count
            
            if removed:
                logger.info(f"{symbol} kalıcı listeden çıkarıldı")
            
            return removed
            
        except Exception as e:
            logger.error(f"Kalıcı listeden çıkarma hatası: {e}")
            return False
    
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
    
    def increment_reverse_momentum_count(self) -> None:
        """Ters momentum tespit sayacını artır"""
        self.system_stats['reverse_momentum_detected'] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Sistem istatistiklerini getir"""
        return {
            **self.system_stats,
            'selected_symbols_count': len(self.selected_symbols),
            'permanent_symbols_count': len(self.permanent_high_consecutive),
            'cache_entries_count': len(self.analysis_cache),
            'reverse_momentum_symbols_count': len(self.get_reverse_momentum_symbols())
        }
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırla"""
        self.system_stats = {
            'total_analyses': 0,
            'last_analysis_time': None,
            'telegram_alerts_sent': 0,
            'reverse_momentum_detected': 0
        }
        logger.info("Sistem istatistikleri sıfırlandı")
    
    # =====================================================
    # UTILITY METHODS
    # =====================================================
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Bellek kullanım özetini getir"""
        return {
            'selected_symbols_count': len(self.selected_symbols),
            'permanent_symbols_count': len(self.permanent_high_consecutive),
            'cache_entries_count': len(self.analysis_cache),
            'total_memory_objects': (
                len(self.selected_symbols) + 
                len(self.permanent_high_consecutive) + 
                len(self.analysis_cache)
            )
        }
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Veri bütünlüğünü kontrol et"""
        issues = []
        
        # Permanent symbols validation
        for symbol_data in self.permanent_high_consecutive:
            if not symbol_data.get('symbol'):
                issues.append("Sembol adı eksik permanent symbol bulundu")
            
            if symbol_data.get('max_consecutive_count', 0) < 0:
                issues.append(f"Negatif ardışık sayı: {symbol_data.get('symbol')}")
        
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
            'permanent_high_consecutive': self.permanent_high_consecutive,
            'system_stats': self.get_system_stats(),
            'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Veriyi import et"""
        try:
            if 'selected_symbols' in data:
                self.selected_symbols = data['selected_symbols']
            
            if 'permanent_high_consecutive' in data:
                self.permanent_high_consecutive = data['permanent_high_consecutive']
            
            if 'system_stats' in data:
                self.system_stats.update(data['system_stats'])
            
            logger.info("Veri başarıyla import edildi")
            return True
            
        except Exception as e:
            logger.error(f"Veri import hatası: {e}")
            return False