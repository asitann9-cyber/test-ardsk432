"""
🎯 Deviso Manager - Timeframe Adaptive Bot Koordinasyonu
Deviso Live Test ve Live Trading botlarını koordine eden ana yönetici
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from config import (
    LOCAL_TZ, DEVISO_LIVE_TEST, DEVISO_LIVE_TRADING, DEVISO_TIMEFRAME_SIGNALS,
    get_strategy_from_timeframe, get_adaptive_ai_threshold, get_scan_interval_for_strategy,
    deviso_current_timeframe, deviso_current_strategy
)

logger = logging.getLogger("crypto-analytics")


class DevisoManager:
    """🎯 Deviso bot sistemi koordinatörü"""
    
    def __init__(self):
        self.live_test_bot = None
        self.live_trading_bot = None
        self.current_timeframe = deviso_current_timeframe
        self.current_strategy = deviso_current_strategy
        
        # Bot durumları
        self.live_test_active = False
        self.live_trading_active = False
        
        logger.info("🎯 Deviso Manager başlatıldı")
        logger.info(f"📊 Başlangıç timeframe: {self.current_timeframe}")
        logger.info(f"🔄 Başlangıç strateji: {self.current_strategy}")
    
    def update_timeframe_settings(self, timeframe: str, min_ai_score: float):
        """
        UI'den gelen timeframe değişikliklerini botlara ilet
        
        Args:
            timeframe (str): Yeni timeframe (1m, 5m, 15m, 1h, 4h)
            min_ai_score (float): Minimum AI skoru (0-100)
        """
        try:
            old_timeframe = self.current_timeframe
            old_strategy = self.current_strategy
            
            # Yeni strateji belirleme
            self.current_timeframe = timeframe
            self.current_strategy = get_strategy_from_timeframe(timeframe)
            
            # Strateji değişimi kontrolü
            strategy_changed = old_strategy != self.current_strategy
            
            if strategy_changed:
                logger.info(f"🔄 Strateji değişimi: {old_strategy} → {self.current_strategy}")
                logger.info(f"📊 Timeframe: {old_timeframe} → {timeframe}")
            
            # Adaptive eşikler
            recommended_ai = get_adaptive_ai_threshold(self.current_strategy)
            scan_interval = get_scan_interval_for_strategy(self.current_strategy)
            
            logger.info(f"🎯 Deviso timeframe güncellendi:")
            logger.info(f"   📊 Timeframe: {timeframe}")
            logger.info(f"   🔄 Strateji: {self.current_strategy.upper()}")
            logger.info(f"   🤖 Önerilen AI eşiği: {recommended_ai}%")
            logger.info(f"   ⏱️ Tarama aralığı: {scan_interval}s")
            
            # Botlara timeframe güncellemesi gönder
            if self.live_test_bot and self.live_test_active:
                self.live_test_bot.update_timeframe_settings(timeframe, min_ai_score)
                logger.info("✅ Live Test bot timeframe güncellendi")
                
            if self.live_trading_bot and self.live_trading_active:
                self.live_trading_bot.update_timeframe_settings(timeframe, min_ai_score)
                logger.info("✅ Live Trading bot timeframe güncellendi")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Timeframe güncelleme hatası: {e}")
            return False
    
    def start_live_test(self) -> bool:
        """Deviso Live Test (Demo) bot'unu başlat"""
        try:
            if self.live_test_active:
                logger.warning("⚠️ Deviso Live Test zaten aktif")
                return False
            
            # Live Test bot'unu import et ve başlat
            from trading.deviso_live_test import DevisoLiveTest
            
            self.live_test_bot = DevisoLiveTest()
            success = self.live_test_bot.start()
            
            if success:
                self.live_test_active = True
                logger.info("🧪 Deviso Live Test başarıyla başlatıldı")
                logger.info(f"💰 Demo bakiye: ${DEVISO_LIVE_TEST['demo_balance']:.2f}")
                logger.info(f"📊 Strateji: {self.current_strategy.upper()}")
                return True
            else:
                logger.error("❌ Deviso Live Test başlatılamadı")
                return False
                
        except Exception as e:
            logger.error(f"❌ Deviso Live Test başlatma hatası: {e}")
            return False
    
    def start_live_trading(self) -> bool:
        """Deviso Live Trading (Gerçek) bot'unu başlat"""
        try:
            if self.live_trading_active:
                logger.warning("⚠️ Deviso Live Trading zaten aktif")
                return False
            
            # Live Trading bot'unu import et ve başlat
            from trading.deviso_live_trading import DevisoLiveTrading
            
            self.live_trading_bot = DevisoLiveTrading()
            success = self.live_trading_bot.start()
            
            if success:
                self.live_trading_active = True
                logger.info("💰 Deviso Live Trading başarıyla başlatıldı")
                logger.info(f"🏦 Futures bakiye: ${DEVISO_LIVE_TRADING['futures_balance']:.2f}")
                logger.info(f"📊 Strateji: {self.current_strategy.upper()}")
                logger.warning("⚠️ GERÇEK PARA ile işlem yapılıyor!")
                return True
            else:
                logger.error("❌ Deviso Live Trading başlatılamadı")
                return False
                
        except Exception as e:
            logger.error(f"❌ Deviso Live Trading başlatma hatası: {e}")
            return False
    
    def stop_live_test(self):
        """Deviso Live Test bot'unu durdur"""
        try:
            if self.live_test_bot and self.live_test_active:
                self.live_test_bot.stop()
                self.live_test_active = False
                logger.info("🧪 Deviso Live Test durduruldu")
            else:
                logger.info("ℹ️ Deviso Live Test zaten durdurulmuş")
        except Exception as e:
            logger.error(f"❌ Deviso Live Test durdurma hatası: {e}")
    
    def stop_live_trading(self):
        """Deviso Live Trading bot'unu durdur"""
        try:
            if self.live_trading_bot and self.live_trading_active:
                self.live_trading_bot.stop()
                self.live_trading_active = False
                logger.info("💰 Deviso Live Trading durduruldu")
            else:
                logger.info("ℹ️ Deviso Live Trading zaten durdurulmuş")
        except Exception as e:
            logger.error(f"❌ Deviso Live Trading durdurma hatası: {e}")
    
    def stop_all_bots(self):
        """Tüm Deviso botlarını durdur"""
        logger.info("⛔ Tüm Deviso botları durduruluyor...")
        self.stop_live_test()
        self.stop_live_trading()
        logger.info("✅ Tüm Deviso botları durduruldu")
    
    def get_status_text(self) -> str:
        """Deviso durumunu metin olarak al"""
        status_parts = []
        
        if self.live_test_active:
            status_parts.append("🧪 Live Test: AKTİF")
        else:
            status_parts.append("🧪 Live Test: DURDURULDU")
        
        if self.live_trading_active:
            status_parts.append("💰 Live Trading: AKTİF")
        else:
            status_parts.append("💰 Live Trading: DURDURULDU")
        
        status_parts.append(f"📊 Strateji: {self.current_strategy.upper()}")
        
        return " | ".join(status_parts)
    
    def get_brief_status(self) -> str:
        """Kısa durum metni"""
        active_count = sum([self.live_test_active, self.live_trading_active])
        return f"{active_count}/2 aktif"
    
    def get_current_strategy_info(self) -> Dict:
        """Mevcut strateji bilgilerini al"""
        analysis_timeframes = DEVISO_TIMEFRAME_SIGNALS['analysis_combinations'].get(
            self.current_timeframe, [self.current_timeframe]
        )
        
        return {
            'timeframe': self.current_timeframe,
            'strategy': self.current_strategy,
            'recommended_ai': get_adaptive_ai_threshold(self.current_strategy),
            'scan_interval': get_scan_interval_for_strategy(self.current_strategy),
            'analysis_timeframes': analysis_timeframes
        }
    
    def get_top3_analysis(self) -> Dict:
        """En iyi 3 seçimi detaylı analizi"""
        analysis = {
            'live_test_top3': [],
            'live_trading_top3': []
        }
        
        try:
            if self.live_test_bot and self.live_test_active:
                analysis['live_test_top3'] = self.live_test_bot.get_last_top3_selection()
            
            if self.live_trading_bot and self.live_trading_active:
                analysis['live_trading_top3'] = self.live_trading_bot.get_last_top3_selection()
                
        except Exception as e:
            logger.debug(f"Top3 analiz hatası: {e}")
        
        return analysis
    
    def get_timeframe_performance(self) -> Dict:
        """Timeframe bazlı performans analizi"""
        try:
            performance = {
                'current_timeframe': self.current_timeframe,
                'current_strategy': self.current_strategy,
                'live_test_performance': {},
                'live_trading_performance': {},
                'best_performing_timeframe': None,
                'recommendation': None
            }
            
            if self.live_test_bot and self.live_test_active:
                performance['live_test_performance'] = self.live_test_bot.get_timeframe_stats()
            
            if self.live_trading_bot and self.live_trading_active:
                performance['live_trading_performance'] = self.live_trading_bot.get_timeframe_stats()
            
            # En iyi performans gösteren timeframe analizi
            performance['best_performing_timeframe'] = self._analyze_best_timeframe()
            performance['recommendation'] = self._get_timeframe_recommendation()
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Timeframe performans analizi hatası: {e}")
            return {}
    
    def _analyze_best_timeframe(self) -> Optional[str]:
        """En iyi performans gösteren timeframe'i analiz et"""
        # Bu fonksiyon gelecekte trade geçmişi analizi yapabilir
        # Şimdilik mevcut timeframe'i döndür
        return self.current_timeframe
    
    def _get_timeframe_recommendation(self) -> Optional[str]:
        """Timeframe önerisi"""
        try:
            current_hour = datetime.now(LOCAL_TZ).hour
            
            # Saat bazlı öneriler
            if 9 <= current_hour <= 11:  # Sabah volatilite
                return "Sabah volatilitesi için 5m-15m timeframe önerilir"
            elif 14 <= current_hour <= 16:  # Öğleden sonra volatilite
                return "Öğleden sonra için 15m-1h timeframe önerilir"
            elif 20 <= current_hour <= 22:  # Akşam volatilite
                return "Akşam seansı için 1h-4h timeframe önerilir"
            else:
                return "Düşük volatilite döneminde 1h+ timeframe önerilir"
                
        except Exception as e:
            logger.debug(f"Timeframe önerisi hatası: {e}")
            return "Mevcut timeframe uygun görünüyor"
    
    def get_full_status_report(self) -> Dict:
        """Tam durum raporu"""
        return {
            'timestamp': datetime.now(LOCAL_TZ).isoformat(),
            'manager_status': {
                'current_timeframe': self.current_timeframe,
                'current_strategy': self.current_strategy,
                'live_test_active': self.live_test_active,
                'live_trading_active': self.live_trading_active
            },
            'strategy_info': self.get_current_strategy_info(),
            'top3_analysis': self.get_top3_analysis(),
            'timeframe_performance': self.get_timeframe_performance(),
            'status_text': self.get_status_text()
        }


# Global instance (config.py'den import edilecek)
deviso_manager_instance = None


def get_deviso_manager():
    """Global Deviso Manager instance'ını al"""
    global deviso_manager_instance
    
    if deviso_manager_instance is None:
        deviso_manager_instance = DevisoManager()
    
    return deviso_manager_instance