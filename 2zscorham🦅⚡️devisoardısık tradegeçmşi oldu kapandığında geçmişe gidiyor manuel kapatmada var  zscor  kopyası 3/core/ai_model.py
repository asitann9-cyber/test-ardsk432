"""
🤖 AI Model Sistemi
Dengeli ve tutarlı machine learning modeli ve tahmin sistemi
🔥 YENİ: Deviso ratio odaklı dengeli skorlama
🔥 DÜZELTME: Pine Script uyumlu TEK Z-Score ceza sistemi
"""

import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config import LOCAL_TZ, AI_MODEL_FILE
from core.utils import gauss_sum

logger = logging.getLogger("crypto-analytics")


class CryptoMLModel:
    """🔥 DÜZELTME: Dengeli AI model sistemi + Pine Script Z-Score ceza sistemi"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'run_count', 'run_perc', 'gauss_run', 'gauss_run_perc',
            'log_volume', 'log_volume_momentum', 'deviso_ratio', 'trend_alignment'
        ]
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # Dengeli eğitim verisi oluştur
        self._create_balanced_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 Dengeli AI modeli oluşturuluyor...")
            self._train_initial_model()
    
    def _create_balanced_training_data(self):
        """🔥 YENİ: Dengeli eğitim verisi - gerçekçi skorlar (log hacim + momentum)"""
        demo_data = []
        
        # Mükemmel sinyaller (%85-95) - ÇOK NADIR
        for i in range(15):
            run_count = random.randint(6, 10)
            run_perc = random.uniform(5.0, 20.0)  
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            log_volume = random.uniform(8.0, 12.0)  
            log_volume_momentum = random.uniform(2.0, 5.0)  
            deviso_ratio = random.uniform(5.0, 25.0)  
            trend_alignment = 1  
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'log_volume': log_volume,
                'log_volume_momentum': log_volume_momentum,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.9
            })
        
        # İyi sinyaller (%70-85) - NADIR
        for i in range(25):
            run_count = random.randint(4, 6)
            run_perc = random.uniform(2.0, 8.0)
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            log_volume = random.uniform(6.0, 10.0)
            log_volume_momentum = random.uniform(1.0, 3.0)
            deviso_ratio = random.uniform(2.0, 8.0)
            trend_alignment = 1
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'log_volume': log_volume,
                'log_volume_momentum': log_volume_momentum,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.75
            })
        
        # Orta sinyaller (%50-70) - NORMAL
        for i in range(35):
            run_count = random.randint(3, 5)
            run_perc = random.uniform(1.0, 4.0)
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            log_volume = random.uniform(4.0, 8.0)
            log_volume_momentum = random.uniform(-1.0, 2.0)
            deviso_ratio = random.uniform(0.5, 3.0)
            trend_alignment = random.choice([0, 1])
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'log_volume': log_volume,
                'log_volume_momentum': log_volume_momentum,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.6
            })
        
        # Zayıf sinyaller (%20-50) - YAYGIN
        for i in range(50):
            run_count = random.randint(2, 4)
            run_perc = random.uniform(0.3, 2.0)  
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            log_volume = random.uniform(2.0, 6.0)  
            log_volume_momentum = random.uniform(-3.0, 1.0)  
            deviso_ratio = random.uniform(-2.0, 1.0)  
            trend_alignment = random.choice([0, 0, 1])  
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'log_volume': log_volume,
                'log_volume_momentum': log_volume_momentum,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.35
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} dengeli eğitim verisi oluşturuldu")

    
    def _train_initial_model(self):
        """🔥 YENİ: Regresyon tabanlı eğitim (0-1 aralığında)"""
        try:
            df = pd.DataFrame(self.training_data)
            X = df[self.feature_names].values
            y = df['target'].values  # 0-1 aralığında sürekli değerler
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Regresyon için RandomForest
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=150,  
                max_depth=10,       
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # R² score hesapla
            r2_score = self.model.score(X_scaled, y)
            self.model_stats['accuracy'] = r2_score
            self.model_stats['last_training'] = datetime.now(LOCAL_TZ).isoformat()
            
            logger.info(f"🎯 Dengeli AI modeli eğitildi: R²={r2_score:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """ML features oluştur - normalize edilmiş"""
        features = {}
        
        # 🔥 YENİ: Daha dengeli normalizasyon
        features['run_count'] = min(metrics.get('run_count', 0), 8) / 8.0  # 0-1 aralığı
        features['run_perc'] = min(abs(metrics.get('run_perc', 0.0) or 0.0), 15.0) / 15.0
        features['gauss_run'] = min(metrics.get('gauss_run', 0.0), 50.0) / 50.0
        features['gauss_run_perc'] = min(abs(metrics.get('gauss_run_perc', 0.0) or 0.0), 500.0) / 500.0
        
        # 🔥 Log Volume
        log_volume = abs(metrics.get('log_volume', 0.0) or 0.0)
        features['log_volume'] = min(log_volume, 15.0) / 15.0
        
        # 🔥 Log Volume Momentum
        log_volume_momentum = metrics.get('log_volume_momentum', 0.0) or 0.0
        features['log_volume_momentum'] = min(max(log_volume_momentum, 0), 5.0) / 5.0
        
        # Deviso ratio normalize et (-25 +25 aralığı)
        deviso_raw = metrics.get('deviso_ratio', 0.0)
        features['deviso_ratio'] = (max(-25, min(25, deviso_raw)) + 25) / 50.0  # 0-1 aralığı
        
        # Trend uyumu
        run_type = metrics.get('run_type', 'none')
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        features['trend_alignment'] = 1.0 if (
            (run_type == 'long' and deviso_ratio > 0.5) or
            (run_type == 'short' and deviso_ratio < -0.5)
        ) else 0.0
        
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array.reshape(1, -1)
    
    def predict_score(self, metrics: Dict) -> float:
        """🔥 DÜZELTME: Pine Script Z-Score ceza sistemi ile skorlama"""
        if not self.is_trained:
            return self._calculate_balanced_score(metrics)
       
        try:
            # ML model prediction
            features = self.create_features(metrics)
            features_scaled = self.scaler.transform(features)
           
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_score = np.clip(ml_prediction, 0.0, 1.0) * 100
           
            # Manuel skorlama
            manual_score = self._calculate_balanced_score(metrics)
           
            # İki skoru dengele (ML %60, Manuel %40)
            base_score = (ml_score * 0.6) + (manual_score * 0.4)
            
            # 🔥 DÜZELTME: Pine Script Z-Score ceza sistemi uygula
            final_score = self._apply_pine_zscore_penalty(base_score, metrics)
           
            return float(np.clip(final_score, 5.0, 95.0))
        except Exception as e:
            logger.debug(f"AI skor hesaplama hatası: {e}")
            return self._calculate_balanced_score(metrics)
    
    def _apply_pine_zscore_penalty(self, base_score: float, metrics: Dict) -> float:
        """
        🔥 DÜZELTME: Pine Script uyumlu Z-Score ceza sistemi
        Artık sadece tek Z-Score kullanıyor: (close - sma) / stdev
        
        Args:
            base_score (float): Temel AI skoru
            metrics (Dict): Metrikler (Pine Script z-score dahil)
            
        Returns:
            float: Cezalı skor
        """
        try:
            # 🔥 Pine Script Z-Score kullan (mutlak değer)
            pine_zscore = abs(metrics.get('max_zscore', 0.0))
            
            # 🔥 Pine Script Z-Score ceza tablosu
            if pine_zscore >= 3.0:
                penalty = 25  # Çok aşırı şişmiş
                logger.debug(f"🚨 Ağır Pine Z-Score cezası: -{penalty} (zscore: {pine_zscore:.2f})")
            elif pine_zscore >= 2.5:
                penalty = 20  # Aşırı şişmiş
                logger.debug(f"⚠️ Orta Pine Z-Score cezası: -{penalty} (zscore: {pine_zscore:.2f})")
            elif pine_zscore >= 2.0:
                penalty = 15  # Şişmiş
                logger.debug(f"⚠️ Hafif Pine Z-Score cezası: -{penalty} (zscore: {pine_zscore:.2f})")
            else:
                penalty = 0  # Normal
            
            final_score = base_score - penalty
            
            if penalty > 0:
                logger.debug(f"🎯 Pine Z-Score ceza uygulandı: {base_score:.1f} → {final_score:.1f} (ceza: -{penalty})")
            
            return final_score
            
        except Exception as e:
            logger.debug(f"Pine Z-Score ceza hesaplama hatası: {e}")
            return base_score
    
    def _calculate_balanced_score(self, metrics: Dict) -> float:
        """🔥 Güncellenmiş: Manuel skorlama (Toplam 100 puan)"""
        score = 0.0
        # 1. Deviso Ratio (30 puan)
        deviso_ratio = abs(metrics.get('deviso_ratio', 0.0))
        if deviso_ratio >= 10.0:
            deviso_score = 30
        elif deviso_ratio >= 5.0:
            deviso_score = 20 + (deviso_ratio - 5.0) / 5.0 * 10
        elif deviso_ratio >= 2.0:
            deviso_score = 10 + (deviso_ratio - 2.0) / 3.0 * 10
        else:
            deviso_score = min(deviso_ratio / 2.0 * 10, 5)
        score += deviso_score
        # 2. Ardışık Metrikler (20 puan)
        run_count = metrics.get('run_count', 0)
        run_perc = abs(metrics.get('run_perc', 0) or 0)
        score += min(run_count / 8.0, 1.0) * 10
        score += min(run_perc / 10.0, 1.0) * 10
        # 3. Hacim Analizi (35 puan) → EN YÜKSEK AĞIRLIK
        log_volume = abs(metrics.get('log_volume', 0.0) or 0.0)
        log_volume_momentum = metrics.get('log_volume_momentum', 0.0) or 0.0
        score += min(log_volume / 15.0, 1.0) * 20 # Log volume
        score += min(abs(log_volume_momentum) / 5.0, 1.0) * 15 # Momentum gücü
        # 4. Trend Uyumu (10 puan)
        run_type = metrics.get('run_type', 'none')
        original_deviso = metrics.get('deviso_ratio', 0.0)
        if ((run_type == 'long' and original_deviso > 0.5) or
            (run_type == 'short' and original_deviso < -0.5)):
            score += 10
        # 5. Momentum Yön Uyumu (±5 puan)
        if ((run_type == 'long' and log_volume_momentum > 0) or
            (run_type == 'short' and log_volume_momentum < 0)):
            score += 5 # uyum bonus
        elif ((run_type == 'long' and log_volume_momentum < 0) or
              (run_type == 'short' and log_volume_momentum > 0)):
            score -= 5 # uyumsuzluk cezası
        # 6. C-Signal Momentum (5 puan bonus)
        c_signal = metrics.get('c_signal_momentum', 0.0)
        if ((run_type == 'long' and c_signal > 0) or
            (run_type == 'short' and c_signal < 0)):
            score += min(abs(c_signal) / 10.0 * 5, 5)
        # 7. Kalite Cezaları (-10 max)
        penalty = 0
        if run_perc < 0.5:
            penalty += 5
        if deviso_ratio < 0.3:
            penalty += 5
        score -= penalty
        return max(5.0, min(score, 95.0))
    
    def save_model(self):
        """Model kaydet"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'model_stats': self.model_stats
            }
            joblib.dump(model_data, AI_MODEL_FILE)
            logger.info(f"💾 Dengeli AI modeli kaydedildi: {AI_MODEL_FILE}")
        except Exception as e:
            logger.debug(f"Model kaydetme hatası: {e}")
    
    def load_model(self):
        """Model yükle"""
        try:
            model_data = joblib.load(AI_MODEL_FILE)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            self.model_stats = model_data.get('model_stats', self.model_stats)
            logger.info(f"✅ Dengeli AI modeli yüklendi: R²={self.model_stats['accuracy']:.2%}")
        except:
            raise FileNotFoundError("Model dosyası bulunamadı")
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            'is_trained': self.is_trained,
            'accuracy': self.model_stats.get('accuracy', 0.0),
            'last_training': self.model_stats.get('last_training'),
            'features': self.feature_names,
            'training_data_size': len(self.training_data),
            'model_type': 'Balanced Regression + Pine Script Z-Score Penalty'
        }


# Global AI model instance
ai_model = CryptoMLModel()