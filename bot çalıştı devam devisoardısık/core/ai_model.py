"""
🤖 AI Model Sistemi
Dengeli ve tutarlı machine learning modeli ve tahmin sistemi
🔥 YENİ: RSI momentum + logaritmik hacim sistemi entegre edildi
🆕 ESKİ VOLUME SİSTEMİ KALDIRILDI
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
    """🆕 GÜNCELLENMIŞ: RSI momentum + logaritmik hacim sistemi ile AI model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 🆕 YENİ FEATURE LİSTESİ - RSI momentum + log volume
        self.feature_names = [
            'run_count', 'run_perc', 'gauss_run', 'gauss_run_perc',
            'rsi_momentum', 'momentum_score', 'log_volume_strength',  # 🆕 YENİ
            'deviso_ratio', 'trend_alignment'
            # ❌ KALDIRILDI: 'vol_ratio', 'hh_vol_streak'
        ]
        
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # Dengeli eğitim verisi oluştur
        self._create_balanced_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 RSI momentum + log volume AI modeli oluşturuluyor...")
            self._train_initial_model()
    
    def _create_balanced_training_data(self):
        """🆕 GÜNCELLENMIŞ: RSI momentum + log volume ile dengeli eğitim verisi"""
        demo_data = []
        
        # Mükemmel sinyaller (%85-95) - ÇOK NADIR
        for i in range(15):
            run_count = random.randint(6, 10)
            run_perc = random.uniform(5.0, 20.0)  
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            
            # 🆕 RSI MOMENTUM METRİKLERİ
            rsi_momentum = random.uniform(10.0, 20.0) * random.choice([1, -1])  # Güçlü momentum
            momentum_score = random.randint(80, 95)  # Yüksek momentum skoru
            log_volume_strength = random.uniform(3.0, 10.0)  # Güçlü hacim
            
            deviso_ratio = random.uniform(5.0, 25.0)  
            trend_alignment = 1  
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'rsi_momentum': rsi_momentum,
                'momentum_score': momentum_score,
                'log_volume_strength': log_volume_strength,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.9  # %90 başarı beklentisi
            })
        
        # İyi sinyaller (%70-85) - NADIR
        for i in range(25):
            run_count = random.randint(4, 6)
            run_perc = random.uniform(2.0, 8.0)
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            
            # 🆕 RSI MOMENTUM METRİKLERİ
            rsi_momentum = random.uniform(5.0, 12.0) * random.choice([1, -1])  # Orta momentum
            momentum_score = random.randint(60, 80)  # Orta momentum skoru
            log_volume_strength = random.uniform(2.0, 5.0)  # Orta hacim
            
            deviso_ratio = random.uniform(2.0, 8.0)
            trend_alignment = 1
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'rsi_momentum': rsi_momentum,
                'momentum_score': momentum_score,
                'log_volume_strength': log_volume_strength,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.75  # %75 başarı beklentisi
            })
        
        # Orta sinyaller (%50-70) - NORMAL
        for i in range(35):
            run_count = random.randint(3, 5)
            run_perc = random.uniform(1.0, 4.0)
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            
            # 🆕 RSI MOMENTUM METRİKLERİ
            rsi_momentum = random.uniform(2.0, 8.0) * random.choice([1, -1])  # Zayıf momentum
            momentum_score = random.randint(40, 65)  # Zayıf momentum skoru
            log_volume_strength = random.uniform(1.0, 3.0)  # Zayıf hacim
            
            deviso_ratio = random.uniform(0.5, 3.0)
            trend_alignment = random.choice([0, 1])
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'rsi_momentum': rsi_momentum,
                'momentum_score': momentum_score,
                'log_volume_strength': log_volume_strength,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.6  # %60 başarı beklentisi
            })
        
        # Zayıf sinyaller (%20-50) - YAYGIN
        for i in range(50):
            run_count = random.randint(2, 4)
            run_perc = random.uniform(0.3, 2.0)  
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            
            # 🆕 RSI MOMENTUM METRİKLERİ
            rsi_momentum = random.uniform(0.1, 3.0) * random.choice([1, -1])  # Çok zayıf momentum
            momentum_score = random.randint(10, 40)  # Düşük momentum skoru
            log_volume_strength = random.uniform(0.1, 2.0)  # Çok zayıf hacim
            
            deviso_ratio = random.uniform(-2.0, 1.0)  
            trend_alignment = random.choice([0, 0, 1])  # Çoğunlukla uyumsuz
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'rsi_momentum': rsi_momentum,
                'momentum_score': momentum_score,
                'log_volume_strength': log_volume_strength,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0.35  # %35 başarı beklentisi
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} RSI momentum + log volume eğitim verisi oluşturuldu")
    
    def _train_initial_model(self):
        """🔄 AYNI: Regresyon tabanlı eğitim (0-1 aralığında)"""
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
            
            logger.info(f"🎯 RSI momentum + log volume AI modeli eğitildi: R²={r2_score:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """🆕 GÜNCELLENMIŞ: RSI momentum + log volume features"""
        features = {}
        
        # Ardışık metrikler (aynı kalıyor)
        features['run_count'] = min(metrics.get('run_count', 0), 8) / 8.0  # 0-1 aralığı
        features['run_perc'] = min(abs(metrics.get('run_perc', 0.0) or 0.0), 15.0) / 15.0
        features['gauss_run'] = min(metrics.get('gauss_run', 0.0), 50.0) / 50.0
        features['gauss_run_perc'] = min(abs(metrics.get('gauss_run_perc', 0.0) or 0.0), 500.0) / 500.0
        
        # 🆕 RSI MOMENTUM FEATURES
        rsi_momentum_raw = metrics.get('rsi_momentum', 0.0)
        features['rsi_momentum'] = (max(-20, min(20, rsi_momentum_raw)) + 20) / 40.0  # 0-1 aralığı (-20,+20 -> 0,1)
        
        momentum_score = metrics.get('momentum_score', 0)
        features['momentum_score'] = min(momentum_score, 100) / 100.0  # 0-1 aralığı
        
        # 🆕 LOG VOLUME FEATURES
        log_vol_strength = metrics.get('log_volume_strength', 0.0)
        features['log_volume_strength'] = min(log_vol_strength, 10.0) / 10.0  # 0-1 aralığı
        
        # Deviso ratio (aynı kalıyor)
        deviso_raw = metrics.get('deviso_ratio', 0.0)
        features['deviso_ratio'] = (max(-25, min(25, deviso_raw)) + 25) / 50.0  # 0-1 aralığı
        
        # Trend uyumu (aynı kalıyor)
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
        """🔄 AYNI: Dengeli skorlama sistemi"""
        if not self.is_trained:
            return self._calculate_balanced_score(metrics)
        
        try:
            # ML model prediction
            features = self.create_features(metrics)
            features_scaled = self.scaler.transform(features)
            
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_score = np.clip(ml_prediction, 0.0, 1.0) * 100
            
            # 🔥 YENİ: Manuel dengeli skorlama
            manual_score = self._calculate_balanced_score(metrics)
            
            # İki skoru dengele (ML %60, Manuel %40)
            final_score = (ml_score * 0.6) + (manual_score * 0.4)
            
            return float(np.clip(final_score, 5.0, 95.0))  # 5-95 aralığında sınırla
            
        except Exception as e:
            logger.debug(f"AI skor hesaplama hatası: {e}")
            return self._calculate_balanced_score(metrics)
    
    def _calculate_balanced_score(self, metrics: Dict) -> float:
        """🆕 GÜNCELLENMIŞ: RSI momentum + log volume manuel skorlama (Toplam 100 puan)"""
        score = 0.0
        
        # 1. Deviso Ratio (30 puan) - Hala önemli ama azaltıldı
        deviso_ratio = abs(metrics.get('deviso_ratio', 0.0))
        
        if deviso_ratio >= 10.0:
            deviso_score = 30
        elif deviso_ratio >= 5.0:
            deviso_score = 22 + (deviso_ratio - 5.0) / 5.0 * 8  # 22-30 arası
        elif deviso_ratio >= 2.0:
            deviso_score = 15 + (deviso_ratio - 2.0) / 3.0 * 7  # 15-22 arası
        elif deviso_ratio >= 0.5:
            deviso_score = 5 + (deviso_ratio - 0.5) / 1.5 * 10   # 5-15 arası
        else:
            deviso_score = deviso_ratio / 0.5 * 5                # 0-5 arası
            
        score += deviso_score
        
        # 2. RSI Momentum (25 puan) - YENİ ANA KRİTER
        rsi_momentum = metrics.get('rsi_momentum', 0.0)
        momentum_score = metrics.get('momentum_score', 0)
        
        # RSI momentum gücü (15 puan)
        abs_rsi_momentum = abs(rsi_momentum)
        if abs_rsi_momentum >= 15:
            rsi_score = 15
        elif abs_rsi_momentum >= 10:
            rsi_score = 10 + (abs_rsi_momentum - 10) / 5.0 * 5  # 10-15 arası
        elif abs_rsi_momentum >= 5:
            rsi_score = 5 + (abs_rsi_momentum - 5) / 5.0 * 5    # 5-10 arası
        else:
            rsi_score = abs_rsi_momentum / 5.0 * 5               # 0-5 arası
        
        score += rsi_score
        
        # Momentum score (10 puan)
        mom_score = min(momentum_score / 100.0, 1.0) * 10
        score += mom_score
        
        # 3. Logaritmik Hacim (20 puan) - YENİ ÖNEMLI KRİTER
        log_vol_strength = metrics.get('log_volume_strength', 0.0)
        
        if log_vol_strength >= 5.0:
            log_vol_score = 20
        elif log_vol_strength >= 3.0:
            log_vol_score = 15 + (log_vol_strength - 3.0) / 2.0 * 5  # 15-20 arası
        elif log_vol_strength >= 1.0:
            log_vol_score = 10 + (log_vol_strength - 1.0) / 2.0 * 5  # 10-15 arası
        else:
            log_vol_score = log_vol_strength / 1.0 * 10               # 0-10 arası
        
        score += log_vol_score
        
        # 4. Ardışık Metrikler (15 puan) - Azaltıldı
        run_count = metrics.get('run_count', 0)
        run_perc = abs(metrics.get('run_perc', 0) or 0)
        
        # Run count (8 puan)
        run_score = min(run_count / 8.0, 1.0) * 8
        score += run_score
        
        # Run percentage (7 puan)
        perc_score = min(run_perc / 10.0, 1.0) * 7
        score += perc_score
        
        # 5. Trend Uyumu (10 puan) - Azaltıldı
        run_type = metrics.get('run_type', 'none')
        original_deviso = metrics.get('deviso_ratio', 0.0)
        
        if ((run_type == 'long' and original_deviso > 1.0) or 
            (run_type == 'short' and original_deviso < -1.0)):
            trend_score = 10  # Güçlü uyum
        elif ((run_type == 'long' and original_deviso > 0.3) or 
              (run_type == 'short' and original_deviso < -0.3)):
            trend_score = 6   # Orta uyum
        elif ((run_type == 'long' and original_deviso > 0) or 
              (run_type == 'short' and original_deviso < 0)):
            trend_score = 3   # Zayıf uyum
        else:
            trend_score = 0   # Uyumsuzluk
            
        score += trend_score
        
        # 6. Kalite Cezaları (maksimum -10 puan)
        penalty = 0
        
        # Çok düşük RSI momentum cezası
        if abs(rsi_momentum) < 1.0:
            penalty += 5
            
        # Çok düşük log volume cezası
        if log_vol_strength < 0.5:
            penalty += 5
            
        score -= penalty
        
        return max(5.0, min(score, 95.0))  # 5-95 aralığında sınırla
    
    def save_model(self):
        """🔄 AYNI: Model kaydet"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'model_stats': self.model_stats
            }
            joblib.dump(model_data, AI_MODEL_FILE)
            logger.info(f"💾 RSI momentum + log volume AI modeli kaydedildi: {AI_MODEL_FILE}")
        except Exception as e:
            logger.debug(f"Model kaydetme hatası: {e}")
    
    def load_model(self):
        """🔄 AYNI: Model yükle"""
        try:
            model_data = joblib.load(AI_MODEL_FILE)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            self.model_stats = model_data.get('model_stats', self.model_stats)
            logger.info(f"✅ RSI momentum + log volume AI modeli yüklendi: R²={self.model_stats['accuracy']:.2%}")
        except:
            raise FileNotFoundError("Model dosyası bulunamadı")
    
    def get_model_info(self) -> Dict:
        """🔄 AYNI: Model bilgilerini döndür"""
        return {
            'is_trained': self.is_trained,
            'accuracy': self.model_stats.get('accuracy', 0.0),
            'last_training': self.model_stats.get('last_training'),
            'features': self.feature_names,
            'training_data_size': len(self.training_data),
            'model_type': 'RSI Momentum + Log Volume Regression'
        }


# Global AI model instance
ai_model = CryptoMLModel()