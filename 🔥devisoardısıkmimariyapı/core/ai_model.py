"""
🤖 AI Model Sistemi
Geliştirilmiş machine learning modeli ve tahmin sistemi
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
    """Geliştirilmiş AI model sistemi"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'run_count', 'run_perc', 'gauss_run', 'gauss_run_perc',
            'vol_ratio', 'hh_vol_streak', 'deviso_ratio', 'trend_alignment'
        ]
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # Geliştirilmiş eğitim verisi oluştur
        self._create_improved_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 Geliştirilmiş AI modeli oluşturuluyor...")
            self._train_initial_model()
    
    def _create_improved_training_data(self):
        """🔥 ÇÖZÜLDİ: Binary sınıflandırma için doğru etiketler"""
        demo_data = []
        
        # Yüksek kaliteli sinyaller (güçlü alım/satım) - LABEL: 1
        for i in range(50):
            run_count = random.randint(4, 8)
            run_perc = random.uniform(2.0, 15.0)  
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            vol_ratio = random.uniform(2.0, 4.0)  
            hh_vol_streak = random.randint(2, 6)  
            deviso_ratio = random.uniform(3.0, 20.0)  
            trend_alignment = 1  
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'vol_ratio': vol_ratio,
                'hh_vol_streak': hh_vol_streak,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 1  # ✅ Binary: 1 = Güçlü sinyal
            })
        
        # Orta kaliteli sinyaller - LABEL: 1 (ama daha az güçlü)
        for i in range(30):
            run_count = random.randint(3, 5)
            run_perc = random.uniform(1.0, 5.0)
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            vol_ratio = random.uniform(1.2, 2.5)
            hh_vol_streak = random.randint(1, 3)
            deviso_ratio = random.uniform(1.0, 8.0)
            trend_alignment = random.choice([0, 1])
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'vol_ratio': vol_ratio,
                'hh_vol_streak': hh_vol_streak,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 1  # ✅ Binary: 1 = Orta sinyal
            })
        
        # Düşük kaliteli sinyaller (zayıf/gürültü) - LABEL: 0
        for i in range(40):
            run_count = random.randint(1, 3)
            run_perc = random.uniform(0.1, 2.0)  
            gauss_run = gauss_sum(run_count)
            gauss_run_perc = gauss_sum(round(run_perc, 2))
            vol_ratio = random.uniform(0.5, 1.5)  
            hh_vol_streak = random.randint(0, 1)
            deviso_ratio = random.uniform(-5.0, 2.0)  
            trend_alignment = 0  
            
            demo_data.append({
                'run_count': run_count,
                'run_perc': run_perc,
                'gauss_run': gauss_run,
                'gauss_run_perc': gauss_run_perc,
                'vol_ratio': vol_ratio,
                'hh_vol_streak': hh_vol_streak,
                'deviso_ratio': deviso_ratio,
                'trend_alignment': trend_alignment,
                'target': 0  # ✅ Binary: 0 = Zayıf sinyal
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} geliştirilmiş eğitim verisi oluşturuldu")
    
    def _train_initial_model(self):
        """🔥 ÇÖZÜLDİ: Binary classification için düzeltilmiş eğitim"""
        try:
            df = pd.DataFrame(self.training_data)
            X = df[self.feature_names].values
            y = df['target'].values  # ✅ Şimdi sadece 0 ve 1 değerleri var
            
            # Target'ın binary olduğunu kontrol et
            unique_targets = np.unique(y)
            logger.info(f"📊 Target values: {unique_targets}")
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.model = RandomForestClassifier(
                n_estimators=100,  
                max_depth=8,       
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            accuracy = self.model.score(X_scaled, y)
            self.model_stats['accuracy'] = accuracy
            self.model_stats['last_training'] = datetime.now(LOCAL_TZ).isoformat()
            
            logger.info(f"🎯 Geliştirilmiş AI modeli eğitildi: Accuracy={accuracy:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """ML features oluştur"""
        features = {}
        
        features['run_count'] = min(metrics.get('run_count', 0), 10)
        features['run_perc'] = min(abs(metrics.get('run_perc', 0.0) or 0.0), 20.0)
        features['gauss_run'] = min(metrics.get('gauss_run', 0.0), 100.0)
        features['gauss_run_perc'] = min(abs(metrics.get('gauss_run_perc', 0.0) or 0.0), 1000.0)
        features['vol_ratio'] = min(metrics.get('vol_ratio', 1.0) or 1.0, 10.0)
        features['hh_vol_streak'] = min(metrics.get('hh_vol_streak', 0), 10)
        features['deviso_ratio'] = max(-50, min(50, metrics.get('deviso_ratio', 0.0)))
        
        # Trend uyumu hesapla
        run_type = metrics.get('run_type', 'none')
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        features['trend_alignment'] = 1 if (
            (run_type == 'long' and deviso_ratio > 0) or
            (run_type == 'short' and deviso_ratio < 0)
        ) else 0
        
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array.reshape(1, -1)
    
    def predict_score(self, metrics: Dict) -> float:
        """🔥 ÇÖZÜLDİ: Binary classification için düzeltilmiş skorlama"""
        if not self.is_trained:
            return self._fallback_score(metrics)
        
        try:
            features = self.create_features(metrics)
            features_scaled = self.scaler.transform(features)
            
            if hasattr(self.model, 'predict_proba'):
                # ✅ Binary classification probability
                probabilities = self.model.predict_proba(features_scaled)[0]
                # probabilities[1] = class 1 (good signal) probability
                positive_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                base_score = positive_prob * 100
            else:
                pred = self.model.predict(features_scaled)[0]
                base_score = pred * 100
            
            # Manuel bonus ekle
            bonus_score = self._calculate_manual_bonus(metrics)
            final_score = base_score + bonus_score
            
            return float(np.clip(final_score, 0.0, 100.0))
            
        except Exception as e:
            logger.debug(f"AI skor hesaplama hatası: {e}")
            return self._fallback_score(metrics)
    
    def _fallback_score(self, metrics: Dict) -> float:
        """Model yokken kullanılacak yedek skorlama"""
        score = 0.0
        
        # Run count bonusu
        run_count = metrics.get('run_count', 0)
        if run_count >= 5:
            score += 25
        elif run_count >= 3:
            score += 15
        
        # Run percentage bonusu
        run_perc = abs(metrics.get('run_perc', 0) or 0)
        if run_perc >= 3.0:
            score += 25
        elif run_perc >= 1.0:
            score += 15
        
        # Volume ratio bonusu
        vol_ratio = metrics.get('vol_ratio')
        if vol_ratio and vol_ratio >= 2.0:
            score += 20
        elif vol_ratio and vol_ratio >= 1.5:
            score += 10
        
        # Deviso trend uyumu
        deviso_ratio = metrics.get('deviso_ratio', 0)
        run_type = metrics.get('run_type', 'none')
        if ((run_type == 'long' and deviso_ratio > 2.0) or 
            (run_type == 'short' and deviso_ratio < -2.0)):
            score += 20
        
        # HH volume streak bonusu
        hh_vol_streak = metrics.get('hh_vol_streak', 0)
        if hh_vol_streak >= 3:
            score += 10
        
        return min(score, 100.0)
    
    def _calculate_manual_bonus(self, metrics: Dict) -> float:
        """Manuel bonus/ceza hesaplama"""
        bonus = 0.0
        
        # Güçlü deviso trend bonusu
        deviso_ratio = metrics.get('deviso_ratio', 0)
        run_type = metrics.get('run_type', 'none')
        
        if run_type == 'long' and deviso_ratio > 5.0:
            bonus += 15  
        elif run_type == 'short' and deviso_ratio < -5.0:
            bonus += 15  
        
        # Yüksek volume bonusu
        vol_ratio = metrics.get('vol_ratio')
        if vol_ratio and vol_ratio > 3.0:
            bonus += 10  
        
        # Uzun streak bonusu
        run_count = metrics.get('run_count', 0)
        if run_count >= 6:
            bonus += 10  
        
        # Büyük hareket bonusu
        run_perc = abs(metrics.get('run_perc', 0) or 0)
        if run_perc >= 5.0:
            bonus += 10  
        
        return bonus
    
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
            logger.info(f"💾 AI modeli kaydedildi: {AI_MODEL_FILE}")
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
            logger.info(f"✅ Geliştirilmiş AI modeli yüklendi: Accuracy={self.model_stats['accuracy']:.2%}")
        except:
            raise FileNotFoundError("Model dosyası bulunamadı")
    
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        return {
            'is_trained': self.is_trained,
            'accuracy': self.model_stats.get('accuracy', 0.0),
            'last_training': self.model_stats.get('last_training'),
            'features': self.feature_names,
            'training_data_size': len(self.training_data)
        }


# Global AI model instance
ai_model = CryptoMLModel()