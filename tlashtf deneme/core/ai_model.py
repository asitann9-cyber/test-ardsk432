"""
🤖 AI Model Sistemi - ULTRA PANEL v5
Machine learning modeli ve tahmin sistemi
🔥 YENİ: Ultra Panel v5 metrikleri ile dengeli skorlama
🔥 ÖZELLIKLER: HTF Count, Power, Whale detection bazlı AI scoring
"""

import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config import LOCAL_TZ, AI_MODEL_FILE

logger = logging.getLogger("crypto-analytics")


class CryptoMLModel:
    """🔥 Ultra Panel v5 AI Model - HTF + Power + Whale bazlı skorlama"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 🔥 YENİ FEATURE SET - Ultra Panel v5
        self.feature_names = [
            'htf_count',        # 0-4 (kaç HTF crossover var)
            'total_power',      # Kümülatif güç skoru
            'whale_active',     # 0 veya 1 (whale var mı)
            'power_per_htf',    # Power / HTF count ratio
            'signal_strength'   # Ultra signal gücü (normalized)
        ]
        
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # Dengeli eğitim verisi oluştur
        self._create_balanced_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 Ultra Panel v5 AI modeli oluşturuluyor...")
            self._train_initial_model()
    
    def _create_balanced_training_data(self):
        """🔥 YENİ: Ultra Panel v5 tabanlı dengeli eğitim verisi"""
        demo_data = []
        
        # 🔥 MÜKEMMEL SİNYALLER (%85-95) - 4/4 HTF + Yüksek Power + Whale
        for i in range(15):
            htf_count = 4  # 4/4 crossover
            total_power = random.uniform(20.0, 40.0)  # Çok yüksek power
            whale_active = 1  # Whale var
            power_per_htf = total_power / htf_count
            signal_strength = 1.0  # Maksimum
            
            demo_data.append({
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'power_per_htf': power_per_htf,
                'signal_strength': signal_strength,
                'target': 0.9  # %90 skor
            })
        
        # 🔥 İYİ SİNYALLER (%70-85) - 4/4 HTF + Orta Power veya 3/4 HTF + Yüksek Power
        for i in range(25):
            htf_count = random.choice([3, 4])
            
            if htf_count == 4:
                total_power = random.uniform(10.0, 20.0)  # Orta power
            else:
                total_power = random.uniform(15.0, 30.0)  # Yüksek power
            
            whale_active = random.choice([0, 1])  # Whale olabilir
            power_per_htf = total_power / htf_count
            signal_strength = 0.8
            
            demo_data.append({
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'power_per_htf': power_per_htf,
                'signal_strength': signal_strength,
                'target': 0.75  # %75 skor
            })
        
        # 🔥 ORTA SİNYALLER (%50-70) - 3/4 HTF + Orta Power
        for i in range(35):
            htf_count = 3  # 3/4 crossover
            total_power = random.uniform(5.0, 15.0)  # Orta power
            whale_active = random.choice([0, 0, 1])  # Whale nadiren
            power_per_htf = total_power / htf_count
            signal_strength = 0.6
            
            demo_data.append({
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'power_per_htf': power_per_htf,
                'signal_strength': signal_strength,
                'target': 0.6  # %60 skor
            })
        
        # 🔥 ZAYIF SİNYALLER (%20-50) - 3/4 HTF + Düşük Power veya sinyal yok
        for i in range(50):
            htf_count = random.choice([0, 1, 2, 3])  # Çoğunlukla zayıf
            
            if htf_count >= 3:
                total_power = random.uniform(1.0, 8.0)  # Düşük power
            else:
                total_power = random.uniform(0.5, 5.0)  # Çok düşük
            
            whale_active = 0  # Whale yok genelde
            power_per_htf = total_power / max(htf_count, 1)  # Sıfıra bölme koruması
            signal_strength = 0.3
            
            demo_data.append({
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'power_per_htf': power_per_htf,
                'signal_strength': signal_strength,
                'target': 0.35  # %35 skor
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} Ultra Panel v5 eğitim verisi oluşturuldu")
    
    def _train_initial_model(self):
        """🔥 YENİ: Ultra Panel v5 tabanlı model eğitimi"""
        try:
            df = pd.DataFrame(self.training_data)
            X = df[self.feature_names].values
            y = df['target'].values
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # RandomForest Regressor
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # R² score
            r2_score = self.model.score(X_scaled, y)
            self.model_stats['accuracy'] = r2_score
            self.model_stats['last_training'] = datetime.now(LOCAL_TZ).isoformat()
            
            logger.info(f"🎯 Ultra Panel v5 AI modeli eğitildi: R²={r2_score:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """🔥 YENİ: Ultra Panel v5 metrikleri → ML features"""
        features = {}
        
        # 1. HTF Count (0-4) → normalize
        htf_count = metrics.get('htf_count', 0)
        features['htf_count'] = min(htf_count, 4) / 4.0
        
        # 2. Total Power → normalize (0-50 arası bekleniyor)
        total_power = metrics.get('total_power', 0.0)
        features['total_power'] = min(total_power, 50.0) / 50.0
        
        # 3. Whale Active (0 veya 1)
        whale_active = 1.0 if metrics.get('whale_active', False) else 0.0
        features['whale_active'] = whale_active
        
        # 4. Power per HTF
        if htf_count > 0:
            features['power_per_htf'] = min(total_power / htf_count, 15.0) / 15.0
        else:
            features['power_per_htf'] = 0.0
        
        # 5. Signal Strength
        ultra_signal = metrics.get('ultra_signal', 'NONE')
        if ultra_signal == 'BUY' or ultra_signal == 'SELL':
            # 4/4 → 1.0, 3/4 → 0.75, 2/4 → 0.5, etc.
            features['signal_strength'] = min(htf_count / 4.0, 1.0)
        else:
            features['signal_strength'] = 0.0
        
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array.reshape(1, -1)
    
    def predict_score(self, metrics: Dict) -> float:
        """
        🔥 YENİ: Ultra Panel v5 metrikleri ile AI skor tahmini
        
        Args:
            metrics (Dict): Ultra Panel v5 metrikleri
            
        Returns:
            float: AI skoru (5-95 arası)
        """
        if not self.is_trained:
            return self._calculate_manual_score(metrics)
        
        try:
            # ML model prediction
            features = self.create_features(metrics)
            features_scaled = self.scaler.transform(features)
            
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_score = np.clip(ml_prediction, 0.0, 1.0) * 100
            
            # Manuel skorlama
            manual_score = self._calculate_manual_score(metrics)
            
            # İki skoru dengele (ML %70, Manuel %30)
            base_score = (ml_score * 0.7) + (manual_score * 0.3)
            
            # 🔥 Whale bonus
            if metrics.get('whale_active', False):
                base_score = min(base_score * 1.15, 95.0)  # %15 bonus
                logger.debug(f"🐋 Whale bonus uygulandı: {base_score:.1f}")
            
            return float(np.clip(base_score, 5.0, 95.0))
            
        except Exception as e:
            logger.debug(f"AI skor hesaplama hatası: {e}")
            return self._calculate_manual_score(metrics)
    
    def _calculate_manual_score(self, metrics: Dict) -> float:
        """
        🔥 YENİ: Ultra Panel v5 manuel skorlama
        Toplam 100 puan sistemi
        """
        score = 0.0
        
        # 1. HTF COUNT (40 puan) - EN ÖNEMLİ
        htf_count = metrics.get('htf_count', 0)
        if htf_count == 4:
            score += 40  # 4/4 mükemmel
        elif htf_count == 3:
            score += 30  # 3/4 iyi
        elif htf_count == 2:
            score += 15  # 2/4 zayıf
        elif htf_count == 1:
            score += 5   # 1/4 çok zayıf
        # 0/4 → 0 puan
        
        # 2. TOTAL POWER (35 puan)
        total_power = metrics.get('total_power', 0.0)
        if total_power >= 30.0:
            score += 35  # Çok yüksek power
        elif total_power >= 20.0:
            score += 28
        elif total_power >= 10.0:
            score += 20
        elif total_power >= 5.0:
            score += 12
        else:
            score += min(total_power / 5.0 * 12, 12)
        
        # 3. WHALE ACTIVE (15 puan bonus)
        if metrics.get('whale_active', False):
            score += 15
            logger.debug("🐋 Whale bonus: +15 puan")
        
        # 4. ULTRA SIGNAL CONSISTENCY (10 puan)
        ultra_signal = metrics.get('ultra_signal', 'NONE')
        if ultra_signal in ['BUY', 'SELL']:
            score += 10
        
        # 5. POWER PER HTF EFFICIENCY BONUS (maks +10 puan)
        if htf_count > 0:
            power_per_htf = total_power / htf_count
            if power_per_htf >= 10.0:
                score += 10  # Çok verimli
            elif power_per_htf >= 5.0:
                score += 5   # Verimli
        
        # Kalite cezası
        if htf_count == 0 or total_power < 1.0:
            score *= 0.5  # %50 ceza
        
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
            logger.info(f"💾 Ultra Panel v5 AI modeli kaydedildi: {AI_MODEL_FILE}")
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
            logger.info(f"✅ Ultra Panel v5 AI modeli yüklendi: R²={self.model_stats['accuracy']:.2%}")
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
            'model_type': 'Ultra Panel v5 - HTF + Power + Whale Detection'
        }


# Global AI model instance
ai_model = CryptoMLModel()