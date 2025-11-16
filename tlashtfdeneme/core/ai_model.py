"""
🤖 AI Model Sistemi - Ultra Panel v5 Multi-HTF Tabanlı
🔥 Heikin Ashi Multi-Timeframe analizi bazlı skorlama
🔥 Ultra Signal (3/4 HTF crossover) + Candle Power
🔥 Whale Detection entegrasyonu
🔥 ESKİ SİSTEM TAMAMEN KALDIRILDI: VPMV, Gauss, Deviso, Z-Score
"""

import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config import LOCAL_TZ, AI_MODEL_FILE

logger = logging.getLogger("crypto-analytics")


class CryptoMLModel:
    """🔥 Ultra Panel v5: Multi-HTF bazlı AI model sistemi"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'bull_count',        # 0-4 (kaç HTF bull)
            'bear_count',        # 0-4 (kaç HTF bear)
            'htf_count',         # 0-4 (max(bull, bear))
            'total_power',       # 0-100+ (candle power toplamı)
            'whale_active'       # 0-1 (whale detection)
        ]
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # Dengeli eğitim verisi oluştur
        self._create_balanced_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 Ultra Panel AI modeli oluşturuluyor (Multi-HTF)...")
            self._train_initial_model()
    
    def _create_balanced_training_data(self):
        """
        🔥 Ultra Panel v5: Multi-HTF bazlı dengeli eğitim verisi
        
        Skor Dağılımı:
        - Mükemmel (85-95): 15 örnek - 4/4 Ultra + Yüksek Power + Whale
        - İyi (70-85): 25 örnek - 3/4 veya 4/4 Ultra + Orta Power
        - Orta (50-70): 35 örnek - 3/4 Ultra + Düşük Power
        - Zayıf (20-50): 50 örnek - Düşük HTF + Düşük Power
        """
        demo_data = []
        
        # ======================================
        # 1. MÜKEMMEL SİNYALLER (85-95 puan)
        # 4/4 Ultra + Yüksek Power + Whale
        # ======================================
        for i in range(15):
            # 4/4 Perfect Ultra Signal
            is_bull = random.choice([True, False])
            
            if is_bull:
                bull_count = 4
                bear_count = 0
            else:
                bull_count = 0
                bear_count = 4
            
            htf_count = 4
            total_power = random.uniform(25.0, 50.0)  # Yüksek power
            whale_active = 1.0  # Whale var
            
            demo_data.append({
                'bull_count': bull_count,
                'bear_count': bear_count,
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'target': 0.9  # %90 skor
            })
        
        # ======================================
        # 2. İYİ SİNYALLER (70-85 puan)
        # 3/4 veya 4/4 Ultra + Orta Power
        # ======================================
        for i in range(25):
            is_bull = random.choice([True, False])
            htf_count = random.choice([3, 4])  # 3/4 veya 4/4
            
            if is_bull:
                bull_count = htf_count
                bear_count = 0
            else:
                bull_count = 0
                bear_count = htf_count
            
            total_power = random.uniform(15.0, 30.0)  # Orta power
            whale_active = random.choice([0.0, 1.0])  # Whale olabilir
            
            demo_data.append({
                'bull_count': bull_count,
                'bear_count': bear_count,
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'target': 0.75  # %75 skor
            })
        
        # ======================================
        # 3. ORTA SİNYALLER (50-70 puan)
        # 3/4 Ultra + Düşük Power
        # ======================================
        for i in range(35):
            is_bull = random.choice([True, False])
            htf_count = 3  # 3/4 Ultra
            
            if is_bull:
                bull_count = 3
                bear_count = 0
            else:
                bull_count = 0
                bear_count = 3
            
            total_power = random.uniform(8.0, 18.0)  # Düşük-orta power
            whale_active = 0.0  # Whale yok genelde
            
            demo_data.append({
                'bull_count': bull_count,
                'bear_count': bear_count,
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'target': 0.6  # %60 skor
            })
        
        # ======================================
        # 4. ZAYIF SİNYALLER (20-50 puan)
        # Düşük HTF + Düşük Power
        # ======================================
        for i in range(50):
            is_bull = random.choice([True, False])
            htf_count = random.choice([0, 1, 2])  # Zayıf HTF
            
            if is_bull:
                bull_count = htf_count
                bear_count = 0
            else:
                bull_count = 0
                bear_count = htf_count
            
            total_power = random.uniform(0.0, 10.0)  # Çok düşük power
            whale_active = 0.0  # Whale yok
            
            demo_data.append({
                'bull_count': bull_count,
                'bear_count': bear_count,
                'htf_count': htf_count,
                'total_power': total_power,
                'whale_active': whale_active,
                'target': 0.35  # %35 skor
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} Ultra Panel eğitim verisi oluşturuldu (Multi-HTF)")
    
    def _train_initial_model(self):
        """🔥 RandomForestRegressor ile Ultra Panel model eğitimi"""
        try:
            df = pd.DataFrame(self.training_data)
            X = df[self.feature_names].values
            y = df['target'].values  # 0-1 aralığı
            
            # Normalizasyon
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
            
            logger.info(f"🎯 Ultra Panel AI modeli eğitildi (Multi-HTF): R²={r2_score:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """
        🔥 Ultra Panel v5: Multi-HTF metriklerinden ML features oluştur
        
        Args:
            metrics (Dict): compute_ultra_metrics() çıktısı
            
        Returns:
            np.array: Normalize edilmiş feature array (5 feature)
        """
        features = {}
        
        # 1. Bull Count (0-4) → (0, 1)
        bull_count = int(metrics.get('bull_count', 0))
        features['bull_count'] = bull_count / 4.0
        
        # 2. Bear Count (0-4) → (0, 1)
        bear_count = int(metrics.get('bear_count', 0))
        features['bear_count'] = bear_count / 4.0
        
        # 3. HTF Count (0-4) → (0, 1)
        htf_count = int(metrics.get('htf_count', 0))
        features['htf_count'] = htf_count / 4.0
        
        # 4. Total Power (0-100+) → (0, 1)
        # Power genelde 0-50 arası, 50+ çok güçlü
        total_power = float(metrics.get('total_power', 0.0))
        features['total_power'] = min(total_power / 50.0, 1.0)
        
        # 5. Whale Active (0 or 1) → (0, 1)
        whale_active = 1.0 if metrics.get('whale_active', False) else 0.0
        features['whale_active'] = whale_active
        
        # Array'e çevir
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.0)  # NaN → 0
        
        return feature_array.reshape(1, -1)
    
    def predict_score(self, metrics: Dict) -> float:
        """
        🔥 Ultra Panel v5: Multi-HTF bazlı AI skorlama
        
        Skor Hesaplama:
        - ML Model: %60
        - Manuel Skor: %40
        
        Args:
            metrics (Dict): Ultra Panel metrikleri (HTF + Power + Whale)
            
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
            
            # Birleştir (ML %60, Manuel %40)
            final_score = (ml_score * 0.6) + (manual_score * 0.4)
            
            return float(np.clip(final_score, 5.0, 95.0))
            
        except Exception as e:
            logger.debug(f"AI skor hesaplama hatası: {e}")
            return self._calculate_manual_score(metrics)
    
    def _calculate_manual_score(self, metrics: Dict) -> float:
        """
        🔥 Ultra Panel v5: Manuel Multi-HTF skorlama sistemi
        
        Puan Dağılımı (Toplam 100 puan):
        - HTF Count: 40 puan (en önemli)
        - Total Power: 40 puan
        - Whale Detection: 20 puan
        """
        score = 0.0
        
        # =====================================
        # 1. HTF COUNT (40 puan) ⭐⭐⭐
        # =====================================
        htf_count = int(metrics.get('htf_count', 0))
        
        if htf_count == 4:
            score += 40  # Perfect 4/4
        elif htf_count == 3:
            score += 30  # Good 3/4
        elif htf_count == 2:
            score += 15  # Weak
        elif htf_count == 1:
            score += 5   # Very weak
        # htf_count == 0 → 0 puan
        
        # =====================================
        # 2. TOTAL POWER (40 puan) ⭐⭐⭐
        # =====================================
        total_power = float(metrics.get('total_power', 0.0))
        
        if total_power >= 40.0:
            score += 40  # Çok güçlü
        elif total_power >= 25.0:
            score += 30 + (total_power - 25.0) / 15.0 * 10
        elif total_power >= 15.0:
            score += 20 + (total_power - 15.0) / 10.0 * 10
        elif total_power >= 8.0:
            score += 10 + (total_power - 8.0) / 7.0 * 10
        elif total_power >= 5.0:
            score += 5 + (total_power - 5.0) / 3.0 * 5
        else:
            score += min(total_power / 5.0 * 5, 5)
        
        # =====================================
        # 3. WHALE DETECTION (20 puan) ⭐
        # =====================================
        whale_active = metrics.get('whale_active', False)
        
        if whale_active:
            score += 20  # Whale var - büyük bonus
            # Whale + güçlü power kombinasyonu için ekstra bonus
            if total_power >= 20.0:
                score += 5  # Ekstra bonus
        
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
            logger.info(f"💾 Ultra Panel AI modeli kaydedildi (Multi-HTF): {AI_MODEL_FILE}")
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
            logger.info(f"✅ Ultra Panel AI modeli yüklendi (Multi-HTF): R²={self.model_stats['accuracy']:.2%}")
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
            'model_type': 'Ultra Panel v5 Multi-HTF RandomForest Regressor'
        }
    
    def get_feature_importance(self) -> Dict:
        """Feature importance döndür (debug için)"""
        if not self.is_trained or self.model is None:
            return {}
        
        try:
            importances = self.model.feature_importances_
            importance_dict = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, importances)
            }
            return importance_dict
        except:
            return {}


# Global AI model instance
ai_model = CryptoMLModel()