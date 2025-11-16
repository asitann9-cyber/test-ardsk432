"""
🤖 AI Model Sistemi - VPMV Tabanlı (SADECE 4 BİLEŞEN)
🔥 SADECE: VPMV (Volume-Price-Momentum-Volatility) bazlı skorlama
🔥 TIME Alignment KALDIRILDI
🔥 Tetikleyici sistemi entegrasyonu
🔥 ESKİ SİSTEM TAMAMEN KALDIRILDI: Gauss, Deviso, Z-Score, Log Volume
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
    """🔥 SADECE 4 BİLEŞEN: VPMV bazlı AI model sistemi"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'volume_component',      # -50 ile +50
            'price_component',       # -50 ile +50
            'momentum_component',    # -50 ile +50
            'volatility_component',  # -50 ile +50
            'vpmv_score'            # -50 ile +50
        ]
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # Dengeli eğitim verisi oluştur
        self._create_balanced_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 VPMV AI modeli oluşturuluyor (SADECE 4 BİLEŞEN)...")
            self._train_initial_model()
    
    def _create_balanced_training_data(self):
        """
        🔥 SADECE 4 BİLEŞEN: VPMV bazlı dengeli eğitim verisi
        
        Skor Dağılımı:
        - Mükemmel (85-95): 15 örnek
        - İyi (70-85): 25 örnek
        - Orta (50-70): 35 örnek
        - Zayıf (20-50): 50 örnek
        """
        demo_data = []
        
        # ======================================
        # 1. MÜKEMMEL SİNYALLER (85-95 puan)
        # ======================================
        for i in range(15):
            # Güçlü pozitif bileşenler
            volume_comp = random.uniform(30.0, 50.0)
            price_comp = random.uniform(35.0, 50.0)
            momentum_comp = random.uniform(30.0, 50.0)
            volatility_comp = random.uniform(25.0, 40.0)
            vpmv = random.uniform(35.0, 50.0)
            
            demo_data.append({
                'volume_component': volume_comp,
                'price_component': price_comp,
                'momentum_component': momentum_comp,
                'volatility_component': volatility_comp,
                'vpmv_score': vpmv,
                'target': 0.9  # %90 skor
            })
        
        # ======================================
        # 2. İYİ SİNYALLER (70-85 puan)
        # ======================================
        for i in range(25):
            volume_comp = random.uniform(15.0, 35.0)
            price_comp = random.uniform(20.0, 40.0)
            momentum_comp = random.uniform(15.0, 35.0)
            volatility_comp = random.uniform(10.0, 30.0)
            vpmv = random.uniform(20.0, 40.0)
            
            demo_data.append({
                'volume_component': volume_comp,
                'price_component': price_comp,
                'momentum_component': momentum_comp,
                'volatility_component': volatility_comp,
                'vpmv_score': vpmv,
                'target': 0.75  # %75 skor
            })
        
        # ======================================
        # 3. ORTA SİNYALLER (50-70 puan)
        # ======================================
        for i in range(35):
            volume_comp = random.uniform(-5.0, 20.0)
            price_comp = random.uniform(5.0, 25.0)
            momentum_comp = random.uniform(-5.0, 20.0)
            volatility_comp = random.uniform(5.0, 20.0)
            vpmv = random.uniform(5.0, 25.0)
            
            demo_data.append({
                'volume_component': volume_comp,
                'price_component': price_comp,
                'momentum_component': momentum_comp,
                'volatility_component': volatility_comp,
                'vpmv_score': vpmv,
                'target': 0.6  # %60 skor
            })
        
        # ======================================
        # 4. ZAYIF SİNYALLER (20-50 puan)
        # ======================================
        for i in range(50):
            volume_comp = random.uniform(-30.0, 10.0)
            price_comp = random.uniform(-10.0, 15.0)
            momentum_comp = random.uniform(-30.0, 10.0)
            volatility_comp = random.uniform(-10.0, 15.0)
            vpmv = random.uniform(-20.0, 15.0)
            
            demo_data.append({
                'volume_component': volume_comp,
                'price_component': price_comp,
                'momentum_component': momentum_comp,
                'volatility_component': volatility_comp,
                'vpmv_score': vpmv,
                'target': 0.35  # %35 skor
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} VPMV eğitim verisi oluşturuldu (SADECE 4 BİLEŞEN)")
    
    def _train_initial_model(self):
        """🔥 RandomForestRegressor ile VPMV model eğitimi (SADECE 4 BİLEŞEN)"""
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
            
            logger.info(f"🎯 VPMV AI modeli eğitildi (4 BİLEŞEN): R²={r2_score:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """
        🔥 SADECE 4 BİLEŞEN: VPMV metriklerinden ML features oluştur
        
        Args:
            metrics (Dict): compute_vpmv_metrics() çıktısı
            
        Returns:
            np.array: Normalize edilmiş feature array (5 feature: 4 bileşen + vpmv_score)
        """
        features = {}
        
        # 1. Volume Component (-50, +50) → (0, 1)
        vol_raw = metrics.get('volume_component', 0.0)
        features['volume_component'] = (vol_raw + 50.0) / 100.0
        
        # 2. Price Component (-50, +50) → (0, 1)
        price_raw = metrics.get('price_component', 0.0)
        features['price_component'] = (price_raw + 50.0) / 100.0
        
        # 3. Momentum Component (-50, +50) → (0, 1)
        mom_raw = metrics.get('momentum_component', 0.0)
        features['momentum_component'] = (mom_raw + 50.0) / 100.0
        
        # 4. Volatility Component (-50, +50) → (0, 1)
        vol_comp_raw = metrics.get('volatility_component', 0.0)
        features['volatility_component'] = (vol_comp_raw + 50.0) / 100.0
        
        # 5. VPMV Score (-50, +50) → (0, 1)
        vpmv_raw = metrics.get('vpmv_score', 0.0)
        features['vpmv_score'] = (vpmv_raw + 50.0) / 100.0
        
        # Array'e çevir
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.5)  # NaN → 0.5 (orta değer)
        
        return feature_array.reshape(1, -1)
    
    def predict_score(self, metrics: Dict) -> float:
        """
        🔥 SADECE 4 BİLEŞEN: VPMV bazlı AI skorlama
        
        Skor Hesaplama:
        - ML Model: %60
        - Manuel Skor: %40
        
        Args:
            metrics (Dict): VPMV metrikleri (4 bileşen)
            
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
        🔥 SADECE 4 BİLEŞEN: Manuel VPMV skorlama sistemi
        
        Puan Dağılımı (Toplam 100 puan):
        - Volume: 10 puan
        - Price: 70 puan (en yüksek - ağırlık arttırıldı)
        - Momentum: 10 puan
        - Volatility: 10 puan
        """
        score = 0.0
        
        # =====================================
        # 1. VOLUME BİLEŞENİ (10 puan)
        # =====================================
        vol_comp = abs(metrics.get('volume_component', 0.0))
        if vol_comp >= 40.0:
            score += 10
        elif vol_comp >= 25.0:
            score += 7 + (vol_comp - 25.0) / 15.0 * 3
        elif vol_comp >= 15.0:
            score += 4 + (vol_comp - 15.0) / 10.0 * 3
        else:
            score += min(vol_comp / 15.0 * 4, 4)
        
        # =====================================
        # 2. PRICE BİLEŞENİ (70 puan) ⭐⭐⭐
        # =====================================
        price_comp = abs(metrics.get('price_component', 0.0))
        if price_comp >= 40.0:
            score += 70
        elif price_comp >= 25.0:
            score += 50 + (price_comp - 25.0) / 15.0 * 20
        elif price_comp >= 15.0:
            score += 30 + (price_comp - 15.0) / 10.0 * 20
        else:
            score += min(price_comp / 15.0 * 30, 30)
        
        # =====================================
        # 3. MOMENTUM BİLEŞENİ (10 puan)
        # =====================================
        mom_comp = abs(metrics.get('momentum_component', 0.0))
        if mom_comp >= 40.0:
            score += 10
        elif mom_comp >= 25.0:
            score += 7 + (mom_comp - 25.0) / 15.0 * 3
        elif mom_comp >= 15.0:
            score += 4 + (mom_comp - 15.0) / 10.0 * 3
        else:
            score += min(mom_comp / 15.0 * 4, 4)
        
        # =====================================
        # 4. VOLATILITY BİLEŞENİ (10 puan)
        # =====================================
        vol_comp_val = abs(metrics.get('volatility_component', 0.0))
        if vol_comp_val >= 30.0:
            score += 10
        elif vol_comp_val >= 20.0:
            score += 7 + (vol_comp_val - 20.0) / 10.0 * 3
        elif vol_comp_val >= 10.0:
            score += 4 + (vol_comp_val - 10.0) / 10.0 * 3
        else:
            score += min(vol_comp_val / 10.0 * 4, 4)
        
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
            logger.info(f"💾 VPMV AI modeli kaydedildi (4 BİLEŞEN): {AI_MODEL_FILE}")
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
            logger.info(f"✅ VPMV AI modeli yüklendi (4 BİLEŞEN): R²={self.model_stats['accuracy']:.2%}")
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
            'model_type': 'VPMV-Based RandomForest Regressor (4 Components Only)'
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