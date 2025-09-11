"""
🤖 AI Model Sistemi - CVD + ROC Momentum
Tamamen yeni CVD momentum tabanlı machine learning modeli
🔥 YENİ: CVD, momentum strength, buy/sell pressure tabanlı skorlama
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
    """CVD + ROC Momentum tabanlı AI model sistemi"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # YENİ CVD FEATURE'LAR
        self.feature_names = [
            'cvd_roc_momentum',      # CVD ROC momentum (-50 ile +50)
            'momentum_strength',     # CVD momentum gücü (0-100)
            'buy_pressure',          # Alıcı baskısı (0-100)
            'sell_pressure',         # Satıcı baskısı (0-100)
            'deviso_cvd_harmony',    # Deviso-CVD uyum skoru (0-100)
            'trend_strength',        # Kombine trend gücü (0-100)
            'deviso_ratio',          # Deviso ratio (-25 ile +25)
            'cvd_alignment'          # CVD-Deviso uyumu (0 veya 1)
        ]
        
        self.training_data = []
        self.model_stats = {'accuracy': 0.0, 'last_training': None}
        
        # CVD tabanlı eğitim verisi oluştur
        self._create_cvd_training_data()
        
        try:
            self.load_model()
        except:
            logger.info("🤖 CVD AI modeli oluşturuluyor...")
            self._train_initial_model()
    
    def _create_cvd_training_data(self):
        """CVD momentum tabanlı dengeli eğitim verisi"""
        demo_data = []
        
        # Mükemmel CVD sinyaller (%85-95) - ÇOK NADIR
        for i in range(15):
            cvd_roc_momentum = random.uniform(20.0, 45.0) * random.choice([1, -1])  # Güçlü momentum
            momentum_strength = random.uniform(80.0, 100.0)
            buy_pressure = random.uniform(75.0, 95.0) if cvd_roc_momentum > 0 else random.uniform(5.0, 25.0)
            sell_pressure = 100.0 - buy_pressure
            deviso_cvd_harmony = random.uniform(85.0, 100.0)  # Mükemmel uyum
            trend_strength = random.uniform(80.0, 100.0)
            deviso_ratio = random.uniform(8.0, 20.0) * (1 if cvd_roc_momentum > 0 else -1)
            cvd_alignment = 1  # Mükemmel uyum
            
            demo_data.append({
                'cvd_roc_momentum': cvd_roc_momentum,
                'momentum_strength': momentum_strength,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'deviso_cvd_harmony': deviso_cvd_harmony,
                'trend_strength': trend_strength,
                'deviso_ratio': deviso_ratio,
                'cvd_alignment': cvd_alignment,
                'target': 0.9  # %90 başarı beklentisi
            })
        
        # İyi CVD sinyaller (%70-85) - NADIR
        for i in range(25):
            cvd_roc_momentum = random.uniform(10.0, 25.0) * random.choice([1, -1])
            momentum_strength = random.uniform(60.0, 85.0)
            buy_pressure = random.uniform(65.0, 80.0) if cvd_roc_momentum > 0 else random.uniform(20.0, 35.0)
            sell_pressure = 100.0 - buy_pressure
            deviso_cvd_harmony = random.uniform(70.0, 85.0)
            trend_strength = random.uniform(60.0, 80.0)
            deviso_ratio = random.uniform(3.0, 8.0) * (1 if cvd_roc_momentum > 0 else -1)
            cvd_alignment = 1
            
            demo_data.append({
                'cvd_roc_momentum': cvd_roc_momentum,
                'momentum_strength': momentum_strength,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'deviso_cvd_harmony': deviso_cvd_harmony,
                'trend_strength': trend_strength,
                'deviso_ratio': deviso_ratio,
                'cvd_alignment': cvd_alignment,
                'target': 0.75  # %75 başarı beklentisi
            })
        
        # Orta CVD sinyaller (%50-70) - NORMAL
        for i in range(35):
            cvd_roc_momentum = random.uniform(5.0, 15.0) * random.choice([1, -1])
            momentum_strength = random.uniform(40.0, 65.0)
            buy_pressure = random.uniform(55.0, 70.0) if cvd_roc_momentum > 0 else random.uniform(30.0, 45.0)
            sell_pressure = 100.0 - buy_pressure
            deviso_cvd_harmony = random.uniform(50.0, 70.0)
            trend_strength = random.uniform(40.0, 65.0)
            deviso_ratio = random.uniform(1.0, 4.0) * (1 if cvd_roc_momentum > 0 else -1)
            cvd_alignment = random.choice([0, 1])
            
            demo_data.append({
                'cvd_roc_momentum': cvd_roc_momentum,
                'momentum_strength': momentum_strength,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'deviso_cvd_harmony': deviso_cvd_harmony,
                'trend_strength': trend_strength,
                'deviso_ratio': deviso_ratio,
                'cvd_alignment': cvd_alignment,
                'target': 0.6  # %60 başarı beklentisi
            })
        
        # Zayıf CVD sinyaller (%20-50) - YAYGIN
        for i in range(50):
            cvd_roc_momentum = random.uniform(-5.0, 8.0)  # Zayıf momentum
            momentum_strength = random.uniform(10.0, 45.0)
            buy_pressure = random.uniform(45.0, 65.0)  # Karışık baskı
            sell_pressure = 100.0 - buy_pressure
            deviso_cvd_harmony = random.uniform(20.0, 50.0)  # Zayıf uyum
            trend_strength = random.uniform(15.0, 45.0)
            deviso_ratio = random.uniform(-3.0, 3.0)  # Karışık deviso
            cvd_alignment = random.choice([0, 0, 1])  # Çoğunlukla uyumsuz
            
            demo_data.append({
                'cvd_roc_momentum': cvd_roc_momentum,
                'momentum_strength': momentum_strength,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'deviso_cvd_harmony': deviso_cvd_harmony,
                'trend_strength': trend_strength,
                'deviso_ratio': deviso_ratio,
                'cvd_alignment': cvd_alignment,
                'target': 0.35  # %35 başarı beklentisi
            })
        
        self.training_data = demo_data
        logger.info(f"📊 {len(demo_data)} CVD tabanlı eğitim verisi oluşturuldu")
    
    def _train_initial_model(self):
        """CVD tabanlı regresyon eğitimi"""
        try:
            df = pd.DataFrame(self.training_data)
            X = df[self.feature_names].values
            y = df['target'].values  # 0-1 aralığında sürekli değerler
            
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # RandomForest Regressor
            self.model = RandomForestRegressor(
                n_estimators=200,  # CVD için daha fazla ağaç
                max_depth=12,       
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # R² score hesapla
            r2_score = self.model.score(X_scaled, y)
            self.model_stats['accuracy'] = r2_score
            self.model_stats['last_training'] = datetime.now(LOCAL_TZ).isoformat()
            
            logger.info(f"🎯 CVD AI modeli eğitildi: R²={r2_score:.2%}")
            self.save_model()
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
    
    def create_features(self, metrics: Dict) -> np.array:
        """CVD metriklerini ML features'a çevir - normalize edilmiş"""
        features = {}
        
        # CVD ROC Momentum (-50 ile +50 aralığı → 0-1)
        cvd_roc = metrics.get('cvd_roc_momentum', 0.0)
        features['cvd_roc_momentum'] = (max(-50, min(50, cvd_roc)) + 50) / 100.0
        
        # Momentum Strength (0-100 → 0-1)
        features['momentum_strength'] = min(metrics.get('momentum_strength', 0.0), 100.0) / 100.0
        
        # Buy Pressure (0-100 → 0-1)
        features['buy_pressure'] = min(metrics.get('buy_pressure', 50.0), 100.0) / 100.0
        
        # Sell Pressure (0-100 → 0-1)
        features['sell_pressure'] = min(metrics.get('sell_pressure', 50.0), 100.0) / 100.0
        
        # Deviso CVD Harmony (0-100 → 0-1)
        features['deviso_cvd_harmony'] = min(metrics.get('deviso_cvd_harmony', 50.0), 100.0) / 100.0
        
        # Trend Strength (0-100 → 0-1)
        features['trend_strength'] = min(metrics.get('trend_strength', 0.0), 100.0) / 100.0
        
        # Deviso Ratio (-25 ile +25 aralığı → 0-1)
        deviso_raw = metrics.get('deviso_ratio', 0.0)
        features['deviso_ratio'] = (max(-25, min(25, deviso_raw)) + 25) / 50.0
        
        # CVD Alignment (CVD direction ve Deviso direction uyumu)
        cvd_direction = metrics.get('cvd_direction', 'neutral')
        signal_type = metrics.get('signal_type', 'neutral')
        deviso_ratio = metrics.get('deviso_ratio', 0.0)
        
        # CVD-Deviso uyumu hesapla
        if ((cvd_direction == 'bullish' and deviso_ratio > 1.0) or 
            (cvd_direction == 'bearish' and deviso_ratio < -1.0) or
            (signal_type in ['long', 'strong_long'] and deviso_ratio > 0.5) or
            (signal_type in ['short', 'strong_short'] and deviso_ratio < -0.5)):
            features['cvd_alignment'] = 1.0
        else:
            features['cvd_alignment'] = 0.0
        
        feature_array = np.array([features[name] for name in self.feature_names], dtype=np.float32)
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        
        return feature_array.reshape(1, -1)
    
    def predict_score(self, metrics: Dict) -> float:
        """CVD tabanlı skorlama sistemi"""
        if not self.is_trained:
            return self._calculate_cvd_score(metrics)
        
        try:
            # ML model prediction
            features = self.create_features(metrics)
            features_scaled = self.scaler.transform(features)
            
            ml_prediction = self.model.predict(features_scaled)[0]
            ml_score = np.clip(ml_prediction, 0.0, 1.0) * 100
            
            # Manuel CVD skorlama
            manual_score = self._calculate_cvd_score(metrics)
            
            # İki skoru dengele (ML %70, Manuel %30)
            final_score = (ml_score * 0.7) + (manual_score * 0.3)
            
            return float(np.clip(final_score, 5.0, 95.0))  # 5-95 aralığında sınırla
            
        except Exception as e:
            logger.debug(f"AI skor hesaplama hatası: {e}")
            return self._calculate_cvd_score(metrics)
    
    def _calculate_cvd_score(self, metrics: Dict) -> float:
        """CVD tabanlı manuel skorlama sistemi (Toplam 100 puan)"""
        score = 0.0
        
        # 1. CVD Momentum Strength (35 puan) - ANA KRİTER
        momentum_strength = metrics.get('momentum_strength', 0.0)
        momentum_score = min(momentum_strength / 100.0, 1.0) * 35
        score += momentum_score
        
        # 2. Deviso-CVD Harmony (25 puan)
        harmony = metrics.get('deviso_cvd_harmony', 50.0)
        if harmony >= 80:
            harmony_score = 25
        elif harmony >= 65:
            harmony_score = 20 + (harmony - 65) / 15 * 5  # 20-25 arası
        elif harmony >= 45:
            harmony_score = 10 + (harmony - 45) / 20 * 10  # 10-20 arası
        else:
            harmony_score = harmony / 45 * 10  # 0-10 arası
        score += harmony_score
        
        # 3. Buy/Sell Pressure Dominance (20 puan)
        buy_pressure = metrics.get('buy_pressure', 50.0)
        sell_pressure = metrics.get('sell_pressure', 50.0)
        pressure_dominance = abs(buy_pressure - sell_pressure)
        
        if pressure_dominance >= 30:
            pressure_score = 20  # Güçlü dominance
        elif pressure_dominance >= 20:
            pressure_score = 15 + (pressure_dominance - 20) / 10 * 5  # 15-20 arası
        elif pressure_dominance >= 10:
            pressure_score = 10 + (pressure_dominance - 10) / 10 * 5  # 10-15 arası
        else:
            pressure_score = pressure_dominance / 10 * 10  # 0-10 arası
        score += pressure_score
        
        # 4. Deviso Ratio Gücü (15 puan)
        deviso_ratio = abs(metrics.get('deviso_ratio', 0.0))
        if deviso_ratio >= 5.0:
            deviso_score = 15
        elif deviso_ratio >= 2.0:
            deviso_score = 10 + (deviso_ratio - 2.0) / 3.0 * 5  # 10-15 arası
        elif deviso_ratio >= 0.5:
            deviso_score = 5 + (deviso_ratio - 0.5) / 1.5 * 5   # 5-10 arası
        else:
            deviso_score = deviso_ratio / 0.5 * 5  # 0-5 arası
        score += deviso_score
        
        # 5. CVD-Deviso Alignment Bonus (5 puan)
        cvd_direction = metrics.get('cvd_direction', 'neutral')
        deviso_ratio_raw = metrics.get('deviso_ratio', 0.0)
        
        if ((cvd_direction == 'bullish' and deviso_ratio_raw > 1.0) or 
            (cvd_direction == 'bearish' and deviso_ratio_raw < -1.0)):
            alignment_score = 5  # Mükemmel uyum
        elif ((cvd_direction == 'bullish' and deviso_ratio_raw > 0) or 
              (cvd_direction == 'bearish' and deviso_ratio_raw < 0)):
            alignment_score = 3  # İyi uyum
        else:
            alignment_score = 0  # Uyumsuzluk
        score += alignment_score
        
        # Kalite cezaları
        penalty = 0
        
        # Çok zayıf momentum cezası
        if momentum_strength < 20:
            penalty += 5
            
        # Zayıf harmony cezası
        if harmony < 40:
            penalty += 5
            
        score -= penalty
        
        return max(5.0, min(score, 95.0))  # 5-95 aralığında sınırla
    
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
            logger.info(f"💾 CVD AI modeli kaydedildi: {AI_MODEL_FILE}")
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
            logger.info(f"✅ CVD AI modeli yüklendi: R²={self.model_stats['accuracy']:.2%}")
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
            'model_type': 'CVD Momentum Regression'
        }


# Global AI model instance
ai_model = CryptoMLModel()