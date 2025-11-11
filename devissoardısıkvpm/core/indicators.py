"""
📈 Teknik Göstergeler - VPMV Sistemi
🔥 YENİ: SuperTrend bazlı VPMV (Volume-Price-Momentum-Volatility) sistemi
🔥 MTF (Multi-Timeframe) desteği: 1H, 2H, 4H - TAM ENTEGRE
🔥 TIME alignment sinyalleri: 1H, 2H, 4H, 1D, 1W
🔥 Tetikleyici sistemi: Price, Momentum, Volume, Volatility
🔥 Pine Script ile %100 uyumlu
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger("crypto-analytics")


# ============================================
# TANH FONKSİYONU
# ============================================

def tanh_approximation(x: float) -> float:
    """
    TANH yaklaşık hesaplama (Pine Script uyumlu)
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    
    Args:
        x (float): Giriş değeri
        
    Returns:
        float: TANH değeri (-1, +1 arası)
    """
    try:
        e_pos = math.exp(x)
        e_neg = math.exp(-x)
        return (e_pos - e_neg) / (e_pos + e_neg)
    except:
        return 0.0


# ============================================
# ATR VE SUPERTREND HESAPLAMA
# ============================================

def calculate_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    ATR (Average True Range) hesaplama
    Pine Script uyumlu - Manuel EMA yaklaşımı
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        period (int): ATR periyodu
        
    Returns:
        pd.Series: ATR değerleri
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range hesaplama
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Manuel ATR (EMA benzeri)
        atr = pd.Series(index=df.index, dtype=float)
        atr_sma = tr.rolling(window=period).mean()
        
        for i in range(len(df)):
            if i < period:
                atr.iloc[i] = atr_sma.iloc[i]
            else:
                if pd.notna(atr.iloc[i-1]):
                    atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
                else:
                    atr.iloc[i] = atr_sma.iloc[i]
        
        return atr
        
    except Exception as e:
        logger.error(f"ATR hesaplama hatası: {e}")
        return pd.Series([0.0] * len(df), index=df.index)


def calculate_supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    SuperTrend hesaplama
    Pine Script uyumlu mantık
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        atr_period (int): ATR periyodu
        multiplier (float): ATR çarpanı
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (trend, buy_signal, sell_signal)
            trend: 1 (long), -1 (short)
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # ATR hesapla
        atr = calculate_atr(df, atr_period)
        
        # SuperTrend bantları
        hl_avg = (high + low) / 2
        
        up = hl_avg - multiplier * atr
        dn = hl_avg + multiplier * atr
        
        trend = pd.Series([1] * len(df), index=df.index)
        
        # SuperTrend mantığı
        for i in range(1, len(df)):
            # UP band güncelleme
            if close.iloc[i-1] > up.iloc[i-1]:
                up.iloc[i] = max(up.iloc[i], up.iloc[i-1])
            
            # DOWN band güncelleme
            if close.iloc[i-1] < dn.iloc[i-1]:
                dn.iloc[i] = min(dn.iloc[i], dn.iloc[i-1])
            
            # Trend belirleme
            if trend.iloc[i-1] == -1 and close.iloc[i] > dn.iloc[i-1]:
                trend.iloc[i] = 1
            elif trend.iloc[i-1] == 1 and close.iloc[i] < up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        # Buy/Sell sinyalleri
        buy_signal = (trend == 1) & (trend.shift(1) == -1)
        sell_signal = (trend == -1) & (trend.shift(1) == 1)
        
        return trend, buy_signal, sell_signal
        
    except Exception as e:
        logger.error(f"SuperTrend hesaplama hatası: {e}")
        return pd.Series([1] * len(df)), pd.Series([False] * len(df)), pd.Series([False] * len(df))


# ============================================
# VPMV BİLEŞENLERİ HESAPLAMA
# ============================================

def calculate_vpmv_components(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> Dict:
    """
    🔥 YENİ: VPMV (Volume-Price-Momentum-Volatility) bileşenlerini hesapla
    SuperTrend bazlı reset mekanizması ile
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        atr_period (int): ATR periyodu
        multiplier (float): SuperTrend çarpanı
        
    Returns:
        Dict: VPMV bileşenleri ve skor
    """
    try:
        if df is None or df.empty or len(df) < 30:
            return {
                'volume_component': 0.0,
                'price_component': 0.0,
                'momentum_component': 0.0,
                'volatility_component': 0.0,
                'vpmv_score': 0.0,
                'signal_type': 'none'
            }
        
        close = df['close']
        volume = df['volume']
        open_price = df['open']
        
        # SuperTrend hesapla
        trend, buy_signal, sell_signal = calculate_supertrend(df, atr_period, multiplier)
        
        # ATR hesapla (volatility için)
        atr = calculate_atr(df, atr_period)
        
        # Kümülatif değişkenler
        cumulative_volume = 0.0
        cumulative_momentum = 0.0
        signal_price = None
        avg_volume = volume.rolling(window=20).mean()
        
        vol_component = 0.0
        mom_component = 0.0
        price_component = 0.0
        volatility_component = 0.0
        
        # Her bar'ı işle
        for i in range(1, len(df)):
            # RESET: Buy veya Sell sinyali
            if buy_signal.iloc[i] or sell_signal.iloc[i]:
                cumulative_volume = 0.0
                cumulative_momentum = 0.0
                signal_price = close.iloc[i]
                continue
            
            # Signal price yoksa başlat
            if signal_price is None:
                signal_price = close.iloc[i]
                continue
            
            # 1. VOLUME BİLEŞENİ (Net Hacim - Alış/Satış)
            buy_vol = volume.iloc[i] if close.iloc[i] > open_price.iloc[i] else 0.0
            sell_vol = volume.iloc[i] if close.iloc[i] < open_price.iloc[i] else 0.0
            net_volume = buy_vol - sell_vol
            cumulative_volume += net_volume
            
            vol_ratio = cumulative_volume / (avg_volume.iloc[i] * 5 + 0.0001)
            vol_component = tanh_approximation(vol_ratio) * 50
            
            # 2. MOMENTUM BİLEŞENİ (Kümülatif % değişim)
            momentum_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * 100
            cumulative_momentum += momentum_change
            mom_component = tanh_approximation(cumulative_momentum / 10) * 50
            
            # 3. PRICE BİLEŞENİ (Signal fiyatından % değişim)
            price_change = (close.iloc[i] - signal_price) / signal_price * 100
            price_component = tanh_approximation(price_change / 10) * 50
            
            # 4. VOLATILITY BİLEŞENİ (ATR/Price * 100)
            volatility_pct = (atr.iloc[i] / close.iloc[i]) * 100
            volatility_component = tanh_approximation(volatility_pct / 5) * 50
        
        # VPMV SKORU (Ağırlıklı ortalama)
        vpmv_score = (
            price_component * 0.7 +      # %70 - En yüksek ağırlık
            vol_component * 0.1 +         # %10
            mom_component * 0.1 +         # %10
            volatility_component * 0.1    # %10
        )
        
        # Sinyal tipi
        signal_type = 'long' if vpmv_score >= 0 else 'short'
        
        return {
            'volume_component': float(vol_component),
            'price_component': float(price_component),
            'momentum_component': float(mom_component),
            'volatility_component': float(volatility_component),
            'vpmv_score': float(vpmv_score),
            'signal_type': signal_type
        }
        
    except Exception as e:
        logger.error(f"VPMV hesaplama hatası: {e}")
        return {
            'volume_component': 0.0,
            'price_component': 0.0,
            'momentum_component': 0.0,
            'volatility_component': 0.0,
            'vpmv_score': 0.0,
            'signal_type': 'none'
        }


# ============================================
# 🔥 YENİ: MULTI-TIMEFRAME VPMV HESAPLAMA
# ============================================

def compute_mtf_vpmv_components(symbol: str) -> Dict:
    """
    🔥 YENİ: Multi-Timeframe VPMV bileşenlerini hesapla
    Pine Script'teki gibi 1H, 2H, 4H VPMV değerleri
    
    Args:
        symbol (str): Trading sembolü
        
    Returns:
        Dict: MTF VPMV verileri {
            '1H': {'volume': x, 'price': y, 'momentum': z, 'volatility': w, 'vpmv_score': s, 'trigger': t},
            '2H': {...},
            '4H': {...}
        }
    """
    try:
        from data.fetch_data import fetch_klines
        
        timeframes = {
            '1H': '1h',
            '2H': '2h',
            '4H': '4h'
        }
        
        mtf_data = {}
        
        for tf_name, tf_value in timeframes.items():
            try:
                df_tf = fetch_klines(symbol, tf_value)
                if df_tf is not None and not df_tf.empty and len(df_tf) >= 30:
                    # VPMV bileşenlerini hesapla
                    components = calculate_vpmv_components(df_tf)
                    
                    # VPMV skorunu hesapla (zaten calculate_vpmv_components içinde var ama açık yapalım)
                    vpmv_score = (
                        components['price_component'] * 0.7 +
                        components['volume_component'] * 0.1 +
                        components['momentum_component'] * 0.1 +
                        components['volatility_component'] * 0.1
                    )
                    
                    # Tetikleyiciyi hesapla
                    trigger = calculate_trigger(components)
                    
                    mtf_data[tf_name] = {
                        'volume': components['volume_component'],
                        'price': components['price_component'],
                        'momentum': components['momentum_component'],
                        'volatility': components['volatility_component'],
                        'vpmv_score': vpmv_score,
                        'trigger': trigger
                    }
                else:
                    # Veri yetersiz - sıfır değerler
                    mtf_data[tf_name] = {
                        'volume': 0.0,
                        'price': 0.0,
                        'momentum': 0.0,
                        'volatility': 0.0,
                        'vpmv_score': 0.0,
                        'trigger': 'Yok'
                    }
            except Exception as e:
                logger.debug(f"{symbol} {tf_name} MTF VPMV hatası: {e}")
                mtf_data[tf_name] = {
                    'volume': 0.0,
                    'price': 0.0,
                    'momentum': 0.0,
                    'volatility': 0.0,
                    'vpmv_score': 0.0,
                    'trigger': 'Yok'
                }
        
        return mtf_data
        
    except Exception as e:
        logger.error(f"MTF VPMV hesaplama hatası: {e}")
        return {}


# ============================================
# TETİKLEYİCİ SİSTEMİ
# ============================================

def calculate_trigger(vpmv_components: Dict) -> str:
    """
    Tetikleyici sistemi - Hangi bileşen dominant?
    
    Koşullar:
    - Price: >= 20
    - Momentum: >= 10
    - Volume: >= 15
    - Volatility: >= 8
    
    Args:
        vpmv_components (Dict): VPMV bileşenleri
        
    Returns:
        str: Tetikleyici tipi ("Price", "Momentum", "Volume", "Volatility", "Yok")
    """
    try:
        price_val = abs(vpmv_components.get('price_component', 0.0))
        mom_val = abs(vpmv_components.get('momentum_component', 0.0))
        vol_val = abs(vpmv_components.get('volume_component', 0.0))
        volatility_val = abs(vpmv_components.get('volatility_component', 0.0))
        
        # Tetikleyici kontrolü
        price_trig = price_val >= 20
        momentum_trig = mom_val >= 10
        volume_trig = vol_val >= 15
        volatility_trig = volatility_val >= 8
        
        # Price tetiklenmediyse "Yok"
        if not price_trig:
            return "Yok"
        
        # En yüksek değere sahip bileşeni bul
        max_val = 0.0
        trigger_name = "Price"
        
        if momentum_trig and mom_val > max_val:
            max_val = mom_val
            trigger_name = "Momentum"
        
        if volume_trig and vol_val > max_val:
            max_val = vol_val
            trigger_name = "Volume"
        
        if volatility_trig and volatility_val > max_val:
            trigger_name = "Volatility"
        
        return trigger_name
        
    except Exception as e:
        logger.debug(f"Tetikleyici hesaplama hatası: {e}")
        return "Yok"


# ============================================
# TIME ALIGNMENT (MTF SİNYALLERİ)
# ============================================

def calculate_time_alignment(symbol: str, current_vpmv_direction: int) -> Dict:
    """
    🔥 YENİ: TIME alignment hesaplama
    1H, 2H, 4H, 1D, 1W timeframe'lerden close/open karşılaştırması
    
    Args:
        symbol (str): Trading sembolü
        current_vpmv_direction (int): Mevcut VPMV yönü (1: long, -1: short)
        
    Returns:
        Dict: TIME alignment bilgileri
    """
    try:
        from data.fetch_data import fetch_klines
        
        timeframes = {
            '1H': '1h',
            '2H': '2h', 
            '4H': '4h',
            '1D': '1d',
            '1W': '1w'
        }
        
        directions = {}
        match_count = 0
        
        for tf_name, tf_value in timeframes.items():
            try:
                df_tf = fetch_klines(symbol, tf_value)
                if df_tf is not None and not df_tf.empty:
                    last_close = df_tf['close'].iloc[-1]
                    last_open = df_tf['open'].iloc[-1]
                    
                    # Yön belirleme
                    if last_close > last_open:
                        directions[tf_name] = 1  # Long
                    elif last_close < last_open:
                        directions[tf_name] = -1  # Short
                    else:
                        directions[tf_name] = 0  # Nötr
                    
                    # Uyum kontrolü
                    if directions[tf_name] == current_vpmv_direction and directions[tf_name] != 0:
                        match_count += 1
                else:
                    directions[tf_name] = 0
            except:
                directions[tf_name] = 0
        
        return {
            'directions': directions,
            'match_count': match_count,
            'total_timeframes': len(timeframes)
        }
        
    except Exception as e:
        logger.debug(f"TIME alignment hesaplama hatası: {e}")
        return {
            'directions': {},
            'match_count': 0,
            'total_timeframes': 5
        }


# ============================================
# COMPUTE METRICS (ANA FONKSİYON) - 🔥 MTF ENTEGRE
# ============================================

def compute_vpmv_metrics(df: pd.DataFrame, symbol: str = None) -> Dict:
    """
    🔥 GÜNCELLEME: VPMV metriklerini hesapla + MTF VPMV desteği
    Pine Script ile %100 uyumlu
    
    Args:
        df (pd.DataFrame): OHLCV verileri (current timeframe)
        symbol (str): Sembol adı (MTF ve TIME alignment için GEREKLİ)
        
    Returns:
        Dict: Tüm VPMV metrikleri + MTF verileri
    """
    if df is None or df.empty:
        return {}
    
    try:
        # Current timeframe VPMV bileşenleri
        vpmv_data = calculate_vpmv_components(df)
        
        # Current timeframe tetikleyici
        trigger = calculate_trigger(vpmv_data)
        
        # TIME alignment hesapla (eğer sembol verilmişse)
        time_alignment = {'match_count': 0, 'directions': {}, 'total_timeframes': 5}
        if symbol:
            vpmv_direction = 1 if vpmv_data['vpmv_score'] >= 0 else -1
            time_alignment = calculate_time_alignment(symbol, vpmv_direction)
        
        # 🔥 YENİ: Multi-Timeframe VPMV hesapla (1H, 2H, 4H)
        mtf_vpmv = {}
        if symbol:
            mtf_vpmv = compute_mtf_vpmv_components(symbol)
            logger.debug(f"{symbol} MTF VPMV hesaplandı: {list(mtf_vpmv.keys())}")
        
        # Tüm metrikleri birleştir
        return {
            # Current Timeframe VPMV Bileşenleri
            'volume_component': vpmv_data['volume_component'],
            'price_component': vpmv_data['price_component'],
            'momentum_component': vpmv_data['momentum_component'],
            'volatility_component': vpmv_data['volatility_component'],
            'vpmv_score': vpmv_data['vpmv_score'],
            
            # Sinyal tipi
            'run_type': vpmv_data['signal_type'],  # 'long' veya 'short'
            
            # Current Timeframe Tetikleyici
            'trigger_type': trigger,
            
            # TIME Alignment (5 timeframe)
            'time_match_count': time_alignment['match_count'],
            'time_directions': time_alignment['directions'],
            'time_total': time_alignment['total_timeframes'],
            
            # 🔥 YENİ: Multi-Timeframe VPMV (1H, 2H, 4H)
            'mtf_vpmv': mtf_vpmv
        }
        
    except Exception as e:
        logger.error(f"VPMV metrics hesaplama hatası: {e}")
        return {}


# ============================================
# ESKİ FONKSİYON UYUMLULUĞU
# ============================================

def compute_consecutive_metrics(df: pd.DataFrame) -> Dict:
    """
    🔄 ESKİ FONKSİYON UYUMLULUĞU: compute_vpmv_metrics'e yönlendir
    DEPRECATED: Yeni kodda compute_vpmv_metrics() kullan
    """
    logger.warning("⚠️ compute_consecutive_metrics() deprecated - compute_vpmv_metrics() kullan")
    return compute_vpmv_metrics(df)


# ============================================
# 🔥 YENİ: DEPRECATED FONKSIYONLAR (ESKİ SİSTEM)
# ============================================

def calculate_zigzag_high(*args, **kwargs):
    """DEPRECATED - Eski sistem"""
    logger.warning("⚠️ calculate_zigzag_high() deprecated - artık kullanılmıyor")
    return 0.0

def calculate_zigzag_low(*args, **kwargs):
    """DEPRECATED - Eski sistem"""
    logger.warning("⚠️ calculate_zigzag_low() deprecated - artık kullanılmıyor")
    return 0.0

def calculate_deviso_ratio(*args, **kwargs):
    """DEPRECATED - Eski sistem"""
    logger.warning("⚠️ calculate_deviso_ratio() deprecated - artık kullanılmıyor")
    return 0.0