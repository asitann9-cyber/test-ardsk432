"""
📈 Teknik Göstergeler - CVD + ROC Momentum Sistemi
🔥 TAMAMEN YENİ: CVD (Cumulative Volume Delta) + ROC momentum sistemi
🎯 Deviso ratio + CVD momentum uyum sistemi
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from config import DEVISO_PARAMS
from core.utils import pine_sma, crossunder, crossover, cross

logger = logging.getLogger("crypto-analytics")


# ============================================================================
# CVD (CUMULATIVE VOLUME DELTA) SİSTEMİ
# ============================================================================

def calculate_cvd_data(df: pd.DataFrame) -> Dict:
    """
    CVD (Cumulative Volume Delta) hesaplama - Logaritmik Simetrik Yaklaşım
    
    Her mumda:
    - Close pozisyonuna göre logaritmik ağırlıklandırma (daha simetrik)
    - Kümülatif topla (tüm mumların birikimi)
    
    Returns:
        Dict: CVD verileri
    """
    if df is None or df.empty:
        return {
            'cvd_values': [],
            'buy_volumes': [],
            'sell_volumes': [],
            'current_cvd': 0.0
        }
    
    cvd_values = []
    buy_volumes = []
    sell_volumes = []
    cumulative_cvd = 0.0
    
    for i in range(len(df)):
        o, h, l, c, v = df.iloc[i][['open', 'high', 'low', 'close', 'volume']]
        
        # Price range kontrolü
        price_range = h - l
        if price_range == 0 or pd.isna(price_range) or v == 0:
            # Doji mum veya sıfır volume - eşit dağılım
            buy_vol = v * 0.5
            sell_vol = v * 0.5
        else:
            # Lineer pozisyon hesapla
            linear_position = (c - l) / price_range  # 0-1 arası
            
            # Logaritmik ağırlıklandırma (simetrik)
            if linear_position > 0.5:
                # Üst yarıda - buy bias ile logaritmik
                log_position = 0.5 + 0.5 * (math.log(2 * linear_position) / math.log(2))
            elif linear_position > 0:
                # Alt yarıda - sell bias ile logaritmik
                log_position = 0.5 * (math.log(2 * linear_position + 1) / math.log(2))
            else:
                # Sıfır pozisyon
                log_position = 0.0
            
            # Volume dağılımı
            buy_vol = v * log_position
            sell_vol = v * (1 - log_position)
        
        buy_volumes.append(buy_vol)
        sell_volumes.append(sell_vol)
        
        # CVD kümülatif hesaplama
        volume_delta = buy_vol - sell_vol
        cumulative_cvd += volume_delta
        cvd_values.append(cumulative_cvd)
    
    return {
        'cvd_values': cvd_values,
        'buy_volumes': buy_volumes,
        'sell_volumes': sell_volumes,
        'current_cvd': cvd_values[-1] if cvd_values else 0.0
    }


def calculate_cvd_roc_momentum(cvd_values: list, window: int = 14) -> Dict:
    """
    CVD ROC (Rate of Change) Momentum hesaplama
    
    Log ROC kullanarak simetrik momentum:
    CVD_ROC = ln(|CVD_now| / |CVD_prev|) * 100 * sign(CVD_now)
    
    Args:
        cvd_values: CVD değerleri listesi
        window: ROC hesaplama penceresi
        
    Returns:
        Dict: ROC momentum metrikleri
    """
    if not cvd_values or len(cvd_values) < window + 1:
        return {
            'roc_momentum': 0.0,
            'momentum_direction': 'neutral',
            'momentum_strength': 0.0,
            'momentum_acceleration': 0.0
        }
    
    try:
        # Son CVD değeri
        current_cvd = cvd_values[-1]
        previous_cvd = cvd_values[-(window + 1)]
        
        # Sıfır kontrolü
        if abs(previous_cvd) < 1e-10:
            roc_momentum = 0.0
        else:
            # Log ROC hesaplama (simetrik)
            abs_ratio = abs(current_cvd) / abs(previous_cvd)
            log_roc = math.log(abs_ratio) * 100
            
            # İşaret korunumu
            if current_cvd >= 0:
                roc_momentum = log_roc
            else:
                roc_momentum = -log_roc
        
        # Momentum direction
        if roc_momentum > 2.0:
            momentum_direction = 'bullish'
        elif roc_momentum < -2.0:
            momentum_direction = 'bearish'
        else:
            momentum_direction = 'neutral'
        
        # Momentum strength (0-100)
        momentum_strength = min(abs(roc_momentum), 50.0) * 2  # Max 100
        
        # Momentum acceleration (son 3 periyot karşılaştırması)
        if len(cvd_values) >= window * 2:
            prev_roc_cvd = cvd_values[-(window * 2)]
            if abs(prev_roc_cvd) > 1e-10:
                prev_roc = math.log(abs(previous_cvd) / abs(prev_roc_cvd)) * 100
                momentum_acceleration = roc_momentum - prev_roc
            else:
                momentum_acceleration = 0.0
        else:
            momentum_acceleration = 0.0
        
        return {
            'roc_momentum': roc_momentum,
            'momentum_direction': momentum_direction,
            'momentum_strength': momentum_strength,
            'momentum_acceleration': momentum_acceleration
        }
        
    except Exception as e:
        logger.debug(f"CVD ROC hesaplama hatası: {e}")
        return {
            'roc_momentum': 0.0,
            'momentum_direction': 'neutral',
            'momentum_strength': 0.0,
            'momentum_acceleration': 0.0
        }


def calculate_volume_pressure(buy_volumes: list, sell_volumes: list, window: int = 20) -> Dict:
    """
    Alıcı/Satıcı baskısı hesaplama
    
    Args:
        buy_volumes: Alıcı hacim listesi
        sell_volumes: Satıcı hacim listesi
        window: Analiz penceresi
        
    Returns:
        Dict: Baskı metrikleri
    """
    if not buy_volumes or not sell_volumes or len(buy_volumes) < window:
        return {
            'buy_pressure': 50.0,
            'sell_pressure': 50.0,
            'pressure_dominance': 'neutral',
            'pressure_intensity': 0.0
        }
    
    # Son N periyot hacim analizi
    recent_buy = sum(buy_volumes[-window:])
    recent_sell = sum(sell_volumes[-window:])
    total_volume = recent_buy + recent_sell
    
    if total_volume > 0:
        buy_pressure = (recent_buy / total_volume) * 100
        sell_pressure = (recent_sell / total_volume) * 100
    else:
        buy_pressure = sell_pressure = 50.0
    
    # Dominance belirleme
    pressure_diff = abs(buy_pressure - sell_pressure)
    if buy_pressure > 60:
        pressure_dominance = 'buy_dominant'
    elif sell_pressure > 60:
        pressure_dominance = 'sell_dominant'
    else:
        pressure_dominance = 'neutral'
    
    # Intensity (0-100)
    pressure_intensity = min(pressure_diff, 50.0) * 2  # Max 100
    
    return {
        'buy_pressure': buy_pressure,
        'sell_pressure': sell_pressure,
        'pressure_dominance': pressure_dominance,
        'pressure_intensity': pressure_intensity
    }


# ============================================================================
# DEVİSO RATIO SİSTEMİ (Korundu)
# ============================================================================

def calculate_zigzag_high(high_series: pd.Series, period: int) -> pd.Series:
    """ZigZag yüksek noktaları hesapla"""
    result = []
    for i in range(len(high_series)):
        if i < period - 1:
            result.append(np.nan)
        else:
            window = high_series.iloc[i-period+1:i+1]
            if high_series.iloc[i] == window.max():
                result.append(high_series.iloc[i])
            else:
                result.append(np.nan)
    return pd.Series(result, index=high_series.index)


def calculate_zigzag_low(low_series: pd.Series, period: int) -> pd.Series:
    """ZigZag düşük noktaları hesapla"""
    result = []
    for i in range(len(low_series)):
        if i < period - 1:
            result.append(np.nan)
        else:
            window = low_series.iloc[i-period+1:i+1]
            if low_series.iloc[i] == window.min():
                result.append(low_series.iloc[i])
            else:
                result.append(np.nan)
    return pd.Series(result, index=low_series.index)


def calculate_deviso_signals(df: pd.DataFrame, 
                           zigzag_high_period: int = 10,
                           zigzag_low_period: int = 10,
                           min_movement_pct: float = 0.160,
                           ma_period: int = 20,
                           std_mult: float = 2.0,
                           ma_length: int = 10):
    """Deviso sinyallerini hesaplar - Pine Script mantığıyla"""
    
    result_df = df.copy()
    
    # ZigZag hesaplama
    zigzag_high = calculate_zigzag_high(result_df['high'], zigzag_high_period)
    zigzag_low = calculate_zigzag_low(result_df['low'], zigzag_low_period)
    
    # Minimum hareket
    min_movement = result_df['close'] * min_movement_pct / 100

    # LongMumu / ShortMumu sinyalleri
    long_mumu = []
    short_mumu = []

    for i in range(len(result_df)):
        long_condition = False
        short_condition = False
        
        # LongMumu koşulu
        if not pd.isna(zigzag_high.iloc[i]):
            prev_indices = []
            for j in range(i-1, -1, -1):
                if not pd.isna(zigzag_high.iloc[j]):
                    prev_indices.append(j)
                if len(prev_indices) >= 1:
                    break
            
            if len(prev_indices) >= 1:
                prev_idx = prev_indices[0]
                prev_zigzag_high = zigzag_high.iloc[prev_idx]
                prev_close = result_df['close'].iloc[prev_idx]
                
                long_condition = (zigzag_high.iloc[i] > prev_zigzag_high and
                                result_df['close'].iloc[i] - prev_close >= min_movement.iloc[i])
        
        # ShortMumu koşulu
        if not pd.isna(zigzag_low.iloc[i]):
            prev_indices = []
            for j in range(i-1, -1, -1):
                if not pd.isna(zigzag_low.iloc[j]):
                    prev_indices.append(j)
                if len(prev_indices) >= 1:
                    break
            
            if len(prev_indices) >= 1:
                prev_idx = prev_indices[0]
                prev_zigzag_low = zigzag_low.iloc[prev_idx]
                prev_close = result_df['close'].iloc[prev_idx]
                
                short_condition = (zigzag_low.iloc[i] < prev_zigzag_low and
                                 prev_close - result_df['close'].iloc[i] >= min_movement.iloc[i])
        
        long_mumu.append(long_condition)
        short_mumu.append(short_condition)

    # Hareketli ortalama ve bantlar
    ma = result_df['close'].rolling(ma_period).mean()
    upper_band = ma + std_mult * result_df['close'].rolling(ma_period).std()
    lower_band = ma - std_mult * result_df['close'].rolling(ma_period).std()

    # Sinyalleri float değerlere çevir
    long_mumu_float = pd.Series(np.where(long_mumu, result_df['close'], np.nan))
    short_mumu_float = pd.Series(np.where(short_mumu, result_df['close'], np.nan))

    # Mumu sinyallerinin hareketli ortalamaları
    long_mumu_ma = pine_sma(long_mumu_float, 2)
    short_mumu_ma = pine_sma(short_mumu_float, 2)

    # Pullback seviyeleri
    long_pullback_level = result_df['close'].where(long_mumu).ffill()
    short_pullback_level = result_df['close'].where(short_mumu).ffill()

    # Potansiyel mumlar
    short_potential_candles = ((result_df['close'] > ma) & 
                              crossunder(result_df['close'], long_pullback_level) & 
                              (~crossunder(result_df['close'], upper_band)))

    long_potential_candles2 = (crossover(result_df['close'], short_pullback_level) & 
                              crossover(result_df['close'], short_mumu_ma) & 
                              (result_df['close'] < ma))

    # Tüm sinyaller
    all_signals_values = []
    for i in range(len(result_df)):
        if long_mumu[i]:
            all_signals_values.append(result_df['low'].iloc[i])
        elif short_mumu[i]:
            all_signals_values.append(result_df['high'].iloc[i])
        else:
            all_signals_values.append(np.nan)

    all_signals = pd.Series(all_signals_values, index=result_df.index)

    # İkinci sinyal serisi
    all_signals2_values = []
    for i in range(len(result_df)):
        if short_potential_candles.iloc[i]:
            all_signals2_values.append(result_df['low'].iloc[i])
        elif long_potential_candles2.iloc[i]:
            all_signals2_values.append(result_df['high'].iloc[i])
        else:
            all_signals2_values.append(np.nan)

    all_signals2 = pd.Series(all_signals2_values, index=result_df.index)

    # Hareketli ortalamalar (3 çizgi)
    ma_all_signals = pine_sma(all_signals, ma_length)      # Mavi
    ma_all_signals2 = pine_sma(all_signals2, ma_length)    # Mor
    ma_mid = (ma_all_signals + ma_all_signals2) / 2        # Sarı

    # Trading sinyalleri
    long_crossover_signal = crossover(result_df['close'], ma_all_signals)
    short_crossover_signal = crossunder(result_df['close'], ma_all_signals)
    
    # Çapraz sinyaller
    cross_signals = cross(result_df['close'], ma_all_signals2) & cross(result_df['close'], ma_all_signals)

    # Pullback seviyesi hesaplama
    var_long_pullback_level2 = pd.Series(index=result_df.index, dtype=float)
    var_short_pullback_level2 = pd.Series(index=result_df.index, dtype=float)
    
    var_long_pullback_level2.iloc[0] = np.nan
    var_short_pullback_level2.iloc[0] = np.nan
    is_after_short = False
    
    for i in range(1, len(result_df)):
        var_long_pullback_level2.iloc[i] = var_long_pullback_level2.iloc[i-1]
        var_short_pullback_level2.iloc[i] = var_short_pullback_level2.iloc[i-1]
        
        if short_crossover_signal.iloc[i]:
            var_short_pullback_level2.iloc[i] = result_df['high'].iloc[i]
            var_long_pullback_level2.iloc[i] = var_short_pullback_level2.iloc[i]
            is_after_short = True
        
        if long_crossover_signal.iloc[i]:
            if pd.isna(var_long_pullback_level2.iloc[i]):
                var_long_pullback_level2.iloc[i] = result_df['low'].iloc[i]
            elif is_after_short:
                var_long_pullback_level2.iloc[i] = result_df['low'].iloc[i]
                is_after_short = False

    # Ratio yüzdesi hesaplama
    diff_percent = pd.Series(index=result_df.index, dtype=float)
    
    for i in range(len(result_df)):
        current_pullback_level = var_long_pullback_level2.iloc[i]
        current_price = result_df['close'].iloc[i]
        
        if pd.isna(current_pullback_level) or current_pullback_level == 0:
            diff_percent.iloc[i] = np.nan
        else:
            diff_percent.iloc[i] = ((current_price - current_pullback_level) / current_pullback_level) * 100

    # Sütunları ekle
    result_df['ratio_percent'] = diff_percent
    
    return result_df


def calculate_deviso_ratio(df: pd.DataFrame, 
                          zigzag_high_period: int = 10,
                          zigzag_low_period: int = 10,
                          min_movement_pct: float = 0.160,
                          ma_period: int = 20,
                          std_mult: float = 2.0,
                          ma_length: int = 10) -> float:
    """Deviso ratio hesaplama"""
    try:
        deviso_df = calculate_deviso_signals(
            df, zigzag_high_period, zigzag_low_period, 
            min_movement_pct, ma_period, std_mult, ma_length
        )
        
        if 'ratio_percent' in deviso_df.columns:
            last_ratio = deviso_df['ratio_percent'].iloc[-1]
            return float(last_ratio) if not pd.isna(last_ratio) else 0.0
        else:
            return 0.0
        
    except Exception as e:
        logger.debug(f"Deviso ratio hesaplama hatası: {e}")
        return 0.0


# ============================================================================
# CVD + DEVİSO UYUM ANALİZİ
# ============================================================================

def calculate_deviso_cvd_harmony(deviso_ratio: float, cvd_momentum: Dict, pressure_data: Dict) -> Dict:
    """
    Deviso Ratio ile CVD Momentum uyum hesaplama
    
    Args:
        deviso_ratio: Deviso ratio değeri
        cvd_momentum: CVD momentum metrikleri
        pressure_data: Hacim baskı metrikleri
        
    Returns:
        Dict: Uyum analizi
    """
    
    # Deviso yönü
    if deviso_ratio > 1.0:
        deviso_direction = 'bullish'
    elif deviso_ratio < -1.0:
        deviso_direction = 'bearish'
    else:
        deviso_direction = 'neutral'
    
    # CVD yönü
    cvd_direction = cvd_momentum.get('momentum_direction', 'neutral')
    
    # Baskı yönü
    pressure_dominance = pressure_data.get('pressure_dominance', 'neutral')
    
    # Uyum skoru hesaplama (0-100)
    harmony_score = 50  # Nötr başlangıç
    
    # 1. Deviso-CVD uyumu
    if deviso_direction == cvd_direction and deviso_direction != 'neutral':
        harmony_score += 20  # Güçlü uyum
    elif deviso_direction != 'neutral' and cvd_direction != 'neutral' and deviso_direction != cvd_direction:
        harmony_score -= 15  # Çelişki
    
    # 2. CVD-Pressure uyumu
    if (cvd_direction == 'bullish' and pressure_dominance == 'buy_dominant') or \
       (cvd_direction == 'bearish' and pressure_dominance == 'sell_dominant'):
        harmony_score += 15
    elif cvd_direction != 'neutral' and pressure_dominance != 'neutral':
        if not ((cvd_direction == 'bullish' and pressure_dominance == 'buy_dominant') or \
                (cvd_direction == 'bearish' and pressure_dominance == 'sell_dominant')):
            harmony_score -= 10
    
    # 3. Momentum gücü bonus
    momentum_strength = cvd_momentum.get('momentum_strength', 0.0)
    if momentum_strength > 70:
        harmony_score += 10
    elif momentum_strength < 30:
        harmony_score -= 5
    
    # 4. Deviso gücü bonus
    deviso_strength = min(abs(deviso_ratio), 10.0) / 10.0 * 100
    if deviso_strength > 70:
        harmony_score += 10
    elif deviso_strength < 30:
        harmony_score -= 5
    
    # Sınırla
    harmony_score = max(0, min(100, harmony_score))
    
    # Uyum seviyesi
    if harmony_score >= 80:
        harmony_level = 'excellent'
    elif harmony_score >= 65:
        harmony_level = 'good'
    elif harmony_score >= 45:
        harmony_level = 'fair'
    else:
        harmony_level = 'poor'
    
    return {
        'harmony_score': harmony_score,
        'harmony_level': harmony_level,
        'deviso_direction': deviso_direction,
        'cvd_direction': cvd_direction,
        'pressure_direction': pressure_dominance,
        'alignment_summary': f"{deviso_direction}|{cvd_direction}|{pressure_dominance}"
    }


def determine_cvd_signal_type(cvd_momentum: Dict, pressure_data: Dict, deviso_ratio: float) -> str:
    """CVD momentum bazlı sinyal tipi belirleme"""
    cvd_direction = cvd_momentum.get('momentum_direction', 'neutral')
    pressure_dominance = pressure_data.get('pressure_dominance', 'neutral')
    momentum_strength = cvd_momentum.get('momentum_strength', 0.0)
    
    # Güçlü sinyal kriterleri
    if momentum_strength > 60:
        if cvd_direction == 'bullish' and pressure_dominance == 'buy_dominant':
            return 'strong_long'
        elif cvd_direction == 'bearish' and pressure_dominance == 'sell_dominant':
            return 'strong_short'
    
    # Orta sinyal kriterleri
    if momentum_strength > 30:
        if cvd_direction == 'bullish' or pressure_dominance == 'buy_dominant':
            return 'long'
        elif cvd_direction == 'bearish' or pressure_dominance == 'sell_dominant':
            return 'short'
    
    # Deviso ratio ile teyit
    if abs(deviso_ratio) > 2.0:
        if deviso_ratio > 0:
            return 'long'
        else:
            return 'short'
    
    return 'neutral'


def calculate_signal_strength(harmony_data: Dict, cvd_momentum: Dict, deviso_ratio: float) -> float:
    """Toplam sinyal gücü hesaplama (0-100)"""
    
    # Uyum skoru (40% ağırlık)
    harmony_score = harmony_data.get('harmony_score', 50.0)
    harmony_weight = harmony_score * 0.4
    
    # CVD momentum gücü (35% ağırlık)
    momentum_strength = cvd_momentum.get('momentum_strength', 0.0)
    momentum_weight = momentum_strength * 0.35
    
    # Deviso gücü (25% ağırlık)
    deviso_strength = min(abs(deviso_ratio), 10.0) / 10.0 * 100
    deviso_weight = deviso_strength * 0.25
    
    # Toplam güç
    total_strength = harmony_weight + momentum_weight + deviso_weight
    
    return max(0.0, min(100.0, total_strength))


# ============================================================================
# ANA FONKSİYON: CVD MOMENTUM METRİKLERİ
# ============================================================================

def compute_cvd_momentum_metrics(df: pd.DataFrame) -> Dict:
    """
    ANA FONKSİYON: CVD + ROC Momentum sistemi
    
    KALDIRILAN ESKİ ALANLAR:
    ❌ run_type, run_count, run_perc
    ❌ gauss_run, gauss_run_perc
    ❌ vol_ratio, hh_vol_streak
    ❌ Ardışık mum analizi
    
    YENİ CVD ALANLARI:
    ✅ cvd_roc_momentum - Log ROC momentum
    ✅ cvd_direction - bullish/bearish/neutral
    ✅ momentum_strength - CVD momentum gücü (0-100)
    ✅ buy_pressure/sell_pressure - Alıcı/satıcı baskısı (%)
    ✅ deviso_cvd_harmony - Deviso+CVD uyum skoru (0-100)
    ✅ trend_strength - Kombinasyon güç skoru
    ✅ signal_type - CVD bazlı sinyal tipi
    
    Args:
        df: OHLCV verileri
        
    Returns:
        Dict: CVD momentum metrikleri
    """
    if df is None or df.empty:
        return {}

    # 1. CVD hesaplama
    cvd_data = calculate_cvd_data(df)
    
    # 2. CVD ROC momentum
    cvd_momentum = calculate_cvd_roc_momentum(cvd_data['cvd_values'])
    
    # 3. Volume pressure analizi
    pressure_data = calculate_volume_pressure(
        cvd_data['buy_volumes'], 
        cvd_data['sell_volumes']
    )
    
    # 4. Deviso ratio hesaplama
    try:
        deviso_ratio = calculate_deviso_ratio(
            df,
            DEVISO_PARAMS['zigzag_high_period'],
            DEVISO_PARAMS['zigzag_low_period'],
            DEVISO_PARAMS['min_movement_pct'],
            DEVISO_PARAMS['ma_period'],
            DEVISO_PARAMS['std_mult'],
            DEVISO_PARAMS['ma_length']
        )
        logger.debug(f"Deviso ratio hesaplandı: {deviso_ratio:.4f}")
    except Exception as e:
        logger.debug(f"Deviso ratio hesaplama hatası: {e}")
        deviso_ratio = 0.0

    # 5. Deviso-CVD uyum analizi
    harmony_data = calculate_deviso_cvd_harmony(deviso_ratio, cvd_momentum, pressure_data)

    # 6. Trend strength hesaplama
    cvd_strength = cvd_momentum.get('momentum_strength', 0.0)
    deviso_strength = min(abs(deviso_ratio), 10.0) / 10.0 * 100
    combined_strength = (cvd_strength * 0.6) + (deviso_strength * 0.4)

    return {
        # CVD MOMENTUM ALANLARI
        'cvd_roc_momentum': cvd_momentum.get('roc_momentum', 0.0),
        'cvd_direction': cvd_momentum.get('momentum_direction', 'neutral'),
        'momentum_strength': cvd_momentum.get('momentum_strength', 0.0),
        'momentum_acceleration': cvd_momentum.get('momentum_acceleration', 0.0),
        'buy_pressure': pressure_data.get('buy_pressure', 50.0),
        'sell_pressure': pressure_data.get('sell_pressure', 50.0),
        'pressure_dominance': pressure_data.get('pressure_dominance', 'neutral'),
        'pressure_intensity': pressure_data.get('pressure_intensity', 0.0),
        
        # UYUM VE GÜÇ ALANLARI
        'deviso_cvd_harmony': harmony_data.get('harmony_score', 50.0),
        'harmony_level': harmony_data.get('harmony_level', 'fair'),
        'trend_strength': combined_strength,
        'alignment_summary': harmony_data.get('alignment_summary', 'neutral|neutral|neutral'),
        
        # DEVİSO RATIO
        'deviso_ratio': float(deviso_ratio),
        
        # SINYAL TİPİ VE GÜCÜ
        'signal_type': determine_cvd_signal_type(cvd_momentum, pressure_data, deviso_ratio),
        'signal_strength': calculate_signal_strength(harmony_data, cvd_momentum, deviso_ratio)
    }


# ============================================================================
# ESKİ API UYUMLULUĞU (Geçici)
# ============================================================================

def compute_consecutive_metrics(df: pd.DataFrame) -> Dict:
    """
    UYUMLULUK: Eski API'yi yeni CVD sistemine yönlendir
    Bu fonksiyon eski kodların çalışması için gerekli
    """
    logger.warning("compute_consecutive_metrics() deprecated. compute_cvd_momentum_metrics() kullanın.")
    
    # Yeni sistemi çağır
    cvd_metrics = compute_cvd_momentum_metrics(df)
    
    # Eski field'ları emüle et (backward compatibility)
    return {
        # ESKİ ALANLAR (emülasyon - backward compatibility)
        'run_type': cvd_metrics.get('signal_type', 'neutral'),
        'run_count': _emulate_run_count(cvd_metrics),
        'run_perc': _emulate_run_perc(cvd_metrics),
        'gauss_run': _emulate_gauss_run(cvd_metrics),
        'gauss_run_perc': _emulate_gauss_run_perc(cvd_metrics),
        
        # YENİ ALANLAR (direkt aktar)
        'cvd_roc_momentum': cvd_metrics.get('cvd_roc_momentum', 0.0),
        'cvd_direction': cvd_metrics.get('cvd_direction', 'neutral'),
        'momentum_strength': cvd_metrics.get('momentum_strength', 0.0),
        'buy_pressure': cvd_metrics.get('buy_pressure', 50.0),
        'sell_pressure': cvd_metrics.get('sell_pressure', 50.0),
        'deviso_cvd_harmony': cvd_metrics.get('deviso_cvd_harmony', 50.0),
        'trend_strength': cvd_metrics.get('trend_strength', 0.0),
        
        # DEVİSO RATIO
        'deviso_ratio': cvd_metrics.get('deviso_ratio', 0.0)
    }


def _emulate_run_count(cvd_metrics: Dict) -> int:
    """CVD momentum strength'i run_count gibi emüle et"""
    momentum_strength = cvd_metrics.get('momentum_strength', 0.0)
    return int(momentum_strength / 10)  # 0-100 -> 0-10


def _emulate_run_perc(cvd_metrics: Dict) -> float:
    """CVD ROC momentum'unu run_perc gibi emüle et"""
    roc_momentum = cvd_metrics.get('cvd_roc_momentum', 0.0)
    return abs(roc_momentum) / 5.0  # ROC'u percentage'a normalize et


def _emulate_gauss_run(cvd_metrics: Dict) -> float:
    """Trend strength'i gauss_run gibi emüle et"""
    from core.utils import gauss_sum
    trend_strength = cvd_metrics.get('trend_strength', 0.0)
    equivalent_n = int(trend_strength / 15)  # 0-100 -> 0-6 range
    return gauss_sum(max(1, equivalent_n))


def _emulate_gauss_run_perc(cvd_metrics: Dict) -> float:
    """Signal strength'i gauss_run_perc gibi emüle et"""
    from core.utils import gauss_sum
    signal_strength = cvd_metrics.get('signal_strength', 0.0)
    equivalent_perc = signal_strength / 10  # 0-100 -> 0-10 range
    return gauss_sum(max(1, int(equivalent_perc)))


# ============================================================================
# DEVİSO DETAILED ANALYSIS (Korundu)
# ============================================================================

def get_deviso_detailed_analysis(df: pd.DataFrame) -> Dict:
    """Detaylı deviso analizi döndürür (debug ve analiz için)"""
    try:
        deviso_df = calculate_deviso_signals(
            df,
            DEVISO_PARAMS['zigzag_high_period'],
            DEVISO_PARAMS['zigzag_low_period'],
            DEVISO_PARAMS['min_movement_pct'],
            DEVISO_PARAMS['ma_period'],
            DEVISO_PARAMS['std_mult'],
            DEVISO_PARAMS['ma_length']
        )
        
        # Son değerler
        last_idx = -1
        
        analysis = {
            'current_ratio_percent': deviso_df['ratio_percent'].iloc[last_idx] if not pd.isna(deviso_df['ratio_percent'].iloc[last_idx]) else 0.0,
            'trend_direction': "Belirsiz"
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Detaylı deviso analizi hatası: {e}")
        return {
            'current_ratio_percent': 0.0,
            'trend_direction': "Belirsiz",
            'error': str(e)
        }