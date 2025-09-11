"""
📈 Teknik Göstergeler
Deviso ratio, ZigZag ve diğer teknik analiz göstergeleri
🔥 TAMAMEN YENİ DEVISO HESAPLAMA - Pine Script mantığıyla
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from config import DEVISO_PARAMS, VOL_SMA_LEN
from core.utils import gauss_sum, safe_log, pine_sma, crossunder, crossover, cross

logger = logging.getLogger("crypto-analytics")


def calculate_zigzag_high(high_series: pd.Series, period: int) -> pd.Series:
    """
    Pine Script: ta.highestbars(high, period) == 0 ? high : na
    ZigZag yüksek noktaları hesapla
    
    Args:
        high_series (pd.Series): Yüksek fiyat serisi
        period (int): Periyot
        
    Returns:
        pd.Series: ZigZag yüksek noktaları
    """
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
    """
    Pine Script: ta.lowestbars(low, period) == 0 ? low : na
    ZigZag düşük noktaları hesapla
    
    Args:
        low_series (pd.Series): Düşük fiyat serisi
        period (int): Periyot
        
    Returns:
        pd.Series: ZigZag düşük noktaları
    """
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
    """
    🔥 YENİ: Ham deviso kodundan alınan TAMAMEN DOĞRU hesaplama
    Deviso sinyallerini hesaplar ve temiz bir DataFrame döndürür.
    """
    
    # Kopya oluştur
    result_df = df.copy()
    
    # ---------- ZigZag Calculation ----------
    zigzag_high = calculate_zigzag_high(result_df['high'], zigzag_high_period)
    zigzag_low = calculate_zigzag_low(result_df['low'], zigzag_low_period)
    
    # ---------- Minimum Movement ----------
    min_movement = result_df['close'] * min_movement_pct / 100

    # ---------- LongMumu / ShortMumu Signals ----------
    long_mumu = []
    short_mumu = []

    for i in range(len(result_df)):
        long_condition = False
        short_condition = False
        
        # LongMumu condition
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
        
        # ShortMumu condition
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

    # ---------- Moving Average and Bands ----------
    ma = result_df['close'].rolling(ma_period).mean()
    upper_band = ma + std_mult * result_df['close'].rolling(ma_period).std()
    lower_band = ma - std_mult * result_df['close'].rolling(ma_period).std()

    # ---------- Convert signals to float values ----------
    long_mumu_float = pd.Series(np.where(long_mumu, result_df['close'], np.nan))
    short_mumu_float = pd.Series(np.where(short_mumu, result_df['close'], np.nan))

    # Calculate moving averages for Mumu signals
    long_mumu_ma = pine_sma(long_mumu_float, 2)
    short_mumu_ma = pine_sma(short_mumu_float, 2)

    # ---------- Pullback Levels ----------
    long_pullback_level = result_df['close'].where(long_mumu).ffill()
    short_pullback_level = result_df['close'].where(short_mumu).ffill()

    # ---------- Potential Candles ----------
    short_potential_candles = ((result_df['close'] > ma) & 
                              crossunder(result_df['close'], long_pullback_level) & 
                              (~crossunder(result_df['close'], upper_band)))

    long_potential_candles2 = (crossover(result_df['close'], short_pullback_level) & 
                              crossover(result_df['close'], short_mumu_ma) & 
                              (result_df['close'] < ma))

    # ---------- allSignals ----------
    all_signals_values = []
    for i in range(len(result_df)):
        if long_mumu[i]:
            all_signals_values.append(result_df['low'].iloc[i])
        elif short_mumu[i]:
            all_signals_values.append(result_df['high'].iloc[i])
        else:
            all_signals_values.append(np.nan)

    all_signals = pd.Series(all_signals_values, index=result_df.index)

    # ---------- allSignals2 ----------
    all_signals2_values = []
    for i in range(len(result_df)):
        if short_potential_candles.iloc[i]:
            all_signals2_values.append(result_df['low'].iloc[i])
        elif long_potential_candles2.iloc[i]:
            all_signals2_values.append(result_df['high'].iloc[i])
        else:
            all_signals2_values.append(np.nan)

    all_signals2 = pd.Series(all_signals2_values, index=result_df.index)

    # ---------- Calculate Moving Averages (3 lines) ----------
    ma_all_signals = pine_sma(all_signals, ma_length)      # Mavi (Blue)
    ma_all_signals2 = pine_sma(all_signals2, ma_length)    # Mor (Purple)
    ma_mid = (ma_all_signals + ma_all_signals2) / 2        # Sarı (Yellow)

    # ---------- Trading Signals (Pine Script Logic) ----------
    long_crossover_signal = crossover(result_df['close'], ma_all_signals)
    short_crossover_signal = crossunder(result_df['close'], ma_all_signals)
    
    # Cross signals
    cross_signals = cross(result_df['close'], ma_all_signals2) & cross(result_df['close'], ma_all_signals)

    # ---------- Pullback Level for Ratio Calculation ----------
    var_long_pullback_level2 = pd.Series(index=result_df.index, dtype=float)
    var_short_pullback_level2 = pd.Series(index=result_df.index, dtype=float)
    is_after_short_flag = pd.Series([False] * len(result_df), index=result_df.index)
    
    # İlk değerleri başlat
    var_long_pullback_level2.iloc[0] = np.nan
    var_short_pullback_level2.iloc[0] = np.nan
    is_after_short = False
    
    for i in range(1, len(result_df)):
        # Önceki değerleri koru
        var_long_pullback_level2.iloc[i] = var_long_pullback_level2.iloc[i-1]
        var_short_pullback_level2.iloc[i] = var_short_pullback_level2.iloc[i-1]
        is_after_short_flag.iloc[i] = is_after_short
        
        # Short sinyali geldiğinde
        if short_crossover_signal.iloc[i]:
            var_short_pullback_level2.iloc[i] = result_df['high'].iloc[i]
            var_long_pullback_level2.iloc[i] = var_short_pullback_level2.iloc[i]
            is_after_short = True
        
        # Long sinyali geldiğinde
        if long_crossover_signal.iloc[i]:
            if pd.isna(var_long_pullback_level2.iloc[i]):
                var_long_pullback_level2.iloc[i] = result_df['low'].iloc[i]
            elif is_after_short:
                var_long_pullback_level2.iloc[i] = result_df['low'].iloc[i]
                is_after_short = False

    # ---------- Calculate Ratio Percentage ----------
    diff_percent = pd.Series(index=result_df.index, dtype=float)
    
    for i in range(len(result_df)):
        current_pullback_level = var_long_pullback_level2.iloc[i]
        current_price = result_df['close'].iloc[i]
        
        if pd.isna(current_pullback_level) or current_pullback_level == 0:
            diff_percent.iloc[i] = np.nan
        else:
            diff_percent.iloc[i] = ((current_price - current_pullback_level) / current_pullback_level) * 100

    # ---------- Add all columns to result DataFrame ----------
    result_df['zigzag_high'] = zigzag_high
    result_df['zigzag_low'] = zigzag_low
    result_df['long_mumu'] = long_mumu
    result_df['short_mumu'] = short_mumu
    result_df['ma'] = ma
    result_df['upper_band'] = upper_band
    result_df['lower_band'] = lower_band
    result_df['long_pullback_level'] = long_pullback_level
    result_df['short_pullback_level'] = short_pullback_level
    result_df['short_potential_candles'] = short_potential_candles
    result_df['long_potential_candles2'] = long_potential_candles2
    result_df['all_signals'] = all_signals
    result_df['all_signals2'] = all_signals2
    result_df['ma_all_signals'] = ma_all_signals
    result_df['ma_all_signals2'] = ma_all_signals2
    result_df['ma_mid'] = ma_mid
    result_df['short_signal'] = short_crossover_signal
    result_df['long_signal'] = long_crossover_signal
    result_df['cross_signal'] = cross_signals
    result_df['long_mumu_ma'] = long_mumu_ma
    result_df['short_mumu_ma'] = short_mumu_ma
    result_df['var_long_pullback_level2'] = var_long_pullback_level2
    result_df['var_short_pullback_level2'] = var_short_pullback_level2
    result_df['ratio_percent'] = diff_percent
    
    return result_df


def calculate_deviso_ratio(df: pd.DataFrame, 
                          zigzag_high_period: int = 10,
                          zigzag_low_period: int = 10,
                          min_movement_pct: float = 0.160,
                          ma_period: int = 20,
                          std_mult: float = 2.0,
                          ma_length: int = 10) -> float:
    """
    🔥 TAMAMEN YENİ: Deviso ratio hesaplama - Doğru Pine Script mantığı
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        zigzag_high_period (int): ZigZag yüksek periyodu
        zigzag_low_period (int): ZigZag düşük periyodu
        min_movement_pct (float): Minimum hareket yüzdesi
        ma_period (int): Hareketli ortalama periyodu
        std_mult (float): Standart sapma çarpanı
        ma_length (int): MA uzunluğu
        
    Returns:
        float: Deviso ratio değeri
    """
    try:
        # Tam deviso hesaplama fonksiyonunu çağır
        deviso_df = calculate_deviso_signals(
            df,
            zigzag_high_period,
            zigzag_low_period,
            min_movement_pct,
            ma_period,
            std_mult,
            ma_length
        )
        
        # Son ratio_percent değerini döndür
        if 'ratio_percent' in deviso_df.columns:
            last_ratio = deviso_df['ratio_percent'].iloc[-1]
            return float(last_ratio) if not pd.isna(last_ratio) else 0.0
        else:
            return 0.0
        
    except Exception as e:
        logger.debug(f"Deviso ratio hesaplama hatası: {e}")
        return 0.0


def compute_consecutive_metrics(df: pd.DataFrame) -> Dict:
    """
    Pine Script mantığıyla TAMAMEN AYNI hesaplama
    Ardışık mum analizi ve volume metrikleri
    🔥 DEVISO RATIO ENTEGRE EDİLDİ
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        
    Returns:
        Dict: Hesaplanan metrikler
    """
    if df is None or df.empty:
        return {}

    # Ardışık sayaçlar
    count_long = 0.0
    long_start_low = None  
    count_short = 0.0
    short_start_high = None

    # Her mumu analiz et
    for i in range(len(df)):
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
        
        # Mum tipi kontrolü
        is_long = c > o
        is_short = c < o
        
        # Long (yeşil) mum işlemi
        if is_long:
            count_long = count_long + 1.0
            if long_start_low is None or count_long == 1.0:
                long_start_low = l
        else:
            count_long = 0.0
            long_start_low = None  
        
        # Short (kırmızı) mum işlemi
        if is_short:
            count_short = count_short + 1.0
            if short_start_high is None or count_short == 1.0:
                short_start_high = h
        else:
            count_short = 0.0
            short_start_high = None  

    # Son değerler
    last_high = df.iloc[-1]['high']
    last_low = df.iloc[-1]['low']

    # Yüzde hesaplamaları
    long_perc = None
    short_perc = None
    
    if count_long > 0 and long_start_low is not None:
        long_perc = (last_high - long_start_low) / long_start_low * 100.0
    
    if count_short > 0 and short_start_high is not None:
        short_perc = (short_start_high - last_low) / short_start_high * 100.0

    # Sonuç belirleme
    if count_long > 0:
        run_type = 'long'
        run_count = int(count_long)
        run_perc = long_perc
        gauss_run = gauss_sum(count_long)
        
        if long_perc is not None:
            gauss_run_perc = gauss_sum(round(long_perc, 2))
        else:
            gauss_run_perc = None
    elif count_short > 0:
        run_type = 'short'
        run_count = int(count_short)
        run_perc = short_perc
        gauss_run = gauss_sum(count_short)
        
        if short_perc is not None:
            gauss_run_perc = gauss_sum(round(short_perc, 2))
        else:
            gauss_run_perc = None
    else:
        run_type = 'none'
        run_count = 0
        run_perc = None
        gauss_run = 0.0
        gauss_run_perc = None

    # Volume analizi
    vol = df['volume'].astype(float)
    
    vol_positive_mask = vol > 0
    if not vol_positive_mask.any():
        vol_ratio = None
        hh_streak = 0
    else:
        valid_count = vol_positive_mask.sum()
        if valid_count < VOL_SMA_LEN:
            vol_ratio = None
        else:
            log_vol = pd.Series(index=vol.index, dtype=float)
            log_vol[vol_positive_mask] = vol[vol_positive_mask].apply(safe_log)
            
            log_vol_sma = log_vol.rolling(VOL_SMA_LEN, min_periods=VOL_SMA_LEN).mean()
            
            last_valid_indices = vol_positive_mask[vol_positive_mask].index
            if len(last_valid_indices) > 0:
                last_valid_idx = last_valid_indices[-1]
                last_log_vol = log_vol[last_valid_idx]
                last_log_sma = log_vol_sma[last_valid_idx]
                
                if pd.notna(last_log_sma):
                    log_ratio = last_log_vol - last_log_sma
                    vol_ratio = math.exp(log_ratio)
                else:
                    vol_ratio = None
            else:
                vol_ratio = None

        # HH volume streak hesaplama
        hh_streak = 0
        if valid_count > 1:
            vol_positive_values = vol[vol_positive_mask]
            log_vol_positive = vol_positive_values.apply(safe_log)
            
            for i in range(len(log_vol_positive) - 1, 0, -1):
                if log_vol_positive.iloc[i] > log_vol_positive.iloc[i - 1]:
                    hh_streak += 1
                else:
                    break
        else:
            hh_streak = 0

    # 🔥 YENİ DEVISO RATIO HESAPLAMA - TAM DOĞRULANMIş
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
        logger.debug(f"🎯 Deviso ratio hesaplandı: {deviso_ratio:.4f}")
    except Exception as e:
        logger.debug(f"Deviso ratio hesaplama hatası: {e}")
        deviso_ratio = 0.0

    return {
        'run_type': run_type,
        'run_count': run_count,
        'run_perc': float(run_perc) if run_perc is not None else None,
        'gauss_run': float(gauss_run),
        'gauss_run_perc': float(gauss_run_perc) if gauss_run_perc is not None else None,
        'vol_ratio': float(vol_ratio) if vol_ratio is not None else None,
        'hh_vol_streak': int(hh_streak),
        'deviso_ratio': float(deviso_ratio)  # 🔥 YENİ DOĞRU HESAPLAMA
    }


# 🔥 YENİ: Detaylı deviso analizi için ek fonksiyon
def get_deviso_detailed_analysis(df: pd.DataFrame) -> Dict:
    """
    Detaylı deviso analizi döndürür (debug ve analiz için)
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        
    Returns:
        Dict: Detaylı analiz sonuçları
    """
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
            'ma_all_signals': deviso_df['ma_all_signals'].iloc[last_idx] if not pd.isna(deviso_df['ma_all_signals'].iloc[last_idx]) else 0.0,
            'ma_all_signals2': deviso_df['ma_all_signals2'].iloc[last_idx] if not pd.isna(deviso_df['ma_all_signals2'].iloc[last_idx]) else 0.0,
            'ma_mid': deviso_df['ma_mid'].iloc[last_idx] if not pd.isna(deviso_df['ma_mid'].iloc[last_idx]) else 0.0,
            'long_signal': bool(deviso_df['long_signal'].iloc[last_idx]),
            'short_signal': bool(deviso_df['short_signal'].iloc[last_idx]),
            'current_price': float(deviso_df['close'].iloc[last_idx]),
            'long_mumu_count': sum(deviso_df['long_mumu']),
            'short_mumu_count': sum(deviso_df['short_mumu']),
            'var_long_pullback_level2': deviso_df['var_long_pullback_level2'].iloc[last_idx] if not pd.isna(deviso_df['var_long_pullback_level2'].iloc[last_idx]) else 0.0,
        }
        
        # Trend direction
        current_price = analysis['current_price']
        ma_all_signals = analysis['ma_all_signals']
        ma_all_signals2 = analysis['ma_all_signals2']
        
        if current_price > ma_all_signals and current_price > ma_all_signals2:
            analysis['trend_direction'] = "Yükseliş"
        elif current_price < ma_all_signals and current_price < ma_all_signals2:
            analysis['trend_direction'] = "Düşüş"
        else:
            analysis['trend_direction'] = "Yan"
            
        return analysis
        
    except Exception as e:
        logger.error(f"Detaylı deviso analizi hatası: {e}")
        return {
            'current_ratio_percent': 0.0,
            'trend_direction': "Belirsiz",
            'error': str(e)
        }