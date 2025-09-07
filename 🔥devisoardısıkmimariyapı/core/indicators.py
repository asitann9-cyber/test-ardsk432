"""
📈 Teknik Göstergeler
Deviso ratio, ZigZag ve diğer teknik analiz göstergeleri
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from config import DEVISO_PARAMS, VOL_SMA_LEN
from core.utils import gauss_sum, safe_log, pine_sma, crossunder, crossover

logger = logging.getLogger("crypto-analytics")


def calculate_zigzag_high(high_series: pd.Series, period: int) -> pd.Series:
    """
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


def calculate_deviso_ratio(df: pd.DataFrame, 
                          zigzag_high_period: int = 10,
                          zigzag_low_period: int = 10,
                          min_movement_pct: float = 0.160,
                          ma_period: int = 20,
                          std_mult: float = 2.0,
                          ma_length: int = 10) -> float:
    """
    Deviso ratio hesaplama - Ana teknik gösterge
    
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
        # ZigZag hesapla
        zigzag_high = calculate_zigzag_high(df['high'], zigzag_high_period)
        zigzag_low = calculate_zigzag_low(df['low'], zigzag_low_period)
        
        # Minimum hareket miktarı
        min_movement = df['close'] * min_movement_pct / 100

        # Long ve short mumu listesi
        long_mumu = []
        short_mumu = []

        for i in range(len(df)):
            long_condition = False
            short_condition = False
            
            # Long kondisyon kontrolü
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
                    prev_close = df['close'].iloc[prev_idx]
                    
                    long_condition = (zigzag_high.iloc[i] > prev_zigzag_high and
                                    df['close'].iloc[i] - prev_close >= min_movement.iloc[i])
            
            # Short kondisyon kontrolü
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
                    prev_close = df['close'].iloc[prev_idx]
                    
                    short_condition = (zigzag_low.iloc[i] < prev_zigzag_low and
                                     prev_close - df['close'].iloc[i] >= min_movement.iloc[i])
            
            long_mumu.append(long_condition)
            short_mumu.append(short_condition)

        # Bollinger bantları
        ma = df['close'].rolling(ma_period).mean()
        upper_band = ma + std_mult * df['close'].rolling(ma_period).std()
        lower_band = ma - std_mult * df['close'].rolling(ma_period).std()

        # Float seriler
        long_mumu_float = pd.Series(np.where(long_mumu, df['close'], np.nan))
        short_mumu_float = pd.Series(np.where(short_mumu, df['close'], np.nan))

        # MA hesaplama
        long_mumu_ma = pine_sma(long_mumu_float, 2)
        short_mumu_ma = pine_sma(short_mumu_float, 2)

        # Pullback seviyeleri
        long_pullback_level = df['close'].where(long_mumu).ffill()
        short_pullback_level = df['close'].where(short_mumu).ffill()

        # Potansiyel mumlar
        short_potential_candles = ((df['close'] > ma) & 
                                  crossunder(df['close'], long_pullback_level) & 
                                  (~crossunder(df['close'], upper_band)))

        long_potential_candles2 = (crossover(df['close'], short_pullback_level) & 
                                  crossover(df['close'], short_mumu_ma) & 
                                  (df['close'] < ma))

        # Sinyal değerleri
        all_signals_values = []
        for i in range(len(df)):
            if long_mumu[i]:
                all_signals_values.append(df['low'].iloc[i])
            elif short_mumu[i]:
                all_signals_values.append(df['high'].iloc[i])
            else:
                all_signals_values.append(np.nan)

        all_signals = pd.Series(all_signals_values, index=df.index)

        # İkinci sinyal serisi
        all_signals2_values = []
        for i in range(len(df)):
            if short_potential_candles.iloc[i]:
                all_signals2_values.append(df['low'].iloc[i])
            elif long_potential_candles2.iloc[i]:
                all_signals2_values.append(df['high'].iloc[i])
            else:
                all_signals2_values.append(np.nan)

        all_signals2 = pd.Series(all_signals2_values, index=df.index)

        # MA hesaplamaları
        ma_all_signals = pine_sma(all_signals, ma_length)      
        ma_all_signals2 = pine_sma(all_signals2, ma_length)   
        ma_mid = (ma_all_signals + ma_all_signals2) / 2        

        # Crossover sinyalleri
        long_crossover_signal = crossover(df['close'], ma_all_signals)
        short_crossover_signal = crossunder(df['close'], ma_all_signals)

        # Değişken pullback seviyeleri
        var_long_pullback_level2 = pd.Series(index=df.index, dtype=float)
        var_short_pullback_level2 = pd.Series(index=df.index, dtype=float)
        is_after_short_flag = pd.Series([False] * len(df), index=df.index)
        
        # Başlangıç değerleri
        var_long_pullback_level2.iloc[0] = np.nan
        var_short_pullback_level2.iloc[0] = np.nan
        is_after_short = False
        
        for i in range(1, len(df)):
            # Önceki değerleri koru
            var_long_pullback_level2.iloc[i] = var_long_pullback_level2.iloc[i-1]
            var_short_pullback_level2.iloc[i] = var_short_pullback_level2.iloc[i-1]
            is_after_short_flag.iloc[i] = is_after_short
            
            # Short crossover kontrolü
            if short_crossover_signal.iloc[i]:
                var_short_pullback_level2.iloc[i] = df['high'].iloc[i]
                var_long_pullback_level2.iloc[i] = var_short_pullback_level2.iloc[i]
                is_after_short = True
            
            # Long crossover kontrolü
            if long_crossover_signal.iloc[i]:
                if pd.isna(var_long_pullback_level2.iloc[i]):
                    var_long_pullback_level2.iloc[i] = df['low'].iloc[i]
                elif is_after_short:
                    var_long_pullback_level2.iloc[i] = df['low'].iloc[i]
                    is_after_short = False

        # Yüzde fark hesaplama
        diff_percent = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            current_pullback_level = var_long_pullback_level2.iloc[i]
            current_price = df['close'].iloc[i]
            
            if pd.isna(current_pullback_level) or current_pullback_level == 0:
                diff_percent.iloc[i] = np.nan
            else:
                diff_percent.iloc[i] = ((current_price - current_pullback_level) / current_pullback_level) * 100

        # Son değeri döndür
        return diff_percent.iloc[-1] if not pd.isna(diff_percent.iloc[-1]) else 0.0
        
    except Exception as e:
        logger.debug(f"Deviso ratio hesaplama hatası: {e}")
        return 0.0


def compute_consecutive_metrics(df: pd.DataFrame) -> Dict:
    """
    Pine Script mantığıyla TAMAMEN AYNI hesaplama
    Ardışık mum analizi ve volume metrikleri
    
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

    # Deviso ratio hesaplama
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
        'deviso_ratio': float(deviso_ratio)
    }