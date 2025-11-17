"""
📈 Teknik Göstergeler - ULTRA PANEL v5
Multi-timeframe Heikin Ashi analizi ile güçlü sinyal tespiti
🔥 YENİ: Pine Script Ultra Panel v5 mantığı - Python'a çevrildi
🔥 ÖZELLIKLER: HTF crossover, Power calculation, Whale detection, Memory system
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from config import VOL_SMA_LEN
from core.utils import safe_log

logger = logging.getLogger("crypto-analytics")


# =======================================
# HEIKIN ASHI HESAPLAMA
# =======================================
def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heikin Ashi mumlarını hesapla
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        
    Returns:
        pd.DataFrame: Heikin Ashi OHLC verileri
    """
    try:
        ha_df = df.copy()
        
        # HA Close
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # HA Open (ilk değer normal open)
        ha_df['ha_open'] = df['open'].copy()
        for i in range(1, len(df)):
            ha_df.loc[ha_df.index[i], 'ha_open'] = (
                ha_df.loc[ha_df.index[i-1], 'ha_open'] + 
                ha_df.loc[ha_df.index[i-1], 'ha_close']
            ) / 2
        
        # HA High & Low
        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        return ha_df[['ha_open', 'ha_high', 'ha_low', 'ha_close']]
        
    except Exception as e:
        logger.error(f"Heikin Ashi hesaplama hatası: {e}")
        return pd.DataFrame()


# =======================================
# CANDLE POWER CALCULATION
# =======================================
def calculate_candle_power(
    ha_high: float, 
    ha_low: float, 
    ha_open: float, 
    ha_close: float, 
    volume: float, 
    volume_ma: float,
    min_candle_change: float = 3.0,
    use_volume: bool = True
) -> Tuple[float, bool]:
    """
    🔥 Ultra Panel v5 - Candle Power hesaplama
    
    Args:
        ha_high, ha_low, ha_open, ha_close: Heikin Ashi OHLC
        volume: Hacim
        volume_ma: Hacim ortalaması
        min_candle_change: Minimum mum değişimi %
        use_volume: Hacim kullanılsın mı
        
    Returns:
        Tuple[float, bool]: (power, is_strong)
    """
    try:
        # Percent change
        percent_change = abs((ha_high - ha_low) / ha_low * 100) if ha_low != 0 else 0
        
        # Body size & ratio
        body_size = abs(ha_close - ha_open)
        candle_range = ha_high - ha_low
        body_ratio = body_size / candle_range if candle_range != 0 else 0
        
        # Volume ratio
        vol_ratio = volume / volume_ma if volume_ma != 0 else 1
        
        # Strong candle check
        is_strong = percent_change > min_candle_change
        
        # Power calculation
        if is_strong:
            power = percent_change * body_ratio
            if use_volume:
                power *= math.log(1 + vol_ratio)
        else:
            power = 0.0
        
        return power, is_strong
        
    except Exception as e:
        logger.debug(f"Candle power hesaplama hatası: {e}")
        return 0.0, False


# =======================================
# HTF CROSSOVER DETECTION
# =======================================
def detect_htf_crossovers(df: pd.DataFrame, htf_multiples: list = [8, 12, 16, 24]) -> Dict:
    """
    🔥 Multi-timeframe Heikin Ashi crossover tespiti
    
    Args:
        df (pd.DataFrame): Base timeframe OHLCV verileri
        htf_multiples (list): HTF multiplier'ları [4H, 6H, 8H, 12H]
        
    Returns:
        Dict: HTF crossover bilgileri
    """
    try:
        if df is None or df.empty or len(df) < max(htf_multiples) * 2:
            return {
                'bull_count': 0,
                'bear_count': 0,
                'htf_signals': {},
                'ultra_buy': False,
                'ultra_sell': False
            }
        
        # Heikin Ashi hesapla
        ha_df = calculate_heikin_ashi(df)
        
        htf_signals = {}
        bull_count = 0
        bear_count = 0
        
        for mult in htf_multiples:
            try:
                # HTF için resample (simulate)
                # Her mult kadar mumu birleştir
                if len(df) < mult * 2:
                    continue
                
                # Son 2 HTF period'u al
                htf_data = []
                for i in range(2):
                    start_idx = -(mult * (i + 1))
                    end_idx = -(mult * i) if i > 0 else None
                    
                    period_df = ha_df.iloc[start_idx:end_idx]
                    if len(period_df) == 0:
                        continue
                    
                    htf_open = period_df['ha_open'].iloc[0]
                    htf_close = period_df['ha_close'].iloc[-1]
                    htf_high = period_df['ha_high'].max()
                    htf_low = period_df['ha_low'].min()
                    
                    # Volume (base df'den)
                    vol_period = df.iloc[start_idx:end_idx]
                    htf_vol = vol_period['volume'].sum()
                    
                    htf_data.append({
                        'open': htf_open,
                        'close': htf_close,
                        'high': htf_high,
                        'low': htf_low,
                        'volume': htf_vol
                    })
                
                if len(htf_data) < 2:
                    continue
                
                # Crossover detection
                current = htf_data[0]
                previous = htf_data[1]
                
                # Bull crossover: current close > current open AND previous close <= previous open
                bull_cross = (current['close'] > current['open']) and (previous['close'] <= previous['open'])
                
                # Bear crossover: current close < current open AND previous close >= previous open
                bear_cross = (current['close'] < current['open']) and (previous['close'] >= previous['open'])
                
                # Volume MA (son 20 HTF period için simüle et - basit)
                vol_ma = df['volume'].tail(mult * 20).mean() * mult
                
                # Power hesapla
                power, is_strong = calculate_candle_power(
                    current['high'],
                    current['low'],
                    current['open'],
                    current['close'],
                    current['volume'],
                    vol_ma
                )
                
                htf_signals[f'htf{mult}'] = {
                    'bull_cross': bull_cross,
                    'bear_cross': bear_cross,
                    'power': power,
                    'is_strong': is_strong,
                    'current_close': current['close'],
                    'current_open': current['open']
                }
                
                if bull_cross:
                    bull_count += 1
                if bear_cross:
                    bear_count += 1
                    
            except Exception as e:
                logger.debug(f"HTF {mult} hesaplama hatası: {e}")
                continue
        
        # Ultra signals (3/4 veya 4/4)
        ultra_buy = bull_count >= 3
        ultra_sell = bear_count >= 3
        
        return {
            'bull_count': bull_count,
            'bear_count': bear_count,
            'htf_signals': htf_signals,
            'ultra_buy': ultra_buy,
            'ultra_sell': ultra_sell
        }
        
    except Exception as e:
        logger.error(f"HTF crossover detection hatası: {e}")
        return {
            'bull_count': 0,
            'bear_count': 0,
            'htf_signals': {},
            'ultra_buy': False,
            'ultra_sell': False
        }


# =======================================
# CUMULATIVE POWER CALCULATION
# =======================================
def calculate_cumulative_power(htf_results: Dict) -> float:
    """
    🔥 Ultra Panel v5 - Kümülatif güç hesaplama
    
    Args:
        htf_results (Dict): HTF crossover sonuçları
        
    Returns:
        float: Kümülatif power skoru
    """
    try:
        cumulative_power = 0.0
        
        if htf_results['ultra_buy']:
            # Bull power topla
            for htf_key, htf_data in htf_results['htf_signals'].items():
                if htf_data['bull_cross'] and htf_data['is_strong']:
                    cumulative_power += htf_data['power']
            
            # Multiplier (4/4 → 2x, 3/4 → 1.5x)
            if htf_results['bull_count'] >= 4:
                cumulative_power *= 2.0
            elif htf_results['bull_count'] >= 3:
                cumulative_power *= 1.5
                
        elif htf_results['ultra_sell']:
            # Bear power topla
            for htf_key, htf_data in htf_results['htf_signals'].items():
                if htf_data['bear_cross'] and htf_data['is_strong']:
                    cumulative_power += htf_data['power']
            
            # Multiplier
            if htf_results['bear_count'] >= 4:
                cumulative_power *= 2.0
            elif htf_results['bear_count'] >= 3:
                cumulative_power *= 1.5
        
        return float(cumulative_power)
        
    except Exception as e:
        logger.debug(f"Cumulative power hesaplama hatası: {e}")
        return 0.0


# =======================================
# WHALE DETECTION
# =======================================
def detect_whale_activity(df: pd.DataFrame, whale_mult: float = 2.5) -> bool:
    """
    🔥 Ultra Panel v5 - Whale (balina) aktivitesi tespiti
    Günlük volume spike kontrolü
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        whale_mult (float): Volume spike çarpanı
        
    Returns:
        bool: Whale aktif mi?
    """
    try:
        if df is None or df.empty or len(df) < 50:
            return False
        
        # Son günlük hacim (simulate - son 24 saatlik veri veya en son değer)
        # Basit yaklaşım: son 1 bar'ın hacmi
        daily_vol = df['volume'].iloc[-1]
        
        # Günlük 50-bar MA (yaklaşık 50 gün)
        daily_vol_ma50 = df['volume'].tail(50).mean()
        
        # Volume spike kontrolü
        volume_spike = daily_vol > (daily_vol_ma50 * whale_mult)
        
        if not volume_spike:
            return False
        
        # Heikin Ashi günlük mum yönü kontrolü
        ha_df = calculate_heikin_ashi(df.tail(1))
        if ha_df.empty:
            return False
        
        ha_close = ha_df['ha_close'].iloc[-1]
        ha_open = ha_df['ha_open'].iloc[-1]
        
        # Whale buy veya sell
        whale_buy = volume_spike and (ha_close > ha_open)
        whale_sell = volume_spike and (ha_close < ha_open)
        
        return whale_buy or whale_sell
        
    except Exception as e:
        logger.debug(f"Whale detection hatası: {e}")
        return False


# =======================================
# MAIN COMPUTE FUNCTION - ULTRA PANEL v5
# =======================================
def compute_consecutive_metrics(df: pd.DataFrame) -> Dict:
    """
    🔥 ULTRA PANEL v5 - Ana metrik hesaplama fonksiyonu
    
    Pine Script Ultra Panel v5'in Python implementasyonu:
    - Multi-timeframe Heikin Ashi crossover (4H, 6H, 8H, 12H)
    - Ultra Signal detection (3/4 veya 4/4 HTF)
    - Cumulative power calculation
    - Whale volume spike detection
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        
    Returns:
        Dict: Ultra Panel v5 metrikleri
    """
    if df is None or df.empty:
        return {}

    try:
        # 🔥 HTF CROSSOVER ANALYSIS
        htf_results = detect_htf_crossovers(df, htf_multiples=[8, 12, 16, 24])
        
        # 🔥 ULTRA SIGNAL TYPE
        if htf_results['ultra_buy']:
            ultra_signal = 'BUY'
            run_type = 'long'
        elif htf_results['ultra_sell']:
            ultra_signal = 'SELL'
            run_type = 'short'
        else:
            ultra_signal = 'NONE'
            run_type = 'none'
        
        # 🔥 HTF COUNT
        htf_count = htf_results['bull_count'] if ultra_signal == 'BUY' else htf_results['bear_count']
        
        # 🔥 CUMULATIVE POWER
        total_power = calculate_cumulative_power(htf_results)
        
        # 🔥 WHALE DETECTION
        whale_active = detect_whale_activity(df)
        
        # 🔥 RETURN ULTRA PANEL v5 METRICS
        return {
            'run_type': run_type,
            'ultra_signal': ultra_signal,
            'htf_count': htf_count,
            'total_power': float(total_power),
            'whale_active': whale_active,
            'htf_details': htf_results['htf_signals']
        }
        
    except Exception as e:
        logger.error(f"Ultra Panel v5 metrik hesaplama hatası: {e}")
        return {
            'run_type': 'none',
            'ultra_signal': 'NONE',
            'htf_count': 0,
            'total_power': 0.0,
            'whale_active': False,
            'htf_details': {}
        }


# =======================================
# BACKWARD COMPATIBILITY (OPSIYONEL)
# =======================================
def get_ultra_signal_summary(df: pd.DataFrame) -> str:
    """
    Ultra Panel özet bilgisi (debugging için)
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        
    Returns:
        str: Özet string
    """
    try:
        metrics = compute_consecutive_metrics(df)
        
        if metrics['ultra_signal'] == 'NONE':
            return "❌ Ultra sinyal yok"
        
        signal = metrics['ultra_signal']
        htf = metrics['htf_count']
        power = metrics['total_power']
        whale = "🐋" if metrics['whale_active'] else ""
        
        return f"🔥 {signal} | HTF: {htf}/4 | Power: {power:.1f} {whale}"
        
    except Exception as e:
        return f"⚠️ Hata: {e}"