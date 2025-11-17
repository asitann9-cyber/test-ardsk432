"""
📈 Teknik Göstergeler - ULTRA PANEL v5
Multi-timeframe Heikin Ashi analizi - Pine Script %100 uyumlu
🔥 ÖZELLIKLER: HTF crossover, Power calculation, Whale detection, Memory system
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List

from config import VOL_SMA_LEN
from core.utils import safe_log

logger = logging.getLogger("crypto-analytics")

# Global memory (Pine Script var benzeri)
_symbol_memory = {}


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
    🔥 Ultra Panel v5 - Candle Power hesaplama (Pine Script f_candle_power)
    
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
        # Percent change (Pine: math.abs((h - l) / l * 100))
        percent_change = abs((ha_high - ha_low) / ha_low * 100) if ha_low != 0 else 0
        
        # Body size & ratio
        body_size = abs(ha_close - ha_open)
        candle_range = ha_high - ha_low
        body_ratio = body_size / candle_range if candle_range != 0 else 0
        
        # Volume ratio
        vol_ratio = volume / volume_ma if volume_ma != 0 else 1
        
        # Strong candle check (Pine: percent_change > min_candle_change)
        is_strong = percent_change > min_candle_change  # ✅ > (>= değil!)
        
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
# HTF BAR OLUŞTURMA (Pine request.security benzeri)
# =======================================
def get_htf_bars(df: pd.DataFrame, ha_df: pd.DataFrame, multiplier: int) -> Optional[Dict]:
    """
    🔥 HTF bar oluştur (Pine Script request.security simülasyonu)
    
    Pine Script: lookahead=barmerge.lookahead_on
    → Mevcut tamamlanmamış HTF bar'ı döndürür
    
    Args:
        df: Base timeframe OHLCV
        ha_df: Heikin Ashi verileri
        multiplier: HTF çarpanı (8=4H, 12=6H, 16=8H, 24=12H)
        
    Returns:
        Dict: {current: {...}, previous: {...}} veya None
    """
    try:
        if len(df) < multiplier * 2:
            return None
        
        # ✅ MEVCUT HTF BAR (son N bar)
        current_period = ha_df.iloc[-multiplier:]
        current_vol_period = df.iloc[-multiplier:]
        
        if len(current_period) < multiplier:
            return None
        
        current_bar = {
            'open': float(current_period['ha_open'].iloc[0]),
            'close': float(current_period['ha_close'].iloc[-1]),
            'high': float(current_period['ha_high'].max()),
            'low': float(current_period['ha_low'].min()),
            'volume': float(current_vol_period['volume'].sum())
        }
        
        # ✅ ÖNCEKİ HTF BAR (ondan önceki N bar)
        previous_period = ha_df.iloc[-multiplier*2:-multiplier]
        previous_vol_period = df.iloc[-multiplier*2:-multiplier]
        
        if len(previous_period) < multiplier:
            return None
        
        previous_bar = {
            'open': float(previous_period['ha_open'].iloc[0]),
            'close': float(previous_period['ha_close'].iloc[-1]),
            'high': float(previous_period['ha_high'].max()),
            'low': float(previous_period['ha_low'].min()),
            'volume': float(previous_vol_period['volume'].sum())
        }
        
        return {
            'current': current_bar,
            'previous': previous_bar
        }
        
    except Exception as e:
        logger.debug(f"HTF bar oluşturma hatası (mult={multiplier}): {e}")
        return None


# =======================================
# HTF CROSSOVER DETECTION
# =======================================
def detect_htf_crossovers(df: pd.DataFrame, htf_multiples: List[int] = [8, 12, 16, 24]) -> Dict:
    """
    🔥 Multi-timeframe Heikin Ashi crossover tespiti (Pine Script %100 uyumlu)
    
    Pine Script logic:
    - ta.crossover(ha_c, ha_o): current > AND previous <=
    - ta.crossunder(ha_c, ha_o): current < AND previous >=
    
    Args:
        df (pd.DataFrame): Base timeframe OHLCV verileri
        htf_multiples (list): HTF multiplier'ları [8, 12, 16, 24]
        
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
        
        # Heikin Ashi hesapla (tüm dataframe için)
        ha_df = calculate_heikin_ashi(df)
        
        htf_signals = {}
        bull_count = 0
        bear_count = 0
        
        for mult in htf_multiples:
            try:
                # HTF bar'ları al
                htf_bars = get_htf_bars(df, ha_df, mult)
                if htf_bars is None:
                    continue
                
                current = htf_bars['current']
                previous = htf_bars['previous']
                
                # ✅ CROSSOVER DETECTION (Pine Script ta.crossover/crossunder)
                # ta.crossover: current close > current open AND previous close <= previous open
                bull_cross = (current['close'] > current['open']) and (previous['close'] <= previous['open'])
                
                # ta.crossunder: current close < current open AND previous close >= previous open
                bear_cross = (current['close'] < current['open']) and (previous['close'] >= previous['open'])
                
                # Volume MA (son 20 HTF period için)
                vol_ma_bars = min(len(df), mult * 20)
                vol_ma = df['volume'].tail(vol_ma_bars).mean() * mult
                
                # Power hesapla
                power, is_strong = calculate_candle_power(
                    current['high'],
                    current['low'],
                    current['open'],
                    current['close'],
                    current['volume'],
                    vol_ma,
                    min_candle_change=3.0,
                    use_volume=True
                )
                
                htf_signals[f'htf{mult}'] = {
                    'bull_cross': bull_cross,
                    'bear_cross': bear_cross,
                    'power': float(power),
                    'is_strong': is_strong,
                    'current_close': current['close'],
                    'current_open': current['open'],
                    'previous_close': previous['close'],
                    'previous_open': previous['open']
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
    🔥 Ultra Panel v5 - Kümülatif güç hesaplama (Pine Script uyumlu)
    
    Pine Script logic:
    - Sadece crossover OLAN ve strong OLAN HTF'ler sayılır
    - 4/4 HTF → 2.0x multiplier
    - 3/4 HTF → 1.5x multiplier
    
    Args:
        htf_results (Dict): HTF crossover sonuçları
        
    Returns:
        float: Kümülatif power skoru
    """
    try:
        cumulative_power = 0.0
        
        if htf_results['ultra_buy']:
            # ✅ Bull power topla (sadece bull_cross=True VE is_strong=True olanlar)
            for htf_key, htf_data in htf_results['htf_signals'].items():
                if htf_data['bull_cross'] and htf_data['is_strong']:
                    cumulative_power += htf_data['power']
            
            # ✅ Multiplier uygula
            if htf_results['bull_count'] >= 4:
                cumulative_power *= 2.0
            elif htf_results['bull_count'] >= 3:
                cumulative_power *= 1.5
                
        elif htf_results['ultra_sell']:
            # ✅ Bear power topla
            for htf_key, htf_data in htf_results['htf_signals'].items():
                if htf_data['bear_cross'] and htf_data['is_strong']:
                    cumulative_power += htf_data['power']
            
            # ✅ Multiplier uygula
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
    
    Pine Script:
    - request.security(symbol, "D", [volume, ta.sma(volume, 50)])
    - volume_spike = daily_vol > daily_vol_ma50 * whale_spike_mult
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        whale_mult (float): Volume spike çarpanı
        
    Returns:
        bool: Whale aktif mi?
    """
    try:
        if df is None or df.empty or len(df) < 50:
            return False
        
        # ✅ Günlük volume simülasyonu
        # Base TF'yi bul (örnek: 1h base → 24 bar = 1 gün)
        # Basit yaklaşım: Son 24 bar'ı günlük say (1h TF için)
        daily_bars = 24  # 1 saatlik TF için
        
        # Son günlük hacim
        daily_vol = df['volume'].tail(daily_bars).sum() if len(df) >= daily_bars else df['volume'].iloc[-1]
        
        # Günlük 50-period MA (her gün için 24 bar)
        daily_periods = []
        for i in range(50):
            start_idx = -(daily_bars * (i + 1))
            end_idx = -(daily_bars * i) if i > 0 else None
            if start_idx < -len(df):
                break
            period_vol = df['volume'].iloc[start_idx:end_idx].sum()
            daily_periods.append(period_vol)
        
        if not daily_periods:
            return False
        
        daily_vol_ma50 = np.mean(daily_periods)
        
        # ✅ Volume spike kontrolü
        volume_spike = daily_vol > (daily_vol_ma50 * whale_mult)
        
        if not volume_spike:
            return False
        
        # ✅ Heikin Ashi günlük mum yönü kontrolü
        # Son 24 bar'ı al ve HA hesapla
        daily_df = df.tail(daily_bars)
        ha_daily = calculate_heikin_ashi(daily_df)
        
        if ha_daily.empty:
            return False
        
        # Günlük HA bar
        ha_daily_open = ha_daily['ha_open'].iloc[0]
        ha_daily_close = ha_daily['ha_close'].iloc[-1]
        
        # Whale buy veya sell
        whale_buy = volume_spike and (ha_daily_close > ha_daily_open)
        whale_sell = volume_spike and (ha_daily_close < ha_daily_open)
        
        return whale_buy or whale_sell
        
    except Exception as e:
        logger.debug(f"Whale detection hatası: {e}")
        return False


# =======================================
# MEMORY SYSTEM (Pine Script var benzeri)
# =======================================
def check_new_ultra_signal(symbol: str, htf_results: Dict) -> Tuple[bool, int]:
    """
    🔥 Yeni ultra sinyal kontrolü (Pine Script memory sistemi)
    
    Pine Script:
    new_ultra = (buy or sell) and (bull_count[1] < 3 and bear_count[1] < 3)
    
    Args:
        symbol: Trading sembolü
        htf_results: Mevcut HTF sonuçları
        
    Returns:
        Tuple[bool, int]: (is_new_signal, bars_ago)
    """
    global _symbol_memory
    
    try:
        current_ultra = htf_results['ultra_buy'] or htf_results['ultra_sell']
        
        # Memory'de yoksa yeni sinyal
        if symbol not in _symbol_memory:
            if current_ultra:
                _symbol_memory[symbol] = {
                    'bull_count': htf_results['bull_count'],
                    'bear_count': htf_results['bear_count'],
                    'bars_since': 0,
                    'ultra_active': True
                }
                return True, 0  # Yeni sinyal, 0 bar önce
            return False, 0
        
        mem = _symbol_memory[symbol]
        
        # ✅ YENİ ULTRA SİNYAL KONTROLÜ
        # Pine: new_ultra = (current ultra) AND (previous[1] < 3)
        previous_had_no_ultra = (mem['bull_count'] < 3 and mem['bear_count'] < 3)
        is_new_signal = current_ultra and previous_had_no_ultra
        
        if is_new_signal:
            # Yeni ultra sinyal başladı
            _symbol_memory[symbol] = {
                'bull_count': htf_results['bull_count'],
                'bear_count': htf_results['bear_count'],
                'bars_since': 0,
                'ultra_active': True
            }
            return True, 0
        
        # Ultra hala devam ediyor
        if current_ultra and mem['ultra_active']:
            mem['bars_since'] += 1
            mem['bull_count'] = htf_results['bull_count']
            mem['bear_count'] = htf_results['bear_count']
            return False, mem['bars_since']  # Eski sinyal, N bar önce
        
        # Ultra kayboldu
        if not current_ultra:
            mem['ultra_active'] = False
            mem['bull_count'] = htf_results['bull_count']
            mem['bear_count'] = htf_results['bear_count']
        
        return False, mem.get('bars_since', 0)
        
    except Exception as e:
        logger.debug(f"Memory check hatası: {e}")
        return False, 0


# =======================================
# MAIN COMPUTE FUNCTION - ULTRA PANEL v5
# =======================================
def compute_consecutive_metrics(df: pd.DataFrame, symbol: str = None) -> Dict:
    """
    🔥 ULTRA PANEL v5 - Ana metrik hesaplama fonksiyonu
    
    Pine Script Ultra Panel v5'in %100 uyumlu Python implementasyonu:
    - Multi-timeframe Heikin Ashi crossover (4H, 6H, 8H, 12H)
    - Ultra Signal detection (3/4 veya 4/4 HTF)
    - Cumulative power calculation
    - Whale volume spike detection
    - Memory system (ta.barssince benzeri)
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        symbol (str): Trading sembolü (memory için)
        
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
        
        # 🔥 MEMORY SYSTEM (bars ago)
        bars_ago = 0
        if symbol:
            is_new, bars_ago = check_new_ultra_signal(symbol, htf_results)
        
        # 🔥 RETURN ULTRA PANEL v5 METRICS
        return {
            'run_type': run_type,
            'ultra_signal': ultra_signal,
            'htf_count': htf_count,
            'total_power': float(total_power),
            'whale_active': whale_active,
            'bars_ago': bars_ago,  # ✅ YENİ
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
            'bars_ago': 0,
            'htf_details': {}
        }


# =======================================
# BACKWARD COMPATIBILITY
# =======================================
def get_ultra_signal_summary(df: pd.DataFrame, symbol: str = None) -> str:
    """
    Ultra Panel özet bilgisi (debugging için)
    
    Args:
        df (pd.DataFrame): OHLCV verileri
        symbol (str): Trading sembolü
        
    Returns:
        str: Özet string
    """
    try:
        metrics = compute_consecutive_metrics(df, symbol)
        
        if metrics['ultra_signal'] == 'NONE':
            return "❌ Ultra sinyal yok"
        
        signal = metrics['ultra_signal']
        htf = metrics['htf_count']
        power = metrics['total_power']
        whale = "🐋" if metrics['whale_active'] else ""
        bars_ago = metrics.get('bars_ago', 0)
        
        return f"🔥 {signal} | HTF: {htf}/4 | Power: {power:.1f} | Bars ago: {bars_ago} {whale}"
        
    except Exception as e:
        return f"⚠️ Hata: {e}"