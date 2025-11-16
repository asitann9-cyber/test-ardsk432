"""
📈 Teknik Göstergeler - Ultra Panel v5 Multi-HTF Sistemi
🔥 Heikin Ashi Multi-Timeframe analizi
🔥 Candle Power hesaplaması
🔥 Ultra Signal (3/4 HTF crossover)
🔥 Whale Volume tespiti
🔥 Pine Script ile %100 uyumlu
🔥 VPMV SİSTEMİ KALDIRILDI - Sadece Ultra Panel
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

logger = logging.getLogger("crypto-analytics")


# ============================================
# HEIKIN ASHI HESAPLAMA
# ============================================

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heikin Ashi mumları hesapla
    Pine Script ticker.heikinashi() ile uyumlu
    
    Args:
        df (pd.DataFrame): Normal OHLCV verileri
        
    Returns:
        pd.DataFrame: Heikin Ashi OHLC verileri
    """
    try:
        ha_df = pd.DataFrame(index=df.index)
        
        # HA Close = (O + H + L + C) / 4
        ha_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # HA Open = (HA Open[prev] + HA Close[prev]) / 2
        ha_df['open'] = 0.0
        ha_df.loc[ha_df.index[0], 'open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        for i in range(1, len(ha_df)):
            ha_df.loc[ha_df.index[i], 'open'] = (
                ha_df['open'].iloc[i-1] + ha_df['close'].iloc[i-1]
            ) / 2
        
        # HA High = max(H, HA Open, HA Close)
        ha_df['high'] = df[['high']].join(ha_df[['open', 'close']]).max(axis=1)
        
        # HA Low = min(L, HA Open, HA Close)
        ha_df['low'] = df[['low']].join(ha_df[['open', 'close']]).min(axis=1)
        
        return ha_df
        
    except Exception as e:
        logger.error(f"Heikin Ashi hesaplama hatası: {e}")
        return df.copy()


# ============================================
# HTF DATA FETCH (RESAMPLE)
# ============================================

def get_htf_data(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """
    Higher Timeframe verisi oluştur (resample ile)
    Pine Script request.security() benzeri
    
    Args:
        df (pd.DataFrame): Base timeframe OHLCV
        multiplier (int): Zaman çarpanı (örn: 8 = 8x base TF)
        
    Returns:
        pd.DataFrame: HTF OHLCV verileri
    """
    try:
        if df is None or df.empty or len(df) < 2:
            return df.copy()
        
        # Base timeframe'i tespit et (dakika cinsinden)
        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
        htf_minutes = int(time_diff * multiplier)
        
        # Resample kuralı
        resample_rule = f'{htf_minutes}T'
        
        # OHLCV resample
        htf_df = pd.DataFrame()
        htf_df['open'] = df['open'].resample(resample_rule).first()
        htf_df['high'] = df['high'].resample(resample_rule).max()
        htf_df['low'] = df['low'].resample(resample_rule).min()
        htf_df['close'] = df['close'].resample(resample_rule).last()
        htf_df['volume'] = df['volume'].resample(resample_rule).sum()
        
        # NaN temizle
        htf_df = htf_df.dropna()
        
        return htf_df
        
    except Exception as e:
        logger.error(f"HTF data oluşturma hatası: {e}")
        return df.copy()


# ============================================
# CANDLE POWER HESAPLAMA
# ============================================

def calculate_candle_power(
    high: float, 
    low: float, 
    open_price: float, 
    close: float,
    volume: float,
    volume_ma: float,
    min_candle_change: float = 3.0,
    use_volume: bool = True
) -> Tuple[float, bool]:
    """
    Pine Script f_candle_power() fonksiyonu
    
    Args:
        high: Mum yüksek
        low: Mum düşük
        open_price: Açılış
        close: Kapanış
        volume: Hacim
        volume_ma: Hacim ortalaması
        min_candle_change: Minimum değişim %
        use_volume: Volume kullanılsın mı
        
    Returns:
        Tuple[float, bool]: (power, is_strong)
    """
    try:
        # Yüzde değişim
        percent_change = abs((high - low) / low * 100) if low != 0 else 0
        
        # Body size ve ratio
        body_size = abs(close - open_price)
        candle_range = high - low
        body_ratio = body_size / candle_range if candle_range != 0 else 0
        
        # Volume ratio
        vol_ratio = volume / volume_ma if volume_ma != 0 else 1
        
        # Strong candle kontrolü
        is_strong = percent_change > min_candle_change
        
        # Power hesaplama
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


# ============================================
# ULTRA SIGNAL DETECTION (MULTI-HTF)
# ============================================

def detect_ultra_signals(
    df: pd.DataFrame,
    tfmult8: int = 8,
    tfmult12: int = 12,
    tfmult16: int = 16,
    tfmult24: int = 24,
    min_candle_change: float = 3.0,
    use_volume_in_power: bool = True
) -> Dict:
    """
    🔥 Ultra Panel v5 - Multi-HTF Signal Detection
    Pine Script f_analyze_symbol() fonksiyonu
    
    Args:
        df: Base timeframe OHLCV verileri
        tfmult8: HTF8 çarpanı (≈4H)
        tfmult12: HTF12 çarpanı (≈6H)
        tfmult16: HTF16 çarpanı (≈8H)
        tfmult24: HTF24 çarpanı (≈12H)
        min_candle_change: Minimum candle değişim %
        use_volume_in_power: Volume kullanılsın mı
        
    Returns:
        Dict: Ultra signal verileri
    """
    try:
        if df is None or df.empty or len(df) < 50:
            return {
                'ultra_strong_buy': False,
                'ultra_strong_sell': False,
                'bull_count': 0,
                'bear_count': 0,
                'total_power': 0.0,
                'whale_active': False,
                'htf_data': {}
            }
        
        # 1. HTF verilerini oluştur
        htf4_df = get_htf_data(df, tfmult8)
        htf6_df = get_htf_data(df, tfmult12)
        htf8_df = get_htf_data(df, tfmult16)
        htf12_df = get_htf_data(df, tfmult24)
        
        # Minimum veri kontrolü
        if len(htf4_df) < 2 or len(htf6_df) < 2 or len(htf8_df) < 2 or len(htf12_df) < 2:
            logger.debug("HTF verisi yetersiz")
            return {
                'ultra_strong_buy': False,
                'ultra_strong_sell': False,
                'bull_count': 0,
                'bear_count': 0,
                'total_power': 0.0,
                'whale_active': False,
                'htf_data': {}
            }
        
        # 2. Her HTF için Heikin Ashi hesapla
        ha4 = calculate_heikin_ashi(htf4_df)
        ha6 = calculate_heikin_ashi(htf6_df)
        ha8 = calculate_heikin_ashi(htf8_df)
        ha12 = calculate_heikin_ashi(htf12_df)
        
        # 3. Volume MA hesapla (her HTF için)
        htf4_vol_ma = htf4_df['volume'].rolling(window=min(20, len(htf4_df))).mean()
        htf6_vol_ma = htf6_df['volume'].rolling(window=min(20, len(htf6_df))).mean()
        htf8_vol_ma = htf8_df['volume'].rolling(window=min(20, len(htf8_df))).mean()
        htf12_vol_ma = htf12_df['volume'].rolling(window=min(20, len(htf12_df))).mean()
        
        # 4. Son mumları al
        ha4_last = ha4.iloc[-1]
        ha6_last = ha6.iloc[-1]
        ha8_last = ha8.iloc[-1]
        ha12_last = ha12.iloc[-1]
        
        ha4_prev = ha4.iloc[-2] if len(ha4) > 1 else ha4.iloc[-1]
        ha6_prev = ha6.iloc[-2] if len(ha6) > 1 else ha6.iloc[-1]
        ha8_prev = ha8.iloc[-2] if len(ha8) > 1 else ha8.iloc[-1]
        ha12_prev = ha12.iloc[-2] if len(ha12) > 1 else ha12.iloc[-1]
        
        # 5. Crossover/Crossunder tespiti (Pine Script ta.crossover/crossunder mantığı)
        ha4_bull = (ha4_last['close'] > ha4_last['open']) and (ha4_prev['close'] <= ha4_prev['open'])
        ha4_bear = (ha4_last['close'] < ha4_last['open']) and (ha4_prev['close'] >= ha4_prev['open'])
        
        ha6_bull = (ha6_last['close'] > ha6_last['open']) and (ha6_prev['close'] <= ha6_prev['open'])
        ha6_bear = (ha6_last['close'] < ha6_last['open']) and (ha6_prev['close'] >= ha6_prev['open'])
        
        ha8_bull = (ha8_last['close'] > ha8_last['open']) and (ha8_prev['close'] <= ha8_prev['open'])
        ha8_bear = (ha8_last['close'] < ha8_last['open']) and (ha8_prev['close'] >= ha8_prev['open'])
        
        ha12_bull = (ha12_last['close'] > ha12_last['open']) and (ha12_prev['close'] <= ha12_prev['open'])
        ha12_bear = (ha12_last['close'] < ha12_last['open']) and (ha12_prev['close'] >= ha12_prev['open'])
        
        # 6. Count hesapla
        bull_count = sum([ha4_bull, ha6_bull, ha8_bull, ha12_bull])
        bear_count = sum([ha4_bear, ha6_bear, ha8_bear, ha12_bear])
        
        # 7. Ultra Signal (3/4 veya 4/4)
        ultra_strong_buy = bull_count >= 3
        ultra_strong_sell = bear_count >= 3
        
        # 8. Power hesaplama
        cumulative_bull_power = 0.0
        cumulative_bear_power = 0.0
        
        if ultra_strong_buy:
            # HTF4 power
            if ha4_bull:
                power4, strong4 = calculate_candle_power(
                    ha4_last['high'], ha4_last['low'], ha4_last['open'], ha4_last['close'],
                    htf4_df['volume'].iloc[-1], htf4_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong4:
                    cumulative_bull_power += power4
            
            # HTF6 power
            if ha6_bull:
                power6, strong6 = calculate_candle_power(
                    ha6_last['high'], ha6_last['low'], ha6_last['open'], ha6_last['close'],
                    htf6_df['volume'].iloc[-1], htf6_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong6:
                    cumulative_bull_power += power6
            
            # HTF8 power
            if ha8_bull:
                power8, strong8 = calculate_candle_power(
                    ha8_last['high'], ha8_last['low'], ha8_last['open'], ha8_last['close'],
                    htf8_df['volume'].iloc[-1], htf8_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong8:
                    cumulative_bull_power += power8
            
            # HTF12 power
            if ha12_bull:
                power12, strong12 = calculate_candle_power(
                    ha12_last['high'], ha12_last['low'], ha12_last['open'], ha12_last['close'],
                    htf12_df['volume'].iloc[-1], htf12_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong12:
                    cumulative_bull_power += power12
            
            # Power multiplier (4/4 ise 2x, 3/4 ise 1.5x)
            if bull_count >= 4:
                cumulative_bull_power *= 2.0
            elif bull_count >= 3:
                cumulative_bull_power *= 1.5
        
        if ultra_strong_sell:
            # HTF4 power
            if ha4_bear:
                power4, strong4 = calculate_candle_power(
                    ha4_last['high'], ha4_last['low'], ha4_last['open'], ha4_last['close'],
                    htf4_df['volume'].iloc[-1], htf4_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong4:
                    cumulative_bear_power += power4
            
            # HTF6 power
            if ha6_bear:
                power6, strong6 = calculate_candle_power(
                    ha6_last['high'], ha6_last['low'], ha6_last['open'], ha6_last['close'],
                    htf6_df['volume'].iloc[-1], htf6_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong6:
                    cumulative_bear_power += power6
            
            # HTF8 power
            if ha8_bear:
                power8, strong8 = calculate_candle_power(
                    ha8_last['high'], ha8_last['low'], ha8_last['open'], ha8_last['close'],
                    htf8_df['volume'].iloc[-1], htf8_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong8:
                    cumulative_bear_power += power8
            
            # HTF12 power
            if ha12_bear:
                power12, strong12 = calculate_candle_power(
                    ha12_last['high'], ha12_last['low'], ha12_last['open'], ha12_last['close'],
                    htf12_df['volume'].iloc[-1], htf12_vol_ma.iloc[-1],
                    min_candle_change, use_volume_in_power
                )
                if strong12:
                    cumulative_bear_power += power12
            
            # Power multiplier
            if bear_count >= 4:
                cumulative_bear_power *= 2.0
            elif bear_count >= 3:
                cumulative_bear_power *= 1.5
        
        # 9. Total power
        total_power = cumulative_bull_power if ultra_strong_buy else cumulative_bear_power if ultra_strong_sell else 0.0
        
        # 10. Whale Volume Detection (Daily)
        whale_active = False
        try:
            # Base TF'den günlük TF çarpanını hesapla
            time_diff_minutes = (df.index[1] - df.index[0]).total_seconds() / 60
            daily_multiplier = int(1440 / time_diff_minutes) if time_diff_minutes > 0 else 1440
            
            # Daily timeframe oluştur
            daily_df = get_htf_data(df, daily_multiplier)
            
            if len(daily_df) >= 50:
                daily_vol = daily_df['volume'].iloc[-1]
                daily_vol_ma50 = daily_df['volume'].rolling(window=50).mean().iloc[-1]
                
                # Whale spike multiplier (2.5x - Pine Script'teki gibi)
                volume_spike = daily_vol > (daily_vol_ma50 * 2.5)
                
                # Heikin Ashi daily
                ha_daily = calculate_heikin_ashi(daily_df)
                ha_d_last = ha_daily.iloc[-1]
                
                whale_buy = volume_spike and (ha_d_last['close'] > ha_d_last['open'])
                whale_sell = volume_spike and (ha_d_last['close'] < ha_d_last['open'])
                
                whale_active = whale_buy or whale_sell
        except Exception as e:
            logger.debug(f"Whale detection hatası: {e}")
        
        # 11. Sonuç döndür
        return {
            'ultra_strong_buy': ultra_strong_buy,
            'ultra_strong_sell': ultra_strong_sell,
            'bull_count': bull_count,
            'bear_count': bear_count,
            'total_power': float(total_power),
            'whale_active': whale_active,
            'htf_data': {
                'htf4_bull': ha4_bull,
                'htf4_bear': ha4_bear,
                'htf6_bull': ha6_bull,
                'htf6_bear': ha6_bear,
                'htf8_bull': ha8_bull,
                'htf8_bear': ha8_bear,
                'htf12_bull': ha12_bull,
                'htf12_bear': ha12_bear,
            }
        }
        
    except Exception as e:
        logger.error(f"Ultra signal detection hatası: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'ultra_strong_buy': False,
            'ultra_strong_sell': False,
            'bull_count': 0,
            'bear_count': 0,
            'total_power': 0.0,
            'whale_active': False,
            'htf_data': {}
        }


# ============================================
# COMPUTE METRICS (ANA FONKSİYON)
# ============================================

def compute_ultra_metrics(
    df: pd.DataFrame, 
    symbol: str = None,
    tfmult8: int = 8,
    tfmult12: int = 12,
    tfmult16: int = 16,
    tfmult24: int = 24
) -> Dict:
    """
    🔥 Ultra Panel v5 - Ana metrik hesaplama fonksiyonu
    
    Args:
        df: Base timeframe OHLCV verileri
        symbol: Sembol adı (opsiyonel)
        tfmult8-24: HTF çarpanları
        
    Returns:
        Dict: Ultra metrics (analyzer.py için uyumlu format)
    """
    if df is None or df.empty:
        return {}
    
    try:
        # Ultra sinyalleri tespit et
        ultra_data = detect_ultra_signals(
            df, tfmult8, tfmult12, tfmult16, tfmult24
        )
        
        # Sinyal tipi belirle
        if ultra_data['ultra_strong_buy']:
            run_type = 'long'
        elif ultra_data['ultra_strong_sell']:
            run_type = 'short'
        else:
            run_type = 'none'
        
        # HTF count (analyzer.py için run_count yerine)
        htf_count = max(ultra_data['bull_count'], ultra_data['bear_count'])
        
        # Sonuç döndür (analyzer.py formatında)
        return {
            # Ultra Panel verileri
            'ultra_strong_buy': ultra_data['ultra_strong_buy'],
            'ultra_strong_sell': ultra_data['ultra_strong_sell'],
            'bull_count': ultra_data['bull_count'],
            'bear_count': ultra_data['bear_count'],
            'htf_count': htf_count,  # 🔥 YENİ: HTF sayısı (3/4 veya 4/4)
            'total_power': ultra_data['total_power'],
            'whale_active': ultra_data['whale_active'],
            
            # Analyzer.py uyumluluğu için
            'run_type': run_type,
            'trigger_type': 'Ultra Signal' if run_type != 'none' else 'Yok',
            
            # HTF detay verileri
            'htf_data': ultra_data['htf_data']
        }
        
    except Exception as e:
        logger.error(f"Ultra metrics hesaplama hatası: {e}")
        return {}


# ============================================
# GERİYE UYUMLULUK FONKSİYONLARI
# ============================================

def compute_vpmv_metrics(df: pd.DataFrame, symbol: str = None) -> Dict:
    """
    🔄 GERİYE UYUMLULUK: compute_ultra_metrics()'e yönlendir
    Analyzer.py'de compute_vpmv_metrics() çağrısı varsa çalışır
    """
    return compute_ultra_metrics(df, symbol)


def compute_consecutive_metrics(df: pd.DataFrame) -> Dict:
    """
    🔄 GERİYE UYUMLULUK: compute_ultra_metrics()'e yönlendir
    """
    return compute_ultra_metrics(df)


def calculate_trigger(ultra_components: Dict) -> str:
    """
    🔄 GERİYE UYUMLULUK: Tetikleyici sistemi
    Ultra Panel'de trigger_type 'Ultra Signal' olarak dönüyor
    """
    if ultra_components.get('ultra_strong_buy') or ultra_components.get('ultra_strong_sell'):
        return 'Ultra Signal'
    return 'Yok'


# ============================================
# DEPRECATED FONKSIYONLAR (SİLİNEN SİSTEM)
# ============================================

def calculate_zigzag_high(*args, **kwargs):
    """DEPRECATED - Eski Deviso sistemi"""
    return 0.0

def calculate_zigzag_low(*args, **kwargs):
    """DEPRECATED - Eski Deviso sistemi"""
    return 0.0

def calculate_deviso_ratio(*args, **kwargs):
    """DEPRECATED - Eski Deviso sistemi"""
    return 0.0

def calculate_atr(*args, **kwargs):
    """DEPRECATED - VPMV SuperTrend sistemi"""
    return pd.Series([0.0])

def calculate_supertrend(*args, **kwargs):
    """DEPRECATED - VPMV SuperTrend sistemi"""
    return pd.Series([1]), pd.Series([False]), pd.Series([False])

def calculate_vpmv_components(*args, **kwargs):
    """DEPRECATED - VPMV sistemi"""
    return {}

def tanh_approximation(*args, **kwargs):
    """DEPRECATED - VPMV sistemi"""
    return 0.0