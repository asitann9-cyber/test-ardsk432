"""
Analiz servisleri
Supertrend analizi, C-Signal ve VPMV hesaplama
🆕 YENİ: VPMV (Volume-Price-Momentum-Volatility) NET POWER sistemi + TETİKLEYİCİ
🆕 YENİ: MULTI-TIMEFRAME TIME SİSTEMİ (1H, 2H, 4H, 6H, 8H, 12H)
✅ SuperTrend Reset mekanizması ile kümülatif hesaplamalar
✅ Pine Script %100 UYUMLU - Wilder's Smoothing ATR
✅ Tetikleyici Sistemi Eklendi
🔥 CRITICAL FIX: Tetikleyici mantığı Pine Script ile %100 uyumlu hale getirildi
🚀 PERFORMANCE FIX: Timestamp JSON serialization düzeltildi
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available, using manual calculations")

from .binance_service import BinanceService

logger = logging.getLogger(__name__)

class AnalysisService:
    """Supertrend, C-Signal ve VPMV analizi için service sınıfı"""
    
    # Supertrend parametreleri
    SUPERTREND_PARAMS = {
        'atr_period': 10,
        'multiplier': 3.0,
        'z_score_length': 14,
        'use_z_score': True,
        'momentum_rsi_period': 14,
        'top_symbols_count': 20,
    }
    
    # 🆕 Dinamik minimum ratio threshold - Panel'den ayarlanabilir
    MIN_RATIO_THRESHOLD = 100.0  # Varsayılan değer
    
    # 🆕 TIME SYSTEM - Multi-Timeframe Periods (Binance API formatı)
    TIME_PERIODS = ['1h', '2h', '4h', '6h', '8h', '12h']  # 1H-12H
    TIME_PERIOD_LABELS = {
        '1h': '1H',
        '2h': '2H', 
        '4h': '4H',
        '6h': '6H',
        '8h': '8H',
        '12h': '12H'
    }
    
    # =====================================================
    # 🆕 MULTI-TIMEFRAME TIME SYSTEM
    # =====================================================
    
    @staticmethod
    def calculate_mtf_time_signals(symbol: str) -> Dict[str, Any]:
        """
        Multi-Timeframe TIME sinyallerini hesapla (Pine Script mantığı)
        
        ✅ FIX: Binance API formatı ile uyumlu hale getirildi
        
        Pine Script Logic:
        - dir = close > open ? 1 : close < open ? -1 : 0
        - sig = dir == 1 ? "Long 🟢" : dir == -1 ? "Short 🔴" : "⚪ Nötr"
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Dict[str, Any]: {
                'time_signals': {
                    '1H': {'direction': 1, 'signal': 'Long 🟢'},
                    '2H': {'direction': -1, 'signal': 'Short 🔴'},
                    ...
                },
                'match_count': int  # VPMV yönü ile kaç periyot eşleşiyor
            }
        """
        try:
            time_signals = {}
            
            # ✅ Her periyot için son kapanış/açılış verilerini al
            for binance_timeframe, label in AnalysisService.TIME_PERIOD_LABELS.items():
                try:
                    # ✅ Binance'den o periyotun verisini çek (DOĞRU FORMAT: '1h', '2h', vb.)
                    df = BinanceService.fetch_klines_data(symbol, binance_timeframe, limit=2)
                    
                    if df is None or len(df) < 1:
                        time_signals[label] = {
                            'direction': 0,
                            'signal': '⚪ Nötr',
                            'close': None,
                            'open': None
                        }
                        continue
                    
                    # Son mum verilerini al
                    last_candle = df.iloc[-1]
                    close_price = float(last_candle['close'])
                    open_price = float(last_candle['open'])
                    
                    # Direction hesapla (Pine Script mantığı)
                    if close_price > open_price:
                        direction = 1
                        signal = 'Long 🟢'
                    elif close_price < open_price:
                        direction = -1
                        signal = 'Short 🔴'
                    else:
                        direction = 0
                        signal = '⚪ Nötr'
                    
                    time_signals[label] = {
                        'direction': direction,
                        'signal': signal,
                        'close': close_price,
                        'open': open_price
                    }
                    
                    # ✅ DEBUG LOG
                    logger.debug(f"🕐 TIME {label} ({binance_timeframe}): {signal} | C={close_price:.4f}, O={open_price:.4f}")
                    
                except Exception as e:
                    logger.error(f"TIME signal hesaplama hatası ({symbol} - {label}): {e}")
                    time_signals[label] = {
                        'direction': 0,
                        'signal': '⚪ Nötr',
                        'close': None,
                        'open': None
                    }
            
            return {
                'time_signals': time_signals,
                'calculation_time': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"MTF TIME signals hatası ({symbol}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Hata durumunda boş sinyaller döndür
            empty_signals = {
                label: {
                    'direction': 0,
                    'signal': '⚪ Nötr',
                    'close': None,
                    'open': None
                } 
                for label in AnalysisService.TIME_PERIOD_LABELS.values()
            }
            return {
                'time_signals': empty_signals,
                'calculation_time': datetime.now().strftime('%H:%M:%S')
            }
    
    @staticmethod
    def calculate_time_match_count(time_signals: Dict[str, Any], vpmv_direction: int) -> int:
        """
        TIME sinyalleri ile VPMV yönünün kaç periyotta eşleştiğini hesapla
        
        Args:
            time_signals (Dict[str, Any]): TIME sinyalleri
            vpmv_direction (int): VPMV yönü (1=Long, -1=Short, 0=Neutral)
            
        Returns:
            int: Eşleşen periyot sayısı (0-6 arası)
        """
        try:
            if not time_signals or vpmv_direction == 0:
                return 0
            
            match_count = 0
            signals = time_signals.get('time_signals', {})
            
            for label in AnalysisService.TIME_PERIOD_LABELS.values():
                signal_data = signals.get(label, {})
                direction = signal_data.get('direction', 0)
                
                # VPMV yönü ile eşleşiyorsa ve nötr değilse
                if direction != 0 and direction == vpmv_direction:
                    match_count += 1
            
            return match_count
            
        except Exception as e:
            logger.debug(f"TIME match count hatası: {e}")
            return 0
    
    # =====================================================
    # 🆕 VPMV (VOLUME-PRICE-MOMENTUM-VOLATILITY) SİSTEMİ
    # =====================================================
    
    @staticmethod
    def detect_supertrend_reset(trend: pd.Series) -> np.ndarray:
        """
        SuperTrend yön değişimini tespit et (Pine: ta.change(st_direction) != 0)
        
        Args:
            trend (pd.Series): SuperTrend trend değerleri (1=Bullish, -1=Bearish)
            
        Returns:
            np.ndarray: Reset sinyalleri (True/False)
        """
        try:
            # Trend değişimlerini tespit et
            trend_change = trend.diff() != 0
            trend_change.iloc[0] = True  # İlk bar reset kabul edilir
            
            return trend_change.values
            
        except Exception as e:
            logger.debug(f"SuperTrend reset tespit hatası: {e}")
            return np.zeros(len(trend), dtype=bool)
    
    @staticmethod
    def calculate_vpmv_net_power(df: pd.DataFrame, trend: pd.Series, reset_signals: np.ndarray) -> Dict[str, Any]:
        """
        🎯 Pine Script VPMV - 4 Bileşenli Sistem + TETİKLEYİCİ (100% UYUMLU)
        
        ✅ DÜZELTMELER:
        1. Wilder's Smoothing ATR (Pine Script mantığı)
        2. Reset anında avg_volume hesaplama (ta.sma(vol, 20))
        3. ✅ Reset bar'ından sonra hesaplama (Pine: not just_reset)
        4. 🔥 CRITICAL FIX: Tetikleyici sistemi Pine Script ile TAM UYUMLU
        
        Pine Script Tetikleyici Mantığı:
        ═══════════════════════════════════════════════════════════════
        triggerName = "Yok"
        if priceTrig
            maxVal = 0.0
            if momentumTrig
                maxVal := math.abs(momentum_component)
                triggerName := "Momentum"
            if volumeTrig and math.abs(vol_component) > maxVal
                maxVal := math.abs(vol_component)
                triggerName := "Hacim"
            if volatilityTrig and math.abs(volatility_component) > maxVal
                triggerName := "Volatilite"
        ═══════════════════════════════════════════════════════════════
        ⚠️ ÖNEMLİ: Son if bloğunda DA maxVal karşılaştırması VAR!

        Returns:
            Dict[str, Any]: {
                'net_power': float,
                'vol_component': float,
                'momentum_component': float,
                'price_component': float,
                'volatility_component': float,
                'trigger_name': str,
                'trigger_active': bool,
                'price_triggered': bool,
                'momentum_triggered': bool,
                'volume_triggered': bool,
                'volatility_triggered': bool
            }
        """
        try:
            if df is None or len(df) < 20:
                return {
                    'net_power': 0.0,
                    'vol_component': 0.0,
                    'momentum_component': 0.0,
                    'price_component': 0.0,
                    'volatility_component': 0.0,
                    'trigger_name': 'Yok',
                    'trigger_active': False,
                    'price_triggered': False,
                    'momentum_triggered': False,
                    'volume_triggered': False,
                    'volatility_triggered': False
                }

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            open_price = df['open'].values

            # Kümülatif değişkenler
            cumulative_volume = 0.0
            cumulative_momentum = 0.0
            signal_price = None
            avg_volume = None

            # Son reset noktasını bul
            last_reset_idx = 0
            for i in range(len(reset_signals) - 1, -1, -1):
                if reset_signals[i]:
                    last_reset_idx = i
                    signal_price = close[i]
                    # ✅ Reset anında avg_volume hesapla (Pine: ta.sma(vol, 20))
                    start_idx = max(0, i - 19)
                    avg_volume = np.mean(volume[start_idx:i+1])
                    break

            if signal_price is None:
                signal_price = close[0]
                last_reset_idx = 0
                avg_volume = np.mean(volume[:20]) if len(volume) >= 20 else np.mean(volume)

            # ✅ Wilder's Smoothing ATR (Pine Script mantığı)
            atr_period = 10
            atr_wilder = None
            
            for i in range(1, len(close)):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                
                if atr_wilder is None:
                    # İlk ATR - Basit ortalama ile başla
                    if i >= atr_period:
                        tr_values = []
                        for j in range(max(1, i - atr_period + 1), i + 1):
                            tr_j = max(
                                high[j] - low[j],
                                abs(high[j] - close[j-1]),
                                abs(low[j] - close[j-1])
                            )
                            tr_values.append(tr_j)
                        atr_wilder = np.mean(tr_values)
                else:
                    # Wilder's Smoothing: ATR = (ATR[1] * (period - 1) + TR) / period
                    atr_wilder = (atr_wilder * (atr_period - 1) + tr) / atr_period
            
            atr = atr_wilder if atr_wilder is not None else 0.01

            # ✅ Reset bar'ından sonra hesaplama (Pine: not just_reset)
            # Pine'da: if not na(vpmv_signal_price) and not just_reset
            # Reset olan bar'da hesaplama yapılmaz, bir sonraki bar'dan başlar
            for i in range(last_reset_idx + 1, len(close)):
                # Volume Component
                buy_volume = volume[i] if close[i] > open_price[i] else 0.0
                sell_volume = volume[i] if close[i] < open_price[i] else 0.0
                net_volume = buy_volume - sell_volume
                cumulative_volume += net_volume

                # Momentum Component
                if i > 0:
                    momentum_change = ((close[i] - close[i-1]) / close[i-1]) * 100
                    cumulative_momentum += momentum_change

            # 1. Volume Component
            vol_ratio = cumulative_volume / (avg_volume * 5 + 0.0001)
            vol_component = np.tanh(vol_ratio) * 100

            # 2. Momentum Component  
            momentum_component = np.tanh(cumulative_momentum / 10) * 100

            # 3. Price Component
            price_change_from_signal = ((close[-1] - signal_price) / signal_price) * 100
            price_component = np.tanh(price_change_from_signal / 10) * 100

            # 4. Volatility Component
            volatility_pct = (atr / close[-1]) * 100
            volatility_component = np.tanh(volatility_pct / 5) * 100

            # NET POWER (Ağırlıklı Toplam)
            net_power = (price_component * 0.7) + \
                        (vol_component * 0.1) + \
                        (momentum_component * 0.1) + \
                        (volatility_component * 0.1)

            # 🔥 CRITICAL FIX: TETİKLEYİCİ SİSTEMİ - Pine Script EXACT Mantığı
            trigger_name = "Yok"
            trigger_active = False
            
            # Tetikleyici eşikleri (Pine Script ile aynı)
            price_trig = abs(price_component) >= 50
            momentum_trig = abs(momentum_component) >= 25
            volume_trig = abs(vol_component) >= 25
            volatility_trig = abs(volatility_component) >= 25
            
            # ⚠️ KRİTİK: Pine Script mantığı - EXACT IMPLEMENTATION
            if price_trig:
                trigger_active = True
                max_val = 0.0
                
                # İLK: Momentum kontrolü
                if momentum_trig:
                    max_val = abs(momentum_component)
                    trigger_name = "Momentum"
                
                # SONRA: Volume kontrolü VE karşılaştırma
                if volume_trig and abs(vol_component) > max_val:
                    max_val = abs(vol_component)
                    trigger_name = "Hacim"
                
                # 🔥 FIX: SON if'te DE maxVal karşılaştırması VAR!
                if volatility_trig and abs(volatility_component) > max_val:
                    trigger_name = "Volatilite"

            return {
                'net_power': round(net_power, 2),
                'vol_component': round(vol_component, 2),
                'momentum_component': round(momentum_component, 2),
                'price_component': round(price_component, 2),
                'volatility_component': round(volatility_component, 2),
                # ✅ Tetikleyici bilgileri
                'trigger_name': trigger_name,
                'trigger_active': trigger_active,
                'price_triggered': price_trig,
                'momentum_triggered': momentum_trig,
                'volume_triggered': volume_trig,
                'volatility_triggered': volatility_trig
            }

        except Exception as e:
            logger.error(f"VPMV hesaplama hatası: {e}")
            return {
                'net_power': 0.0,
                'vol_component': 0.0,
                'momentum_component': 0.0,
                'price_component': 0.0,
                'volatility_component': 0.0,
                'trigger_name': 'Yok',
                'trigger_active': False,
                'price_triggered': False,
                'momentum_triggered': False,
                'volume_triggered': False,
                'volatility_triggered': False
            }
    
    @staticmethod
    def get_vpmv_signal(net_power: float) -> str:
        """
        VPMV NET POWER'a göre sinyal belirle
        
        Args:
            net_power (float): NET POWER değeri
            
        Returns:
            str: Sinyal (STRONG LONG, LONG, SHORT, STRONG SHORT, NEUTRAL)
        """
        if net_power > 10:
            return "STRONG LONG"
        elif net_power > 0:
            return "LONG"
        elif net_power < -10:
            return "STRONG SHORT"
        elif net_power < 0:
            return "SHORT"
        else:
            return "NEUTRAL"
        
    # =====================================================
    # 🆕 İKİ AŞAMALI ANALİZ FONKSİYONLARI
    # =====================================================

    @staticmethod
    def analyze_symbol_basic(symbol: str, timeframe: str = '15m') -> Optional[Dict[str, Any]]:
        """
        ⚡ HIZLI ANALİZ - Sadece Supertrend + Ratio (ANA TABLO için)
        
        🎯 AMAÇ: 535 sembol için hızlı tarama
        📊 HESAPLANAN: Supertrend, Ratio, Z-Score
        ❌ HESAPLANMAYAN: VPMV, C-Signal, Tetikleyici, TIME
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            Optional[Dict[str, Any]]: Temel analiz sonuçları
        """
        try:
            # ✅ Binance'den veri çek (1 API çağrısı)
            df = BinanceService.fetch_klines_data(symbol, timeframe, limit=500)
            if df is None or len(df) < 50:
                return None
            
            # ✅ Supertrend hesapla
            supertrend, trend, upper_band, lower_band = AnalysisService.calculate_supertrend_pine_script(
                df, 
                AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                AnalysisService.SUPERTREND_PARAMS['multiplier']
            )
            
            # ✅ Pullback seviyesi ve ratio hesapla
            pullback_level = AnalysisService.calculate_pullback_levels_pine_script(df, supertrend, trend)
            ratio_percent = AnalysisService.calculate_ratio_percentage_pine_script(df, pullback_level)
            z_score = AnalysisService.calculate_z_score_pine_script(
                ratio_percent, 
                AnalysisService.SUPERTREND_PARAMS['z_score_length']
            )
            
            # ✅ Final ratio
            if AnalysisService.SUPERTREND_PARAMS['use_z_score']:
                final_ratio = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
            else:
                final_ratio = ratio_percent.iloc[-1] if not pd.isna(ratio_percent.iloc[-1]) else 0
            
            # ✅ Son değerler
            current_price = float(df['close'].iloc[-1])
            current_supertrend = supertrend.iloc[-1]
            current_trend = trend.iloc[-1]
            current_ratio_percent = ratio_percent.iloc[-1] if not pd.isna(ratio_percent.iloc[-1]) else 0
            current_z_score = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
            
            trend_direction = 'Bullish' if current_trend > 0 else 'Bearish'
            price_vs_supertrend = 'Above' if current_price > current_supertrend else 'Below'
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'supertrend': current_supertrend,
                'trend_direction': trend_direction,
                'price_vs_supertrend': price_vs_supertrend,
                'ratio_percent': round(current_ratio_percent, 2),
                'z_score': round(current_z_score, 2),
                'final_ratio': round(final_ratio, 2),
                'last_update': df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in df.columns else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 🔥 FIX: Timestamp → String
                
                # ❌ VPMV YOK - WebSocket'te hesaplanacak
                # ❌ C-Signal YOK
                # ❌ Tetikleyici YOK
                # ❌ TIME YOK
            }
            
        except Exception as e:
            logger.debug(f"⚡ Basic analiz hatası {symbol}: {e}")
            return None


    @staticmethod
    def analyze_symbol_full(symbol: str, timeframe: str = '15m') -> Optional[Dict[str, Any]]:
        """
        🐢 TAM DETAYLI ANALİZ - VPMV + C-Signal + Tetikleyici + TIME (KALICI TABLO için)
        
        🎯 AMAÇ: Kalıcı listedeki sembolleri WebSocket'te detaylı analiz
        📊 HESAPLANAN: Supertrend + VPMV + C-Signal + Tetikleyici + TIME
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            Optional[Dict[str, Any]]: Tam analiz sonuçları
        """
        try:
            # ✅ Binance'den veri çek (1 API çağrısı)
            df = BinanceService.fetch_klines_data(symbol, timeframe, limit=500)
            if df is None or len(df) < 50:
                return None
            
            # ✅ Supertrend hesapla
            supertrend, trend, upper_band, lower_band = AnalysisService.calculate_supertrend_pine_script(
                df, 
                AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                AnalysisService.SUPERTREND_PARAMS['multiplier']
            )
            
            # ✅ Pullback seviyesi ve ratio hesapla
            pullback_level = AnalysisService.calculate_pullback_levels_pine_script(df, supertrend, trend)
            ratio_percent = AnalysisService.calculate_ratio_percentage_pine_script(df, pullback_level)
            z_score = AnalysisService.calculate_z_score_pine_script(
                ratio_percent, 
                AnalysisService.SUPERTREND_PARAMS['z_score_length']
            )
            
            # ✅ C-Signal hesapla
            c_signal = AnalysisService.calculate_c_signal(df)
            
            # ✅ VPMV NET POWER hesapla (4 bileşenli + Tetikleyici)
            reset_signals = AnalysisService.detect_supertrend_reset(trend)
            vpmv_result = AnalysisService.calculate_vpmv_net_power(df, trend, reset_signals)
            
            vpmv_net_power = vpmv_result['net_power']
            vpmv_signal = AnalysisService.get_vpmv_signal(vpmv_net_power)
            
            # ✅ TIME sinyallerini hesapla (6 API çağrısı)
            time_result = AnalysisService.calculate_mtf_time_signals(symbol)
            time_signals = time_result['time_signals']
            
            # VPMV yönü ile TIME eşleşmesi
            vpmv_dir = 1 if vpmv_net_power >= 0 else -1
            time_match_count = AnalysisService.calculate_time_match_count(time_result, vpmv_dir)
            
            # ✅ C sinyali momentum
            change_momentum, bars_ago, change_timestamp = AnalysisService.get_latest_supertrend_change_momentum(
                df, timeframe
            )
            
            # ✅ Final ratio
            if AnalysisService.SUPERTREND_PARAMS['use_z_score']:
                final_ratio = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
            else:
                final_ratio = ratio_percent.iloc[-1] if not pd.isna(ratio_percent.iloc[-1]) else 0
            
            # ✅ Son değerler
            current_price = float(df['close'].iloc[-1])
            current_supertrend = supertrend.iloc[-1]
            current_trend = trend.iloc[-1]
            current_ratio_percent = ratio_percent.iloc[-1] if not pd.isna(ratio_percent.iloc[-1]) else 0
            current_z_score = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
            
            trend_direction = 'Bullish' if current_trend > 0 else 'Bearish'
            price_vs_supertrend = 'Above' if current_price > current_supertrend else 'Below'
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'supertrend': current_supertrend,
                'trend_direction': trend_direction,
                'price_vs_supertrend': price_vs_supertrend,
                'ratio_percent': round(current_ratio_percent, 2),
                'z_score': round(current_z_score, 2),
                'final_ratio': round(final_ratio, 2),
                'change_momentum': change_momentum,
                'momentum_bars_ago': bars_ago,
                'change_timestamp': change_timestamp,
                
                # ✅ C-Signal
                'c_signal': c_signal,
                
                # ✅ VPMV NET POWER ve Signal
                'vpmv_net_power': vpmv_net_power,
                'vpmv_signal': vpmv_signal,
                
                # ✅ VPMV 4 Bileşen
                'vpmv_vol_component': vpmv_result['vol_component'],
                'vpmv_momentum_component': vpmv_result['momentum_component'],
                'vpmv_price_component': vpmv_result['price_component'],
                'vpmv_volatility_component': vpmv_result['volatility_component'],
                
                # ✅ Tetikleyici bilgileri
                'vpmv_trigger_name': vpmv_result['trigger_name'],
                'vpmv_trigger_active': vpmv_result['trigger_active'],
                'vpmv_price_triggered': vpmv_result['price_triggered'],
                'vpmv_momentum_triggered': vpmv_result['momentum_triggered'],
                'vpmv_volume_triggered': vpmv_result['volume_triggered'],
                'vpmv_volatility_triggered': vpmv_result['volatility_triggered'],
                
                # ✅ TIME SYSTEM bilgileri
                'time_signals': time_signals,
                'time_match_count': time_match_count,
                'time_calculation_time': time_result['calculation_time'],
                
                'last_update': df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in df.columns else datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 🔥 FIX: Timestamp → String
            }
            
        except Exception as e:
            logger.debug(f"🐢 Full analiz hatası {symbol}: {e}")
            return None


    @staticmethod
    def analyze_single_symbol(symbol: str, timeframe: str = '4h') -> Optional[Dict[str, Any]]:
        """
        ⚠️ DEPRECATED: Geriye uyumluluk için analyze_symbol_full() çağırıyor
        Yeni kod analyze_symbol_full() veya analyze_symbol_basic() kullanmalı!
        """
        logger.warning(f"⚠️ analyze_single_symbol() deprecated! analyze_symbol_full() veya analyze_symbol_basic() kullanın")
        return AnalysisService.analyze_symbol_full(symbol, timeframe)
    
    # =====================================================
    # C-SIGNAL HESAPLAMA
    # =====================================================
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        """
        RSI hesaplama fonksiyonu (Wilder's smoothing method)
        
        Args:
            prices (np.ndarray): Fiyat dizisi
            period (int): RSI periyodu
            
        Returns:
            Optional[float]: RSI değeri
        """
        if len(prices) < period + 1:
            return None
        
        try:
            # İlk değişimleri hesapla
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # İlk ortalama değerler
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Wilder's smoothing ile RSI hesaplama
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
            
        except Exception as e:
            logger.debug(f"RSI hesaplama hatası: {e}")
            return None
    
    @staticmethod
    def calculate_c_signal(df: pd.DataFrame) -> Optional[float]:
        """
        C-Signal hesaplama - RSI(log(close), 14) değişimi
        ℹ️ NOT: Bu fonksiyon sadece C-Signal DEĞER hesaplar, threshold kontrolü yapmaz!
        Threshold kontrolü memory_storage.py'de yapılır.
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            
        Returns:
            Optional[float]: C-Signal değeri
        """
        try:
            if df is None or len(df) < 16:  # RSI için 14 + değişim için 2
                return None
            
            # Log close hesapla
            log_close = np.log(df['close'].values)
            
            if len(log_close) < 16:
                return None
            
            # Son iki RSI değerini hesapla
            current_rsi = AnalysisService.calculate_rsi(log_close)
            previous_rsi = AnalysisService.calculate_rsi(log_close[:-1])
            
            if current_rsi is None or previous_rsi is None:
                return None
            
            # C-Signal = RSI değişimi
            c_signal = current_rsi - previous_rsi
            return round(c_signal, 2)
            
        except Exception as e:
            logger.debug(f"C-Signal hesaplama hatası: {e}")
            return None

    # =====================================================
    # SUPERTREND HESAPLAMA
    # =====================================================

    @staticmethod
    def calculate_atr_manual(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """ATR hesaplama - TA-Lib destekli"""
        try:
            if df is None or len(df) < period:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            # TA-Lib varsa onu kullan (daha güvenilir)
            if TALIB_AVAILABLE:
                try:
                    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
                    return pd.Series(atr, index=df.index).bfill().fillna(0)
                except:
                    pass
            
            # Manuel hesaplama
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            manual_atr = pd.Series(index=df.index, dtype=float)
            for i in range(len(df)):
                if i < period:
                    manual_atr.iloc[i] = np.nan
                elif i == period:
                    manual_atr.iloc[i] = true_range.iloc[:i+1].mean()
                else:
                    manual_atr.iloc[i] = (manual_atr.iloc[i-1] * (period - 1) + true_range.iloc[i]) / period
            
            return manual_atr.fillna(0)
        except Exception as e:
            logger.debug(f"ATR hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_supertrend_pine_script(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Supertrend hesaplama - Standart Pine Script mantığı"""
        try:
            if df is None or len(df) < atr_period:
                default_series = pd.Series(index=df.index if df is not None else [], dtype=float)
                return default_series, default_series, default_series, default_series
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            atr = AnalysisService.calculate_atr_manual(df, atr_period)
            
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = pd.Series(index=df.index, dtype=float)
            trend = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                if i == 0:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = 1
                else:
                    if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                        upper_band.iloc[i] = upper_band.iloc[i]
                    else:
                        upper_band.iloc[i] = upper_band.iloc[i-1]
                    
                    if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                        lower_band.iloc[i] = lower_band.iloc[i]
                    else:
                        lower_band.iloc[i] = lower_band.iloc[i-1]
                    
                    if trend.iloc[i-1] == 1 and close.iloc[i] <= lower_band.iloc[i]:
                        trend.iloc[i] = -1
                    elif trend.iloc[i-1] == -1 and close.iloc[i] >= upper_band.iloc[i]:
                        trend.iloc[i] = 1
                    else:
                        trend.iloc[i] = trend.iloc[i-1]
                    
                    if trend.iloc[i] == 1:
                        supertrend.iloc[i] = lower_band.iloc[i]
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
            
            return supertrend.fillna(0), trend.fillna(0), upper_band.fillna(0), lower_band.fillna(0)
            
        except Exception as e:
            logger.debug(f"Supertrend hesaplama hatası: {e}")
            default_series = pd.Series(index=df.index if df is not None else [], dtype=float)
            return default_series, default_series, default_series, default_series

    @staticmethod
    def calculate_pullback_levels_pine_script(df: pd.DataFrame, supertrend: pd.Series, trend: pd.Series) -> pd.Series:
        """
        Doğru Supertrend pullback seviyesi hesaplama
        Trend başlangıcından itibaren en düşük/yüksek seviyeler
        """
        try:
            if df is None or len(df) < 5:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            pullback_level = pd.Series(index=df.index, dtype=float)
            pullback_level.iloc[0] = close.iloc[0]
            
            # İlk değeri trend'e göre ayarla
            if trend.iloc[0] == 1:  # Bullish başlangıç
                bullish_pullback = low.iloc[0]
                bearish_pullback = high.iloc[0]
            else:  # Bearish başlangıç
                bullish_pullback = low.iloc[0]
                bearish_pullback = high.iloc[0]
            
            for i in range(1, len(df)):
                current_trend = trend.iloc[i]
                previous_trend = trend.iloc[i-1]
                
                if current_trend != previous_trend:
                    if current_trend == 1:  # Bullish'e geçiş
                        bullish_pullback = low.iloc[i]
                    elif current_trend == -1:  # Bearish'e geçiş
                        bearish_pullback = high.iloc[i]
                
                # Pullback seviyesini sabitle
                if current_trend == 1:  # Bullish trend
                    pullback_level.iloc[i] = bullish_pullback
                elif current_trend == -1:  # Bearish trend
                    pullback_level.iloc[i] = bearish_pullback
                else:
                    pullback_level.iloc[i] = pullback_level.iloc[i-1]
            
            return pullback_level.ffill()
            
        except Exception as e:
            logger.debug(f"Pullback hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_ratio_percentage_pine_script(df: pd.DataFrame, pullback_level: pd.Series) -> pd.Series:
        """
        Doğru ratio yüzde hesaplama (Bearish trend için özel mantık)
        """
        try:
            if df is None or len(df) < 5:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            close = df['close']
            ratio_percent = pd.Series(index=df.index, dtype=float)
            
            # Trend bilgisine de ihtiyacımız var
            _, trend, _, _ = AnalysisService.calculate_supertrend_pine_script(df, 10, 3.0)
            
            for i in range(len(df)):
                pullback = pullback_level.iloc[i]
                current_trend = trend.iloc[i] if i < len(trend) else 1
                
                # Güvenlik kontrolleri
                if pd.isna(pullback) or pullback <= 0:
                    ratio_percent.iloc[i] = 0.0
                elif abs(pullback) < 1e-8:
                    ratio_percent.iloc[i] = 0.0
                else:
                    if current_trend == 1:  # Bullish trend
                        ratio_percent.iloc[i] = ((close.iloc[i] - pullback) / pullback) * 100
                    else:  # Bearish trend (-1)
                        ratio_percent.iloc[i] = ((pullback - close.iloc[i]) / pullback) * 100
            
            return ratio_percent
            
        except Exception as e:
            logger.debug(f"Ratio yüzde hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_z_score_pine_script(series: pd.Series, length: int = 14) -> pd.Series:
        """Z-Score hesaplama"""
        try:
            if series is None or len(series) < length:
                return pd.Series(index=series.index if series is not None else [], dtype=float)
            
            rolling_mean = series.rolling(window=length).mean()
            rolling_std = series.rolling(window=length).std()
            
            z_score = pd.Series(index=series.index, dtype=float)
            for i in range(len(series)):
                if pd.isna(rolling_std.iloc[i]) or rolling_std.iloc[i] == 0:
                    z_score.iloc[i] = 0.0
                else:
                    z_score.iloc[i] = (series.iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
            
            return z_score.fillna(0)
        except Exception as e:
            logger.debug(f"Z-Score hesaplama hatası: {e}")
            return pd.Series(index=series.index if series is not None else [], dtype=float)

    @staticmethod
    def get_latest_supertrend_change_momentum(df: pd.DataFrame, timeframe: str = '4h') -> Tuple[float, int, Optional[datetime]]:
        """En son supertrend yön değişimindeki C sinyali momentum"""
        try:
            if df is None or len(df) < 20:
                return np.nan, 0, None
            
            supertrend, trend, upper_band, lower_band = AnalysisService.calculate_supertrend_pine_script(
                df, 
                AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                AnalysisService.SUPERTREND_PARAMS['multiplier']
            )
            
            # TA-Lib ile RSI momentum hesapla
            if TALIB_AVAILABLE:
                try:
                    log_close = np.log(df['close'].values)
                    rsi_values = talib.RSI(log_close, timeperiod=AnalysisService.SUPERTREND_PARAMS['momentum_rsi_period'])
                    rsi_momentum = pd.Series(rsi_values).diff()
                except:
                    rsi_momentum = pd.Series(index=df.index, dtype=float).fillna(0)
            else:
                rsi_momentum = pd.Series(index=df.index, dtype=float).fillna(0)
            
            # Trend değişimlerini tespit et
            direction_changes = []
            for i in range(1, len(trend)):
                current_trend = trend.iloc[i]
                previous_trend = trend.iloc[i-1]
                
                if current_trend != previous_trend and pd.notna(current_trend) and pd.notna(previous_trend):
                    direction_changes.append({
                        'index': i,
                        'from_trend': previous_trend,
                        'to_trend': current_trend,
                        'change_type': 'Bullish' if current_trend > previous_trend else 'Bearish'
                    })
            
            if not direction_changes:
                return np.nan, 0, None
            
            latest_change = direction_changes[-1]
            change_index = latest_change['index']
            
            if change_index < len(rsi_momentum):
                c_signal_value = rsi_momentum.iloc[change_index]
            else:
                c_signal_value = np.nan
            
            bars_ago = len(df) - 1 - change_index
            change_timestamp = df.iloc[change_index]['timestamp'] if 'timestamp' in df.columns else None
            
            return c_signal_value, bars_ago, change_timestamp
            
        except Exception as e:
            logger.debug(f"En son yön değişimi C sinyali hatası: {e}")
            return np.nan, 0, None

    # =====================================================
    # ANA ANALİZ FONKSİYONLARI
    # =====================================================

    
    @staticmethod
    def analyze_multiple_symbols(symbols: List[str], timeframe: str = '4h', max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        ⚡ HIZLI ÇOK SEMBOL ANALİZİ - Sadece Supertrend + Ratio
        
        🎯 AMAÇ: Ana tablo için 535 sembolu hızlı tarama
        📊 KULLANIM: analyze_symbol_basic() ile paralel işleme
        
        Args:
            symbols (List[str]): Sembol listesi
            timeframe (str): Zaman dilimi
            max_workers (int): Maksimum worker sayısı
            
        Returns:
            List[Dict[str, Any]]: Temel analiz sonuçları (VPMV/TIME YOK!)
        """
        try:
            logger.info(f"⚡ {len(symbols)} sembol için {timeframe} HIZLI analiz (Sadece Ratio) başlatılıyor...")
            
            results = []
            
            # ✅ Paralel işleme - analyze_symbol_basic() kullan
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(AnalysisService.analyze_symbol_basic, symbol, timeframe) 
                    for symbol in symbols
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.debug(f"Parallel analiz hatası: {e}")
            
            # ✅ Sıralama
            if AnalysisService.SUPERTREND_PARAMS['use_z_score']:
                results.sort(key=lambda x: abs(x.get('z_score', 0)), reverse=True)
            else:
                results.sort(key=lambda x: abs(x.get('ratio_percent', 0)), reverse=True)
            
            logger.info(f"✅ {len(results)} sembol HIZLI analiz edildi (VPMV/C-Signal/TIME dahil DEĞİL - WebSocket'te hesaplanacak)")
            return results
            
        except Exception as e:
            logger.error(f"❌ Çoklu analiz hatası: {e}")
            return []
    
    @staticmethod
    def create_tradingview_link(symbol: str, timeframe: str) -> str:
        """
        TradingView grafik linki oluştur
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            str: TradingView URL
        """
        try:
            tv_timeframe_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '2h': '120', '4h': '240', '1d': '1D'
            }
            
            tv_timeframe = tv_timeframe_map.get(timeframe, '240')
            base_url = "https://tr.tradingview.com/chart/"
            
            # Binance perpetual futures için sembol formatı
            if symbol.endswith('USDT'):
                tv_symbol = f"{symbol}.P"
            else:
                tv_symbol = symbol
                
            chart_url = f"{base_url}?symbol=BINANCE%3A{tv_symbol}&interval={tv_timeframe}"
            return chart_url
            
        except Exception as e:
            logger.debug(f"TradingView link oluşturma hatası: {e}")
            return "#"
    
    @staticmethod
    def get_analysis_summary(results: List[Dict[str, Any]], timeframe: str) -> Dict[str, Any]:
        """
        Analiz özetini hazırla (VPMV + Tetikleyici + TIME dahil)
        
        Args:
            results (List[Dict[str, Any]]): Analiz sonuçları
            timeframe (str): Zaman dilimi
            
        Returns:
            Dict[str, Any]: Analiz özeti
        """
        try:
            if not results:
                return {
                    'total_symbols': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'high_ratio_count': 0,
                    'max_ratio': 0,
                    'vpmv_strong_long_count': 0,
                    'vpmv_strong_short_count': 0,
                    'trigger_active_count': 0,
                    'avg_time_match_count': 0,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            bullish_count = sum(1 for r in results if r.get('trend_direction') == 'Bullish')
            bearish_count = sum(1 for r in results if r.get('trend_direction') == 'Bearish')
            high_ratio_count = sum(1 for r in results if abs(r.get('ratio_percent', 0)) >= AnalysisService.MIN_RATIO_THRESHOLD)
            max_ratio = max((abs(r.get('ratio_percent', 0)) for r in results), default=0)
            
            # 🆕 VPMV istatistikleri
            vpmv_strong_long_count = sum(1 for r in results if r.get('vpmv_signal') == 'STRONG LONG')
            vpmv_strong_short_count = sum(1 for r in results if r.get('vpmv_signal') == 'STRONG SHORT')
            
            # ✅ YENİ: Tetikleyici istatistikleri
            trigger_active_count = sum(1 for r in results if r.get('vpmv_trigger_active', False))
            
            # 🆕 TIME istatistikleri
            time_matches = [r.get('time_match_count', 0) for r in results]
            avg_time_match_count = sum(time_matches) / len(time_matches) if time_matches else 0
            
            return {
                'total_symbols': len(results),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'high_ratio_count': high_ratio_count,
                'max_ratio': max_ratio,
                'vpmv_strong_long_count': vpmv_strong_long_count,
                'vpmv_strong_short_count': vpmv_strong_short_count,
                'trigger_active_count': trigger_active_count,
                'avg_time_match_count': round(avg_time_match_count, 1),
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Analiz özeti hatası: {e}")
            return {
                'total_symbols': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'high_ratio_count': 0,
                'max_ratio': 0,
                'vpmv_strong_long_count': 0,
                'vpmv_strong_short_count': 0,
                'trigger_active_count': 0,
                'avg_time_match_count': 0,
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    @staticmethod
    def update_symbol_with_analysis(symbol_data: Dict[str, Any], timeframe: str, preserve_manual_type: bool = False) -> Dict[str, Any]:
        """
        Manuel tür korunarak sembol verisini güncelle (VPMV + C-Signal + Tetikleyici + TIME dahil)
        """
        try:
            symbol = symbol_data['symbol']
            
            # Binance'den güncel veri çek
            df = BinanceService.fetch_klines_data(symbol, timeframe, limit=500)
            if df is not None and len(df) >= 50:
                
                # 🔥 C-Signal hesapla (Her durumda!)
                c_signal = AnalysisService.calculate_c_signal(df)
                symbol_data['c_signal'] = c_signal
                symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
                
                # 🆕 TIME sinyallerini hesapla
                time_result = AnalysisService.calculate_mtf_time_signals(symbol)
                symbol_data['time_signals'] = time_result['time_signals']
                symbol_data['time_calculation_time'] = time_result['calculation_time']
                
                # Manuel tür koruma kontrolü
                if preserve_manual_type:
                    # ✅ Sadece VPMV + Tetikleyici + TIME güncelle
                    supertrend, trend, _, _ = AnalysisService.calculate_supertrend_pine_script(
                        df, 
                        AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                        AnalysisService.SUPERTREND_PARAMS['multiplier']
                    )
                    reset_signals = AnalysisService.detect_supertrend_reset(trend)
                    vpmv_result = AnalysisService.calculate_vpmv_net_power(df, trend, reset_signals)
                    
                    symbol_data['vpmv_net_power'] = vpmv_result['net_power']
                    symbol_data['vpmv_signal'] = AnalysisService.get_vpmv_signal(vpmv_result['net_power'])
                    symbol_data['vpmv_trigger_name'] = vpmv_result['trigger_name']
                    symbol_data['vpmv_trigger_active'] = vpmv_result['trigger_active']
                    
                    # TIME match count hesapla
                    vpmv_dir = 1 if vpmv_result['net_power'] >= 0 else -1
                    symbol_data['time_match_count'] = AnalysisService.calculate_time_match_count(time_result, vpmv_dir)
                    
                    logger.info(f"🔒 {symbol} MANUEL TÜR KORUNDU - VPMV + C-Signal + Tetikleyici + TIME güncellendi")
                    
                else:
                    # ✅ TAM ANALİZ - analyze_symbol_full() kullan!
                    full_analysis = AnalysisService.analyze_symbol_full(symbol, timeframe)  # ✅ DEĞİŞİKLİK BURADA
                    
                    if full_analysis:
                        symbol_data['ratio_percent'] = full_analysis.get('ratio_percent', 0)
                        symbol_data['supertrend_type'] = full_analysis.get('trend_direction', 'None')
                        symbol_data['z_score'] = full_analysis.get('z_score', 0)
                        symbol_data['vpmv_net_power'] = full_analysis.get('vpmv_net_power', 0)
                        symbol_data['vpmv_signal'] = full_analysis.get('vpmv_signal', 'NEUTRAL')
                        symbol_data['vpmv_trigger_name'] = full_analysis.get('vpmv_trigger_name', 'Yok')
                        symbol_data['vpmv_trigger_active'] = full_analysis.get('vpmv_trigger_active', False)
                        symbol_data['time_match_count'] = full_analysis.get('time_match_count', 0)
                        
                        logger.info(f"🔄 {symbol} TAM GÜNCELLEME YAPILDI")
                
                # 🔥 LOG - C-Signal değeri
                logger.info(f"💡 C-Signal güncellendi: {symbol} = {c_signal}")
                
            else:
                # Veri yetersiz
                logger.warning(f"⚠️ {symbol} için veri yetersiz, default değerler atanıyor")
                symbol_data['c_signal'] = None
                symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
                symbol_data['vpmv_net_power'] = 0
                symbol_data['vpmv_signal'] = 'NEUTRAL'
                symbol_data['vpmv_trigger_name'] = 'Yok'
                symbol_data['vpmv_trigger_active'] = False
                symbol_data['time_signals'] = {}
                symbol_data['time_match_count'] = 0
            
            return symbol_data
            
        except Exception as e:
            logger.error(f"❌ Analiz güncelleme hatası {symbol_data.get('symbol', 'UNKNOWN')}: {e}")
            symbol_data['c_signal'] = None
            symbol_data['vpmv_net_power'] = 0
            symbol_data['vpmv_signal'] = 'NEUTRAL'
            symbol_data['vpmv_trigger_name'] = 'Yok'
            symbol_data['vpmv_trigger_active'] = False
            symbol_data['time_signals'] = {}
            symbol_data['time_match_count'] = 0
            return symbol_data

    
    @staticmethod
    def filter_results(results: List[Dict[str, Any]], filter_type: str) -> List[Dict[str, Any]]:
        """
        Analiz sonuçları filtrele
        
        Args:
            results (List[Dict[str, Any]]): Tüm sonuçlar
            filter_type (str): Filtre tipi (all, bullish, bearish, high-ratio)
            
        Returns:
            List[Dict[str, Any]]: Filtrelenmiş sonuçlar
        """
        if not results:
            return []
        
        filtered_results = []
        
        if filter_type == 'bullish':
            filtered_results = [r for r in results if r.get('trend_direction') == 'Bullish']
        elif filter_type == 'bearish':
            filtered_results = [r for r in results if r.get('trend_direction') == 'Bearish']
        elif filter_type == 'high-ratio':
            filtered_results = [r for r in results if abs(r.get('ratio_percent', 0)) >= AnalysisService.MIN_RATIO_THRESHOLD]
        else:  # 'all'
            filtered_results = results
        
        # Ratio %'ye göre tekrar sırala
        filtered_results.sort(key=lambda x: abs(x.get('ratio_percent', 0)), reverse=True)
        
        # Rank'ı güncelle
        for i, result in enumerate(filtered_results):
            result['filtered_rank'] = i + 1
        
        return filtered_results
    
    @staticmethod
    def is_high_priority_symbol(result: Dict[str, Any]) -> bool:
        """
        Yüksek öncelikli sembol mu kontrol et
        🆕 Dinamik threshold kullanımı - Panel'den ayarlanabilir
        
        Args:
            result (Dict[str, Any]): Analiz sonucu
            
        Returns:
            bool: Yüksek öncelikli ise True
        """
        ratio_percent = abs(result.get('ratio_percent', 0))
        
        # Dinamik threshold kullan
        return ratio_percent >= AnalysisService.MIN_RATIO_THRESHOLD