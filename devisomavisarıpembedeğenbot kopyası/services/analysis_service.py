"""
Analiz servisleri
Supertrend analizi, C-Signal hesaplama ve ters momentum tespiti
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
    """Supertrend ve ters momentum analizi için service sınıfı"""
    
    # Supertrend parametreleri - ÖRNEK KODDAN ALINMIŞ
    SUPERTREND_PARAMS = {
        'atr_period': 10,
        'multiplier': 3.0,
        'z_score_length': 14,
        'use_z_score': True,
        'momentum_rsi_period': 14,
        'top_symbols_count': 20,
    }
    
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

    @staticmethod
    def calculate_atr_manual(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """ÖRNEK KODDAN ALINMIŞ ATR hesaplama - TA-Lib destekli"""
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
            
            # Manuel hesaplama (örnek koddan alınmış mantık)
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
        """ÖRNEK KODDAN ALINMIŞ Supertrend hesaplama - Standart Pine Script mantığı"""
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
                    # Örnek koddan alınmış mantık
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
        """ÖRNEK KODDAN ALINMIŞ - Basitleştirilmiş pullback seviyesi hesaplama"""
        try:
            if df is None or len(df) < 10:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            high = df['high']
            low = df['low']
            close = df['close']
            
            pullback_level = pd.Series(index=df.index, dtype=float)
            
            # İlk değeri close olarak ata
            pullback_level.iloc[0] = close.iloc[0]
            
            for i in range(1, len(df)):
                # Trend değişimi tespit et
                trend_changed = trend.iloc[i] != trend.iloc[i-1]
                
                if trend_changed:
                    if trend.iloc[i] == 1:  # Bullish'e geçiş
                        # Son düşük seviyeyi pullback olarak al
                        lookback_period = min(10, i)
                        pullback_level.iloc[i] = low.iloc[i-lookback_period:i+1].min()
                    elif trend.iloc[i] == -1:  # Bearish'e geçiş  
                        # Son yüksek seviyeyi pullback olarak al
                        lookback_period = min(10, i)
                        pullback_level.iloc[i] = high.iloc[i-lookback_period:i+1].max()
                    else:
                        # Önceki seviyeyi koru
                        pullback_level.iloc[i] = pullback_level.iloc[i-1]
                else:
                    # Trend devam ediyor, önceki seviyeyi koru
                    pullback_level.iloc[i] = pullback_level.iloc[i-1]
            
            return pullback_level.ffill()

            
        except Exception as e:
            logger.debug(f"Pullback seviyesi hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_ratio_percentage_pine_script(df: pd.DataFrame, pullback_level: pd.Series) -> pd.Series:
        """ÖRNEK KODDAN ALINMIŞ - Güvenli ratio yüzde hesaplama"""
        try:
            if df is None or len(df) < 5:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            close = df['close']
            ratio_percent = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                pullback = pullback_level.iloc[i]
                
                # Güvenlik kontrolleri - ÖRNEK KODDAN ALINMIŞ
                if pd.isna(pullback) or pullback <= 0:
                    ratio_percent.iloc[i] = 0.0
                else:
                    # Sıfıra bölme koruması
                    if abs(pullback) < 1e-8:
                        ratio_percent.iloc[i] = 0.0
                    else:
                        ratio_percent.iloc[i] = ((close.iloc[i] - pullback) / pullback) * 100
            
            return ratio_percent
            
        except Exception as e:
            logger.debug(f"Ratio yüzde hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_z_score_pine_script(series: pd.Series, length: int = 14) -> pd.Series:
        """ÖRNEK KODDAN ALINMIŞ Z-Score hesaplama"""
        try:
            if series is None or len(series) < length:
                return pd.Series(index=series.index if series is not None else [], dtype=float)
            
            rolling_mean = series.rolling(window=length).mean()
            rolling_std = series.rolling(window=length).std()
            
            z_score = pd.Series(index=series.index, dtype=float)
            for i in range(len(series)):
                if pd.isna(rolling_std.iloc[i]) or rolling_std.iloc[i] == 0:
                    z_score.iloc[i] = 0.0  # NaN yerine 0
                else:
                    z_score.iloc[i] = (series.iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
            
            return z_score.fillna(0)
        except Exception as e:
            logger.debug(f"Z-Score hesaplama hatası: {e}")
            return pd.Series(index=series.index if series is not None else [], dtype=float)

    @staticmethod
    def get_latest_supertrend_change_momentum(df: pd.DataFrame, timeframe: str = '4h') -> Tuple[float, int, Optional[datetime]]:
        """ÖRNEK KODDAN ALINMIŞ - En son supertrend yön değişimindeki C sinyali momentum"""
        try:
            if df is None or len(df) < 20:
                return np.nan, 0, None
            
            supertrend, trend, upper_band, lower_band = AnalysisService.calculate_supertrend_pine_script(
                df, 
                AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                AnalysisService.SUPERTREND_PARAMS['multiplier']
            )
            
            # TA-Lib ile RSI momentum hesapla (örnek koddan)
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
    
    @staticmethod
    def detect_reverse_momentum(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ters momentum tespiti:
        - Supertrend trend = Long + C-Signal <= -10 = Ters momentum (C↓)
        - Supertrend trend = Short + C-Signal >= +10 = Ters momentum (C↑)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            
        Returns:
            Dict[str, Any]: Ters momentum analiz sonuçları
        """
        try:
            # Supertrend sisteminde trend_direction kullan
            trend_direction = symbol_data.get('trend_direction', 'None')
            c_signal = symbol_data.get('c_signal', None)
            
            if c_signal is None or trend_direction == 'None':
                return {
                    'has_reverse_momentum': False,
                    'reverse_type': 'None',
                    'signal_strength': 'None',
                    'alert_message': '',
                    'signal_value': None
                }
            
            c_signal_value = float(c_signal)
            
            # --- TERS MOMENTUM TESPİTİ (C10 ve üzeri) ---
            if trend_direction == 'Bullish' and c_signal_value <= -10:
                strength = 'Strong' if abs(c_signal_value) >= 20 else 'Medium'
                return {
                    'has_reverse_momentum': True,
                    'reverse_type': 'C↓',  # Bullish'den Bearish'e dönüş sinyali
                    'signal_strength': strength,
                    'signal_value': c_signal_value,
                    'alert_message': f'TERS MOMENTUM: Bullish trend + C-Signal negatif ({c_signal_value})'
                }
            elif trend_direction == 'Bearish' and c_signal_value >= 10:
                strength = 'Strong' if c_signal_value >= 20 else 'Medium'
                return {
                    'has_reverse_momentum': True,
                    'reverse_type': 'C↑',  # Bearish'den Bullish'e dönüş sinyali
                    'signal_strength': strength,
                    'signal_value': c_signal_value,
                    'alert_message': f'TERS MOMENTUM: Bearish trend + C-Signal pozitif ({c_signal_value})'
                }
            else:
                # Normal momentum (ters momentum yok)
                return {
                    'has_reverse_momentum': False,
                    'reverse_type': 'Normal',
                    'signal_strength': 'None',
                    'alert_message': '',
                    'signal_value': c_signal_value
                }
                
        except Exception as e:
            logger.debug(f"Ters momentum hesaplama hatası: {e}")
            return {
                'has_reverse_momentum': False,
                'reverse_type': 'Error',
                'signal_strength': 'None',
                'alert_message': '',
                'signal_value': None
            }

    @staticmethod
    def analyze_single_symbol(symbol: str, timeframe: str = '4h') -> Optional[Dict[str, Any]]:
        """
        ÖRNEK KODDAN ALINMIŞ - Tek sembol için Supertrend analizi
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            Optional[Dict[str, Any]]: Analiz sonuçları
        """
        try:
            # Binance'den veri çek (ÖRNEK KODDA 500 MUM)
            df = BinanceService.fetch_klines_data(symbol, timeframe, limit=500)
            if df is None or len(df) < 50:
                return None
            
            # Supertrend hesapla
            supertrend, trend, upper_band, lower_band = AnalysisService.calculate_supertrend_pine_script(
                df, 
                AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                AnalysisService.SUPERTREND_PARAMS['multiplier']
            )
            
            # Pullback seviyesi ve ratio % hesapla
            pullback_level = AnalysisService.calculate_pullback_levels_pine_script(df, supertrend, trend)
            ratio_percent = AnalysisService.calculate_ratio_percentage_pine_script(df, pullback_level)
            z_score = AnalysisService.calculate_z_score_pine_script(ratio_percent, AnalysisService.SUPERTREND_PARAMS['z_score_length'])
            
            # ÖRNEK KODDAN - C sinyali momentum
            change_momentum, bars_ago, change_timestamp = AnalysisService.get_latest_supertrend_change_momentum(df, timeframe)
            
            # Final ratio hesaplama - ÖRNEK KODDAN ALINMIŞ MANTIK
            if AnalysisService.SUPERTREND_PARAMS['use_z_score']:
                final_ratio = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
            else:
                final_ratio = ratio_percent.iloc[-1] if not pd.isna(ratio_percent.iloc[-1]) else 0
            
            # Son değerler
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
                'final_ratio': round(final_ratio, 2),  # ÖRNEK KODDAN
                'change_momentum': change_momentum,  # ÖRNEK KODDAN
                'momentum_bars_ago': bars_ago,  # ÖRNEK KODDAN
                'change_timestamp': change_timestamp,  # ÖRNEK KODDAN
                'last_update': df['timestamp'].iloc[-1]
            }
            
        except Exception as e:
            logger.debug(f"Tek sembol Supertrend analiz hatası {symbol}: {e}")
            return None
    
    @staticmethod
    def analyze_multiple_symbols(symbols: List[str], timeframe: str = '4h', max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        ÖRNEK KODDAN ALINMIŞ - Çoklu sembol Supertrend analizi - Paralel işleme
        
        Args:
            symbols (List[str]): Sembol listesi
            timeframe (str): Zaman dilimi
            max_workers (int): Maksimum worker sayısı
            
        Returns:
            List[Dict[str, Any]]: Analiz sonuçları listesi
        """
        try:
            logger.info(f"{len(symbols)} sembol için {timeframe} Supertrend analizi başlatılıyor...")
            
            results = []
            
            # Paralel işleme
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(AnalysisService.analyze_single_symbol, symbol, timeframe) 
                    for symbol in symbols
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.debug(f"Parallel Supertrend analiz hatası: {e}")
            
            # ÖRNEK KODDAN ALINMIŞ SIRALAMA MANTIGI
            if AnalysisService.SUPERTREND_PARAMS['use_z_score']:
                # Z-Score'a göre sırala
                results.sort(key=lambda x: abs(x.get('z_score', 0)), reverse=True)
            else:
                # Ratio %'ye göre sırala (mutlak değerine göre en yüksek üstte)
                results.sort(key=lambda x: abs(x.get('ratio_percent', 0)), reverse=True)
            
            logger.info(f"✅ {len(results)} sembol Supertrend analizi edildi")
            return results
            
        except Exception as e:
            logger.error(f"❌ Çoklu Supertrend analiz hatası: {e}")
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
        Supertrend analiz özetini hazırla
        
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
                    'timeframe': timeframe,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            bullish_count = sum(1 for r in results if r.get('trend_direction') == 'Bullish')
            bearish_count = sum(1 for r in results if r.get('trend_direction') == 'Bearish')
            high_ratio_count = sum(1 for r in results if abs(r.get('ratio_percent', 0)) >= 100)
            max_ratio = max((abs(r.get('ratio_percent', 0)) for r in results), default=0)
            
            return {
                'total_symbols': len(results),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'high_ratio_count': high_ratio_count,
                'max_ratio': max_ratio,
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Supertrend analiz özeti hatası: {e}")
            return {
                'total_symbols': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'high_ratio_count': 0,
                'max_ratio': 0,
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    @staticmethod
    def update_symbol_with_c_signal(symbol_data: Dict[str, Any], timeframe: str, preserve_manual_type: bool = False) -> Dict[str, Any]:
        """
        🔒 MANUEL TÜR KORUNARAK Sembol verisini C-Signal ile güncelle (Supertrend için)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            timeframe (str): Zaman dilimi
            preserve_manual_type (bool): Manuel değiştirilen türü koruma (True = koru, False = güncelle)
            
        Returns:
            Dict[str, Any]: Güncellenmiş sembol verileri
        """
        try:
            symbol = symbol_data['symbol']
            
            # Binance'den güncel veri çek
            df = BinanceService.fetch_klines_data(symbol, timeframe, limit=500)  # ÖRNEK KODDA 500
            if df is not None and len(df) >= 50:
                # C-Signal hesapla
                c_signal = AnalysisService.calculate_c_signal(df)
                symbol_data['c_signal'] = c_signal
                symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
                
                # 🔒 MANUEL TÜR KORUMA KONTROLÜ
                if preserve_manual_type:
                    # Manuel tür koruması aktif - sadece C-Signal güncelle
                    logger.debug(f"🔒 {symbol} manuel tür koruması: {symbol_data.get('max_supertrend_type', 'Unknown')} - TÜR KORUNUYOR")
                else:
                    # Normal güncelleme - Supertrend analizi
                    supertrend_analysis = AnalysisService.analyze_single_symbol(symbol, timeframe)
                    
                    if supertrend_analysis:
                        # Ratio rekoru kontrolü
                        current_ratio = abs(supertrend_analysis.get('ratio_percent', 0))
                        max_ratio = symbol_data.get('max_ratio_percent', 0)
                        
                        if current_ratio > max_ratio:
                            # Yeni rekor - tür ve ratio güncelle
                            symbol_data['max_ratio_percent'] = supertrend_analysis.get('ratio_percent', 0)
                            symbol_data['max_supertrend_type'] = supertrend_analysis.get('trend_direction', 'None')
                            symbol_data['max_z_score'] = supertrend_analysis.get('z_score', 0)
                            logger.debug(f"🔄 {symbol} normal güncelleme: {supertrend_analysis.get('trend_direction', 'None')} - TÜR GÜNCELLENDİ")
                        else:
                            logger.debug(f"🔄 {symbol} normal güncelleme: Rekor kırılmadı, tür korundu")
                
                # Ters momentum tespit et (mevcut türle)
                reverse_momentum = AnalysisService.detect_reverse_momentum(symbol_data)
                symbol_data['reverse_momentum'] = reverse_momentum
                
                logger.debug(f"C-Signal güncellendi: {symbol} = {c_signal} (Preserve: {preserve_manual_type})")
            else:
                # Veri yetersiz
                symbol_data['c_signal'] = None
                symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
                symbol_data['reverse_momentum'] = {
                    'has_reverse_momentum': False,
                    'reverse_type': 'None',
                    'signal_strength': 'None',
                    'alert_message': '',
                    'signal_value': None
                }
            
            return symbol_data
            
        except Exception as e:
            logger.debug(f"C-Signal güncelleme hatası {symbol_data.get('symbol', 'UNKNOWN')}: {e}")
            symbol_data['c_signal'] = None
            symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
            symbol_data['reverse_momentum'] = {
                'has_reverse_momentum': False,
                'reverse_type': 'Error',
                'signal_strength': 'None',
                'alert_message': '',
                'signal_value': None
            }
            return symbol_data
    
    @staticmethod
    def filter_results(results: List[Dict[str, Any]], filter_type: str) -> List[Dict[str, Any]]:
        """
        Supertrend sonuçları filtrele
        
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
            filtered_results = [r for r in results if abs(r.get('ratio_percent', 0)) >= 100]
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
        Supertrend ratio >= 100%
        
        Args:
            result (Dict[str, Any]): Analiz sonucu
            
        Returns:
            bool: Yüksek öncelikli ise True
        """
        ratio_percent = abs(result.get('ratio_percent', 0))
        
        return ratio_percent >= 100.0