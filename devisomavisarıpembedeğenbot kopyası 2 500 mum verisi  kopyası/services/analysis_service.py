"""
Analiz servisleri
Supertrend analizi, C-Signal hesaplama ve ters momentum tespiti
Örnek koddaki daha doğru hesaplama yöntemleri ile optimize edilmiş
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .binance_service import BinanceService

logger = logging.getLogger(__name__)

class AnalysisService:
    """Supertrend ve ters momentum analizi için service sınıfı"""
    
    # Supertrend parametreleri - Örnek koddan alınan değerler
    SUPERTREND_PARAMS = {
        'atr_period': 10,
        'multiplier': 3.0,
        'z_score_length': 14,
        'momentum_rsi_period': 14,
        'minimum_data_length': 100,  # Örnek kodda 500 mum çekiliyor
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
        """
        Manuel ATR hesaplama - Örnek koddan optimize edilmiş
        """
        try:
            if df is None or len(df) < period:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
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
        """
        Supertrend hesaplama - Örnek koddan birebir alınan algorithm
        """
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
        Pullback seviyesi hesaplama - ÖRNEK KODDAN BİREBİR ALINMIŞ
        Bu fonksiyon daha doğru sonuçlar veriyor
        """
        try:
            if df is None or len(df) < 10:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            high = df['high']
            low = df['low']
            
            long_pullback_level = pd.Series(index=df.index, dtype=float)
            is_after_short = pd.Series(index=df.index, dtype=bool, default=False)
            
            # Örnek koddan birebir alınan pullback logic
            for i in range(1, len(df)):
                buy_signal = (trend.iloc[i] == 1 and trend.iloc[i-1] == -1)
                sell_signal = (trend.iloc[i] == -1 and trend.iloc[i-1] == 1)
                
                # Önceki değeri koru
                if i > 0:
                    long_pullback_level.iloc[i] = long_pullback_level.iloc[i-1]
                    is_after_short.iloc[i] = is_after_short.iloc[i-1]
                
                # Sell signal durumu
                if sell_signal:
                    short_pullback_level = high.iloc[i]
                    long_pullback_level.iloc[i] = short_pullback_level
                    is_after_short.iloc[i] = True
                
                # Buy signal durumu
                if buy_signal:
                    if pd.isna(long_pullback_level.iloc[i]):
                        long_pullback_level.iloc[i] = low.iloc[i]
                    elif is_after_short.iloc[i]:
                        long_pullback_level.iloc[i] = low.iloc[i]
                        is_after_short.iloc[i] = False
            
            # Forward fill missing values
            return long_pullback_level.ffill()
            
        except Exception as e:
            logger.debug(f"Pullback seviyesi hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_ratio_percentage_pine_script(df: pd.DataFrame, long_pullback_level: pd.Series) -> pd.Series:
        """
        Ratio yüzde hesaplama - ÖRNEK KODDAN BİREBİR ALINMIŞ
        Bu fonksiyon daha doğru sonuçlar veriyor
        """
        try:
            if df is None or len(df) < 5:
                return pd.Series(index=df.index if df is not None else [], dtype=float)
            
            close = df['close']
            diff_percent = pd.Series(index=df.index, dtype=float)
            
            # Örnek koddan birebir alınan ratio hesaplama
            for i in range(len(df)):
                if pd.isna(long_pullback_level.iloc[i]) or long_pullback_level.iloc[i] == 0:
                    diff_percent.iloc[i] = np.nan
                else:
                    diff_percent.iloc[i] = ((close.iloc[i] - long_pullback_level.iloc[i]) / long_pullback_level.iloc[i]) * 100
            
            return diff_percent
            
        except Exception as e:
            logger.debug(f"Ratio yüzde hesaplama hatası: {e}")
            return pd.Series(index=df.index if df is not None else [], dtype=float)

    @staticmethod
    def calculate_z_score_pine_script(series: pd.Series, length: int = 14) -> pd.Series:
        """
        Z-Score hesaplama - ÖRNEK KODDAN BİREBİR ALINMIŞ
        """
        try:
            if series is None or len(series) < length:
                return pd.Series(index=series.index if series is not None else [], dtype=float)
            
            rolling_mean = series.rolling(window=length).mean()
            rolling_std = series.rolling(window=length).std()
            
            z_score = pd.Series(index=series.index, dtype=float)
            
            # Örnek koddan birebir alınan z-score hesaplama
            for i in range(len(series)):
                if pd.isna(rolling_std.iloc[i]) or rolling_std.iloc[i] == 0:
                    z_score.iloc[i] = np.nan
                else:
                    z_score.iloc[i] = (series.iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
            
            return z_score.fillna(0)
        except Exception as e:
            logger.debug(f"Z-Score hesaplama hatası: {e}")
            return pd.Series(index=series.index if series is not None else [], dtype=float)
    
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
        Tek sembol için Supertrend analizi - ÖRNEK KODDAN OPTIMIZE EDİLMİŞ
        Daha fazla veri ile daha doğru hesaplama
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            Optional[Dict[str, Any]]: Analiz sonuçları
        """
        try:
            # ÖRNEK KODDAN ALINMIŞ: Daha fazla veri çek (500 yerine enhanced method)
            df = BinanceService.fetch_enhanced_klines_data(symbol, timeframe, time_range_days=30)
            if df is None or len(df) < AnalysisService.SUPERTREND_PARAMS['minimum_data_length']:
                logger.debug(f"Yetersiz veri: {symbol} - {len(df) if df is not None else 0} mum")
                return None
            
            # Veri kalitesi kontrolü
            data_quality = BinanceService.get_data_quality_info(df)
            if not data_quality.get('suitable_for_supertrend', False):
                logger.debug(f"Veri kalitesi yetersiz: {symbol} - Quality: {data_quality.get('data_quality', 'UNKNOWN')}")
                return None
            
            # ÖRNEK KODDAN ALINMIŞ: Aynı parametreler ile Supertrend hesapla
            supertrend, trend, upper_band, lower_band = AnalysisService.calculate_supertrend_pine_script(
                df, 
                AnalysisService.SUPERTREND_PARAMS['atr_period'], 
                AnalysisService.SUPERTREND_PARAMS['multiplier']
            )
            
            # ÖRNEK KODDAN ALINMIŞ: Pullback ve ratio hesaplama
            long_pullback_level = AnalysisService.calculate_pullback_levels_pine_script(df, supertrend, trend)
            ratio_percent = AnalysisService.calculate_ratio_percentage_pine_script(df, long_pullback_level)
            z_score = AnalysisService.calculate_z_score_pine_script(ratio_percent, AnalysisService.SUPERTREND_PARAMS['z_score_length'])
            
            # Son değerler
            current_price = float(df['close'].iloc[-1])
            current_supertrend = supertrend.iloc[-1]
            current_trend = trend.iloc[-1]
            current_ratio_percent = ratio_percent.iloc[-1] if not pd.isna(ratio_percent.iloc[-1]) else 0
            current_z_score = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
            
            trend_direction = 'Bullish' if current_trend > 0 else 'Bearish'
            price_vs_supertrend = 'Above' if current_price > current_supertrend else 'Below'
            
            logger.debug(f"✅ {symbol} analiz edildi: {trend_direction}, Ratio: {current_ratio_percent:.2f}%, Z-Score: {current_z_score:.2f}")
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'supertrend': current_supertrend,
                'trend_direction': trend_direction,
                'price_vs_supertrend': price_vs_supertrend,
                'ratio_percent': round(current_ratio_percent, 2),
                'z_score': round(current_z_score, 2),
                'last_update': df['timestamp'].iloc[-1],
                'data_quality': data_quality.get('data_quality', 'GOOD'),
                'candle_count': len(df)
            }
            
        except Exception as e:
            logger.debug(f"Tek sembol Supertrend analiz hatası {symbol}: {e}")
            return None
    
    @staticmethod
    def analyze_multiple_symbols(symbols: List[str], timeframe: str = '4h', max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Çoklu sembol Supertrend analizi - Paralel işleme
        ÖRNEK KODDAN OPTIMIZE EDİLMİŞ: Daha fazla veri ile daha doğru sonuçlar
        
        Args:
            symbols (List[str]): Sembol listesi
            timeframe (str): Zaman dilimi
            max_workers (int): Maksimum worker sayısı
            
        Returns:
            List[Dict[str, Any]]: Analiz sonuçları listesi
        """
        try:
            logger.info(f"🎯 {len(symbols)} sembol için {timeframe} Supertrend analizi başlatılıyor (Örnek kod algoritması)...")
            
            results = []
            processed_count = 0
            error_count = 0
            
            # Paralel işleme
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(AnalysisService.analyze_single_symbol, symbol, timeframe) 
                    for symbol in symbols
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        processed_count += 1
                        
                        if result:
                            results.append(result)
                            if len(results) % 50 == 0:  # Her 50 sonuçta log
                                logger.info(f"📊 {len(results)} sembol başarıyla analiz edildi...")
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        logger.debug(f"Parallel Supertrend analiz hatası: {e}")
                        error_count += 1
            
            # ÖRNEK KODDAN ALINMIŞ: Ratio %'ye göre sırala (mutlak değerine göre en yüksek üstte)
            results.sort(key=lambda x: abs(x.get('ratio_percent', 0)), reverse=True)
            
            # Kalite raporlama
            excellent_count = sum(1 for r in results if r.get('data_quality') == 'EXCELLENT')
            good_count = sum(1 for r in results if r.get('data_quality') == 'GOOD')
            
            logger.info(f"✅ Analiz tamamlandı: {len(results)} başarılı, {error_count} hata")
            logger.info(f"📈 Veri kalitesi: {excellent_count} mükemmel, {good_count} iyi")
            
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
        Supertrend analiz özetini hazırla - ÖRNEK KODDAN GENİŞLETİLMİŞ
        
        Args:
            results (List[Dict[str, Any]]): Analiz sonuçları
            timeframe (str): Zaman dilimi
            
        Returns:
            Dict[str, Any]: Detaylı analiz özeti
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
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_quality_summary': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
                }
            
            bullish_count = sum(1 for r in results if r.get('trend_direction') == 'Bullish')
            bearish_count = sum(1 for r in results if r.get('trend_direction') == 'Bearish')
            high_ratio_count = sum(1 for r in results if abs(r.get('ratio_percent', 0)) >= 100)
            max_ratio = max((abs(r.get('ratio_percent', 0)) for r in results), default=0)
            
            # Veri kalitesi özeti
            quality_summary = {
                'excellent': sum(1 for r in results if r.get('data_quality') == 'EXCELLENT'),
                'good': sum(1 for r in results if r.get('data_quality') == 'GOOD'),
                'fair': sum(1 for r in results if r.get('data_quality') == 'FAIR'),
                'poor': sum(1 for r in results if r.get('data_quality') == 'POOR')
            }
            
            # Z-Score dağılımı
            z_scores = [abs(r.get('z_score', 0)) for r in results]
            high_z_score_count = sum(1 for z in z_scores if z >= 2.0)
            
            # Ortalama mum sayısı
            avg_candle_count = sum(r.get('candle_count', 0) for r in results) / len(results)
            
            return {
                'total_symbols': len(results),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'high_ratio_count': high_ratio_count,
                'high_z_score_count': high_z_score_count,
                'max_ratio': round(max_ratio, 2),
                'avg_candle_count': round(avg_candle_count, 0),
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_quality_summary': quality_summary,
                'algorithm_source': 'Örnek kod optimizasyonu'
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
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_quality_summary': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
            }
    
    @staticmethod
    def update_symbol_with_c_signal(symbol_data: Dict[str, Any], timeframe: str, preserve_manual_type: bool = False) -> Dict[str, Any]:
        """
        🔒 MANUEL TÜR KORUNARAK Sembol verisini C-Signal ile güncelle
        ÖRNEK KODDAN OPTIMIZE EDİLMİŞ: Daha doğru hesaplama
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            timeframe (str): Zaman dilimi
            preserve_manual_type (bool): Manuel değiştirilen türü koruma
            
        Returns:
            Dict[str, Any]: Güncellenmiş sembol verileri
        """
        try:
            symbol = symbol_data['symbol']
            
            # ÖRNEK KODDAN ALINMIŞ: Daha fazla veri çek
            df = BinanceService.fetch_enhanced_klines_data(symbol, timeframe, time_range_days=15)
            if df is not None and len(df) >= AnalysisService.SUPERTREND_PARAMS['minimum_data_length']:
                # C-Signal hesapla
                c_signal = AnalysisService.calculate_c_signal(df)
                symbol_data['c_signal'] = c_signal
                symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
                
                # 🔒 MANUEL TÜR KORUMA KONTROLÜ
                if preserve_manual_type:
                    # Manuel tür koruması aktif - sadece C-Signal güncelle
                    logger.debug(f"🔒 {symbol} manuel tür koruması: {symbol_data.get('max_supertrend_type', 'Unknown')} - TÜR KORUNUYOR")
                else:
                    # Normal güncelleme - ÖRNEK KODDAN OPTIMIZE EDİLMİŞ Supertrend analizi
                    supertrend_analysis = AnalysisService.analyze_single_symbol(symbol, timeframe)
                    
                    if supertrend_analysis:
                        # Ratio rekoru kontrolü
                        current_ratio = abs(supertrend_analysis.get('ratio_percent', 0))
                        max_ratio = abs(symbol_data.get('max_ratio_percent', 0))
                        
                        if current_ratio > max_ratio:
                            # Yeni rekor - tür ve ratio güncelle
                            symbol_data['max_ratio_percent'] = supertrend_analysis.get('ratio_percent', 0)
                            symbol_data['max_supertrend_type'] = supertrend_analysis.get('trend_direction', 'None')
                            symbol_data['max_z_score'] = supertrend_analysis.get('z_score', 0)
                            logger.debug(f"🔄 {symbol} normal güncelleme: {supertrend_analysis.get('trend_direction', 'None')} - TÜR GÜNCELLENDİ (Yeni rekor)")
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
                logger.debug(f"Yetersiz veri: {symbol} - Veri: {len(df) if df is not None else 0}")
            
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
        
        # ÖRNEK KODDAN ALINMIŞ: Ratio %'ye göre tekrar sırala
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
    
    @staticmethod
    def validate_analysis_data(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        ÖRNEK KODDAN EKLENMİŞ: Analiz verisi doğrulama
        
        Args:
            df (pd.DataFrame): OHLCV verisi
            symbol (str): Sembol adı
            
        Returns:
            Dict[str, Any]: Doğrulama sonuçları
        """
        try:
            if df is None or len(df) == 0:
                return {
                    'is_valid': False,
                    'reason': 'No data available',
                    'recommendations': ['Check symbol validity', 'Verify API connection']
                }
            
            # Minimum veri kontrolü
            if len(df) < AnalysisService.SUPERTREND_PARAMS['minimum_data_length']:
                return {
                    'is_valid': False,
                    'reason': f'Insufficient data: {len(df)} candles (minimum: {AnalysisService.SUPERTREND_PARAMS["minimum_data_length"]})',
                    'recommendations': ['Increase time range', 'Use longer timeframe', 'Check if symbol is newly listed']
                }
            
            # Eksik veri kontrolü
            missing_data = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum()
            if missing_data > 0:
                return {
                    'is_valid': False,
                    'reason': f'Missing data points: {missing_data}',
                    'recommendations': ['Use different timeframe', 'Clean data source']
                }
            
            # Fiyat mantıklılık kontrolü
            price_anomalies = 0
            for i in range(len(df)):
                if df['high'].iloc[i] < df['low'].iloc[i]:
                    price_anomalies += 1
                elif df['close'].iloc[i] > df['high'].iloc[i] or df['close'].iloc[i] < df['low'].iloc[i]:
                    price_anomalies += 1
                elif df['open'].iloc[i] > df['high'].iloc[i] or df['open'].iloc[i] < df['low'].iloc[i]:
                    price_anomalies += 1
            
            anomaly_rate = price_anomalies / len(df) * 100
            if anomaly_rate > 5.0:  # %5'ten fazla anomali
                return {
                    'is_valid': False,
                    'reason': f'High price anomaly rate: {anomaly_rate:.1f}%',
                    'recommendations': ['Check data source quality', 'Use different timeframe']
                }
            
            # Başarılı doğrulama
            return {
                'is_valid': True,
                'reason': 'Data validation passed',
                'candle_count': len(df),
                'anomaly_rate': anomaly_rate,
                'recommendations': []
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'reason': f'Validation error: {str(e)}',
                'recommendations': ['Contact technical support']
            }
    
    @staticmethod
    def get_enhanced_symbol_info(symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        ÖRNEK KODDAN EKLENMİŞ: Sembol hakkında detaylı bilgi
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            Dict[str, Any]: Detaylı sembol bilgileri
        """
        try:
            # Temel analiz
            analysis = AnalysisService.analyze_single_symbol(symbol, timeframe)
            if not analysis:
                return {'available': False, 'reason': 'Analysis failed'}
            
            # Market bilgileri
            market_info = BinanceService.get_market_info(symbol)
            
            # Fiyat hassasiyeti
            precision_info = BinanceService.get_symbol_precision(symbol)
            
            # TradingView linki
            tv_link = AnalysisService.create_tradingview_link(symbol, timeframe)
            
            # Özet bilgi
            return {
                'available': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': analysis.get('current_price', 0),
                'trend_direction': analysis.get('trend_direction', 'Unknown'),
                'ratio_percent': analysis.get('ratio_percent', 0),
                'z_score': analysis.get('z_score', 0),
                'data_quality': analysis.get('data_quality', 'Unknown'),
                'candle_count': analysis.get('candle_count', 0),
                '24h_change': market_info.get('price_change_percent', 0),
                '24h_volume': market_info.get('volume', 0),
                'price_precision': precision_info.get('price_precision', 4),
                'tradingview_link': tv_link,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.debug(f"Enhanced symbol info error for {symbol}: {e}")
            return {
                'available': False,
                'reason': f'Error: {str(e)}',
                'symbol': symbol
            }