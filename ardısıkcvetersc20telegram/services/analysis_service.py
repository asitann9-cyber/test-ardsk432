"""
Analiz servisleri
Ardışık mum analizi, C-Signal hesaplama ve ters momentum tespiti
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
    """Ardışık mum ve ters momentum analizi için service sınıfı"""
    
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
    def analyze_consecutive_candles(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ardışık mum yapıları analizi
        
        Args:
            df (pd.DataFrame): OHLCV verileri
            
        Returns:
            Dict[str, Any]: Ardışık mum analiz sonuçları
        """
        try:
            if df is None or len(df) < 5:
                return {
                    'consecutive_type': 'None',
                    'consecutive_count': 0,
                    'percentage_change': 0.0
                }
            
            # Long ve short mum tanımları
            df['is_long'] = df['close'] > df['open']
            df['is_short'] = df['close'] < df['open']
            
            # Mevcut ardışık yapıyı hesapla
            current_consecutive_count = 0
            current_consecutive_type = 'None'
            start_price = 0.0
            end_price = 0.0
            
            # Sondan başlayarak ardışık yapıyı say
            for i in range(len(df) - 1, -1, -1):
                if i == len(df) - 1:  # İlk mum (en son)
                    if df.iloc[i]['is_long']:
                        current_consecutive_type = 'Long'
                        current_consecutive_count = 1
                        start_price = df.iloc[i]['low']
                        end_price = df.iloc[i]['high']
                    elif df.iloc[i]['is_short']:
                        current_consecutive_type = 'Short'
                        current_consecutive_count = 1
                        start_price = df.iloc[i]['high']
                        end_price = df.iloc[i]['low']
                    else:
                        break
                else:
                    # Ardışık yapının devam edip etmediğini kontrol et
                    if (current_consecutive_type == 'Long' and df.iloc[i]['is_long']):
                        current_consecutive_count += 1
                        start_price = df.iloc[i]['low']  # İlk mumun low'u
                    elif (current_consecutive_type == 'Short' and df.iloc[i]['is_short']):
                        current_consecutive_count += 1
                        start_price = df.iloc[i]['high']  # İlk mumun high'ı
                    else:
                        break
            
            # Yüzdelik değişimi hesapla
            percentage_change = 0.0
            if current_consecutive_count > 0 and start_price != 0:
                if current_consecutive_type == 'Long':
                    # (son mumun high - ilk mumun low) / ilk mumun low * 100
                    percentage_change = ((end_price - start_price) / start_price) * 100
                elif current_consecutive_type == 'Short':
                    # (ilk mumun high - son mumun low) / ilk mumun high * 100
                    percentage_change = ((start_price - end_price) / start_price) * 100
            
            return {
                'consecutive_type': current_consecutive_type,
                'consecutive_count': current_consecutive_count,
                'percentage_change': round(percentage_change, 2)
            }
            
        except Exception as e:
            logger.debug(f"Ardışık mum analizi hatası: {e}")
            return {
                'consecutive_type': 'None',
                'consecutive_count': 0,
                'percentage_change': 0.0
            }
    
    @staticmethod
    def detect_reverse_momentum(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ters momentum tespiti:
        - Ardışık Long + C-Signal <= -10 = Ters momentum (C↓)
        - Ardışık Short + C-Signal >= +10 = Ters momentum (C↑)
        
        Args:
            symbol_data (Dict[str, Any]): Sembol verileri
            
        Returns:
            Dict[str, Any]: Ters momentum analiz sonuçları
        """
        try:
            consecutive_type = symbol_data.get('max_consecutive_type', 'None')
            c_signal = symbol_data.get('c_signal', None)
            
            if c_signal is None or consecutive_type == 'None':
                return {
                    'has_reverse_momentum': False,
                    'reverse_type': 'None',
                    'signal_strength': 'None',
                    'alert_message': '',
                    'signal_value': None
                }
            
            c_signal_value = float(c_signal)
            
            # --- TERS MOMENTUM TESPİTİ (C10 ve üzeri) ---
            if consecutive_type == 'Long' and c_signal_value <= -10:
                strength = 'Strong' if abs(c_signal_value) >= 20 else 'Medium'
                return {
                    'has_reverse_momentum': True,
                    'reverse_type': 'C↓',  # Long'dan Short'a dönüş sinyali
                    'signal_strength': strength,
                    'signal_value': c_signal_value,
                    'alert_message': f'TERS MOMENTUM: Long ardışık + C-Signal negatif ({c_signal_value})'
                }
            elif consecutive_type == 'Short' and c_signal_value >= 10:
                strength = 'Strong' if c_signal_value >= 20 else 'Medium'
                return {
                    'has_reverse_momentum': True,
                    'reverse_type': 'C↑',  # Short'dan Long'a dönüş sinyali
                    'signal_strength': strength,
                    'signal_value': c_signal_value,
                    'alert_message': f'TERS MOMENTUM: Short ardışık + C-Signal pozitif ({c_signal_value})'
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
        Tek sembol için ardışık mum analizi
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            
        Returns:
            Optional[Dict[str, Any]]: Analiz sonuçları
        """
        try:
            # Binance'den veri çek
            df = BinanceService.fetch_klines_data(symbol, timeframe)
            if df is None or len(df) < 5:
                return None
            
            # Ardışık mum analizini yap
            consecutive_analysis = AnalysisService.analyze_consecutive_candles(df)
            
            # Son fiyat
            current_price = float(df['close'].iloc[-1])
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'consecutive_type': consecutive_analysis['consecutive_type'],
                'consecutive_count': consecutive_analysis['consecutive_count'],
                'percentage_change': consecutive_analysis['percentage_change'],
                'last_update': df['timestamp'].iloc[-1]
            }
            
        except Exception as e:
            logger.debug(f"Tek sembol analiz hatası {symbol}: {e}")
            return None
    
    @staticmethod
    def analyze_multiple_symbols(symbols: List[str], timeframe: str = '4h', max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Çoklu sembol analizi - Paralel işleme
        
        Args:
            symbols (List[str]): Sembol listesi
            timeframe (str): Zaman dilimi
            max_workers (int): Maksimum worker sayısı
            
        Returns:
            List[Dict[str, Any]]: Analiz sonuçları listesi
        """
        try:
            logger.info(f"{len(symbols)} sembol için {timeframe} ardışık mum analizi başlatılıyor...")
            
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
                        logger.debug(f"Parallel analiz hatası: {e}")
            
            # Ardışık sayıya göre sırala (en yüksek ardışık sayısı üstte)
            results.sort(key=lambda x: x.get('consecutive_count', 0), reverse=True)
            
            logger.info(f"✅ {len(results)} sembol analiz edildi")
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
        Analiz özetini hazırla
        
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
                    'long_count': 0,
                    'short_count': 0,
                    'high_consecutive_count': 0,
                    'max_consecutive': 0,
                    'timeframe': timeframe,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            long_count = sum(1 for r in results if r.get('consecutive_type') == 'Long')
            short_count = sum(1 for r in results if r.get('consecutive_type') == 'Short')
            high_consecutive_count = sum(1 for r in results if r.get('consecutive_count', 0) >= 5)
            max_consecutive = max((r.get('consecutive_count', 0) for r in results), default=0)
            
            return {
                'total_symbols': len(results),
                'long_count': long_count,
                'short_count': short_count,
                'high_consecutive_count': high_consecutive_count,
                'max_consecutive': max_consecutive,
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Analiz özeti hatası: {e}")
            return {
                'total_symbols': 0,
                'long_count': 0,
                'short_count': 0,
                'high_consecutive_count': 0,
                'max_consecutive': 0,
                'timeframe': timeframe,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    @staticmethod
    def update_symbol_with_c_signal(symbol_data: Dict[str, Any], timeframe: str, preserve_manual_type: bool = False) -> Dict[str, Any]:
        """
        🔒 MANUEL TÜR KORUNARAK Sembol verisini C-Signal ile güncelle
        
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
            df = BinanceService.fetch_klines_data(symbol, timeframe)
            if df is not None and len(df) >= 16:
                # C-Signal hesapla
                c_signal = AnalysisService.calculate_c_signal(df)
                symbol_data['c_signal'] = c_signal
                symbol_data['c_signal_update_time'] = datetime.now().strftime('%H:%M')
                
                # 🔒 MANUEL TÜR KORUMA KONTROLÜ
                if preserve_manual_type:
                    # Manuel tür koruması aktif - sadece C-Signal güncelle
                    logger.debug(f"🔒 {symbol} manuel tür koruması: {symbol_data.get('max_consecutive_type', 'Unknown')} - TÜR KORUNUYOR")
                else:
                    # Normal güncelleme - tür de güncellenebilir
                    consecutive_analysis = AnalysisService.analyze_consecutive_candles(df)
                    
                    # Ardışık sayı rekoru kontrolü
                    current_count = consecutive_analysis.get('consecutive_count', 0)
                    max_count = symbol_data.get('max_consecutive_count', 0)
                    
                    if current_count > max_count:
                        # Yeni rekor - tür ve sayı güncelle
                        symbol_data['max_consecutive_count'] = current_count
                        symbol_data['max_consecutive_type'] = consecutive_analysis.get('consecutive_type', 'None')
                        symbol_data['max_percentage_change'] = consecutive_analysis.get('percentage_change', 0)
                        logger.debug(f"🔄 {symbol} normal güncelleme: {consecutive_analysis.get('consecutive_type', 'None')} - TÜR GÜNCELLENDİ")
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
        Sonuçları filtrele
        
        Args:
            results (List[Dict[str, Any]]): Tüm sonuçlar
            filter_type (str): Filtre tipi (all, long, short, high-count)
            
        Returns:
            List[Dict[str, Any]]: Filtrelenmiş sonuçlar
        """
        if not results:
            return []
        
        filtered_results = []
        
        if filter_type == 'long':
            filtered_results = [r for r in results if r.get('consecutive_type') == 'Long']
        elif filter_type == 'short':
            filtered_results = [r for r in results if r.get('consecutive_type') == 'Short']
        elif filter_type == 'high-count':
            filtered_results = [r for r in results if r.get('consecutive_count', 0) >= 5]
        else:  # 'all'
            filtered_results = results
        
        # Ardışık sayıya göre tekrar sırala
        filtered_results.sort(key=lambda x: x.get('consecutive_count', 0), reverse=True)
        
        # Rank'ı güncelle
        for i, result in enumerate(filtered_results):
            result['filtered_rank'] = i + 1
        
        return filtered_results
    
    @staticmethod
    def is_high_priority_symbol(result: Dict[str, Any]) -> bool:
        """
        Yüksek öncelikli sembol mu kontrol et
        5+ ardışık VEYA %10+ değişim
        
        Args:
            result (Dict[str, Any]): Analiz sonucu
            
        Returns:
            bool: Yüksek öncelikli ise True
        """
        consecutive_count = result.get('consecutive_count', 0)
        percentage_change = abs(result.get('percentage_change', 0))
        
        return consecutive_count >= 5 and percentage_change >= 10.0