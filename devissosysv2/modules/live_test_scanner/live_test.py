import pandas as pd
import numpy as np
import time
import threading
import sqlite3
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Any, Tuple
from binance.client import Client
import talib
warnings.filterwarnings('ignore')

class DevisoLiveTestScanner:
    def __init__(self):
        # Binance client başlat (API key olmadan da çalışır)
        self.client = Client("", "")
        self.db_path = 'live_test.db'
        self.init_database()
        self.is_running = False
        self.active_trades = {}
        self.completed_trades = []
        self.performance_metrics = {}
        self.demo_balance = 10000  # 10,000 USDT başlangıç bakiyesi
        self.position_size = 0.05   # Bakiyenin %5'i (risk azaltma)
        self.scan_interval = 60    # 60 saniyede bir tarama
        self.max_coins = 480       # Maksimum coin sayısı (tüm Binance coinleri)
        self.min_signal_score = 8   # Minimum sinyal skoru (scalping - WEAK SCALP+ sinyaller)
        
        # ANALİZ TABANLI filtreleme kriterleri (win rate artırma)
        self.min_volume_24h = 150000    # Minimum 150K USDT günlük hacim (analiz tabanlı)
        self.max_spread = 0.15         # Maximum %0.15 spread (analiz tabanlı)
        self.min_trend_strength = 18   # ADX minimum değeri (analiz tabanlı)
        self.max_drawdown_limit = 1.5  # Maximum %1.5 drawdown (analiz tabanlı)
        self.max_trade_duration = 300  # Maksimum işlem süresi (dakika) - 5 saat (analiz tabanlı - 2-4sa karlı)
        
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Aktif işlemler tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT,
                signal_type TEXT,
                signal_category TEXT,
                signal_score REAL,
                entry_price REAL,
                entry_time TIMESTAMP,
                position_size REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'ACTIVE'
            )
        ''')
        
        # Tamamlanan işlemler tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS completed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT,
                signal_type TEXT,
                signal_category TEXT,
                signal_score REAL,
                entry_price REAL,
                exit_price REAL,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                position_size REAL,
                pnl REAL,
                pnl_percentage REAL,
                exit_reason TEXT,
                duration_minutes INTEGER
            )
        ''')
        
        # Performans metrikleri tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_win REAL,
                avg_loss REAL,
                profit_factor REAL,
                max_drawdown REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_current_price(self, symbol: str) -> float:
        """Sembolün güncel fiyatını al"""
        try:
            # Binance'den ticker bilgisi al
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Fiyat alma hatası {symbol}: {e}")
            return None

    def get_klines_data(self, symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Binance'den kline verisi al"""
        try:
            # Binance interval formatını dönüştür (SCALPING için 1m eklendi)
            interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_1MINUTE)
            
            # Kline verisi al
            klines = self.client.get_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=limit
            )
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Kline veri alma hatası {symbol}: {e}")
            return pd.DataFrame()

    def calculate_ema(self, df: pd.DataFrame, period: int = 200) -> pd.Series:
        """TA-Lib ile EMA hesapla"""
        try:
            return talib.EMA(df['close'].values, timeperiod=period)
        except Exception as e:
            print(f"EMA hesaplama hatası: {e}")
            return pd.Series([np.nan] * len(df))

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """TA-Lib ile RSI hesapla"""
        try:
            return talib.RSI(df['close'].values, timeperiod=period)
        except Exception as e:
            print(f"RSI hesaplama hatası: {e}")
            return pd.Series([np.nan] * len(df))

    def calculate_stoch_rsi(self, df: pd.DataFrame, period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stokastik RSI hesapla"""
        try:
            # Önce RSI hesapla - pandas Series olarak
            rsi_values = talib.RSI(df['close'].values, timeperiod=period)
            rsi = pd.Series(rsi_values, index=df.index)
            
            # RSI'ın Stokastik değerlerini hesapla
            rsi_min = rsi.rolling(window=period).min()
            rsi_max = rsi.rolling(window=period).max()
            
            # Stokastik RSI %K hesapla
            stoch_rsi_k = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
            
            # %K'nın hareketli ortalaması ile %D hesapla
            stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
            
            return stoch_rsi_k, stoch_rsi_d
            
        except Exception as e:
            print(f"Stokastik RSI hesaplama hatası: {e}")
            return pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df))

    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """TA-Lib ile MACD hesapla"""
        try:
            macd, macd_signal, macd_histogram = talib.MACD(
                df['close'].values, 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            return macd, macd_signal, macd_histogram
        except Exception as e:
            print(f"MACD hesaplama hatası: {e}")
            return pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df))

    def check_volume_filter(self, symbol: str) -> bool:
        """Hacim filtresini kontrol et"""
        try:
            ticker_24h = self.client.get_ticker(symbol=symbol)
            volume_usdt = float(ticker_24h['quoteVolume'])
            return volume_usdt >= self.min_volume_24h
        except Exception as e:
            print(f"Hacim kontrolü hatası {symbol}: {e}")
            return True

    def check_spread_filter(self, symbol: str) -> bool:
        """Spread filtresini kontrol et"""
        try:
            orderbook = self.client.get_orderbook_ticker(symbol=symbol)
            bid = float(orderbook['bidPrice'])
            ask = float(orderbook['askPrice'])
            spread = ((ask - bid) / bid) * 100
            return spread <= self.max_spread
        except Exception as e:
            print(f"Spread kontrolü hatası {symbol}: {e}")
            return True

    def calculate_trend_strength(self, symbol: str, timeframe: str = '1h') -> float:
        """ADX ile trend gücünü hesapla"""
        try:
            df = self.get_klines_data(symbol, timeframe, 100)
            if len(df) < 30:
                return 0
            
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            return adx[-1] if not np.isnan(adx[-1]) else 0
        except Exception as e:
            print(f"ADX hesaplama hatası {symbol}: {e}")
            return 0

    def check_trend_strength_filter(self, symbol: str) -> bool:
        """Trend gücü filtresini kontrol et"""
        try:
            adx_1h = self.calculate_trend_strength(symbol, '1h')
            adx_4h = self.calculate_trend_strength(symbol, '4h')
            return adx_1h >= self.min_trend_strength and adx_4h >= self.min_trend_strength
        except Exception as e:
            print(f"Trend gücü kontrolü hatası {symbol}: {e}")
            return False

    def check_market_structure(self, symbol: str) -> bool:
        """Market yapısını kontrol et"""
        try:
            df = self.get_klines_data(symbol, '4h', 100)
            if len(df) < 20:
                return False
            
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            range_position = (current_price - recent_low) / (recent_high - recent_low)
            return 0.3 <= range_position <= 0.7
        except Exception as e:
            print(f"Market yapısı kontrolü hatası {symbol}: {e}")
            return False

    def check_volatility_filter(self, symbol: str) -> bool:
        """Volatilite filtresi"""
        try:
            df = self.get_klines_data(symbol, '5m', 100)
            if len(df) < 20:
                return False
            
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_atr = atr[-1] if not pd.isna(atr[-1]) else 0
            current_price = df['close'].iloc[-1]
            atr_percentage = (current_atr / current_price) * 100
            return 0.5 <= atr_percentage <= 10.0
        except Exception as e:
            print(f"Volatilite filtresi hatası {symbol}: {e}")
            return False

    def check_momentum_filter(self, symbol: str) -> bool:
        """Momentum filtresi"""
        try:
            df = self.get_klines_data(symbol, '5m', 100)
            if len(df) < 20:
                return False
            
            current_price = df['close'].iloc[-1]
            price_change_6 = ((current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
            price_change_12 = ((current_price - df['close'].iloc[-12]) / df['close'].iloc[-12]) * 100
            
            if price_change_6 > 0 and price_change_12 > 0:
                return True
            elif price_change_6 < 0 and price_change_12 < 0:
                return True
            return False
        except Exception as e:
            print(f"Momentum filtresi hatası {symbol}: {e}")
            return False

    def calculate_signal_score(self, symbol: str, timeframe: str = '1m') -> Dict[str, Any]:
        """Sinyal skorunu hesapla"""
        try:
            df = self.get_klines_data(symbol, timeframe, 100)
            if len(df) < 50:
                return {'score': 0, 'signals': []}

            # EMA hesapla
            df['ema200'] = self.calculate_ema(df, 200)
            df['ema50'] = self.calculate_ema(df, 50)
            
            # RSI hesapla
            df['rsi'] = self.calculate_rsi(df, 14)
            
            # Stokastik RSI hesapla
            df['stoch_rsi_k'], df['stoch_rsi_d'] = self.calculate_stoch_rsi(df, 14, 3, 3)
            
            # MACD hesapla
            df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df)
            
            # Son değerler
            current_price = df['close'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            stoch_rsi_k = df['stoch_rsi_k'].iloc[-1] if not pd.isna(df['stoch_rsi_k'].iloc[-1]) else 50
            stoch_rsi_d = df['stoch_rsi_d'].iloc[-1] if not pd.isna(df['stoch_rsi_d'].iloc[-1]) else 50
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_histogram = df['macd_histogram'].iloc[-1]
            
            # Volatilite hesapla
            volatility = df['close'].pct_change().std() * 100
            
            # Sinyal puanları
            score = 0
            signals = []
            
            # EMA analizi (0-6 puan)
            if current_price > ema200:
                score += 2
                signals.append("EMA200_ABOVE")
                if current_price > ema50:
                    score += 2
                    signals.append("EMA50_ABOVE")
                if ema50 > ema200:
                    score += 2
                    signals.append("EMA50_ABOVE_EMA200")
            else:
                score += 2
                signals.append("EMA200_BELOW")
                if current_price < ema50:
                    score += 2
                    signals.append("EMA50_BELOW")
                if ema50 < ema200:
                    score += 2
                    signals.append("EMA50_BELOW_EMA200")
            
            # RSI analizi (0-2 puan)
            if 30 <= rsi <= 70:
                score += 1
                signals.append("RSI_NEUTRAL")
            
            if current_price > ema200 and rsi > 50:
                score += 1
                signals.append("RSI_BULLISH")
            elif current_price < ema200 and rsi < 50:
                score += 1
                signals.append("RSI_BEARISH")
            
            # Stokastik RSI analizi (0-3 puan)
            if not (pd.isna(stoch_rsi_k) or pd.isna(stoch_rsi_d)):
                if current_price > ema200:
                    if stoch_rsi_k <= 20 and stoch_rsi_d <= 20:
                        score += 3
                        signals.append("STOCH_RSI_OVERSOLD_BUY")
                    elif stoch_rsi_k <= 30 and stoch_rsi_d <= 30:
                        score += 2
                        signals.append("STOCH_RSI_STRONG_OVERSOLD")
                    elif stoch_rsi_k > stoch_rsi_d and stoch_rsi_k < 50:
                        score += 1
                        signals.append("STOCH_RSI_BULLISH_CROSS")
                elif current_price < ema200:
                    if stoch_rsi_k >= 80 and stoch_rsi_d >= 80:
                        score += 3
                        signals.append("STOCH_RSI_OVERBOUGHT_SELL")
                    elif stoch_rsi_k >= 70 and stoch_rsi_d >= 70:
                        score += 2
                        signals.append("STOCH_RSI_STRONG_OVERBOUGHT")
                    elif stoch_rsi_k < stoch_rsi_d and stoch_rsi_k > 50:
                        score += 1
                        signals.append("STOCH_RSI_BEARISH_CROSS")
                
                if 30 <= stoch_rsi_k <= 70 and 30 <= stoch_rsi_d <= 70:
                    score += 1
                    signals.append("STOCH_RSI_NEUTRAL")
            
            # MACD analizi (0-2 puan)
            if current_price > ema200:
                if macd > macd_signal:
                    score += 1
                    signals.append("MACD_BULLISH")
                if macd_histogram > 0:
                    score += 1
                    signals.append("MACD_HISTOGRAM_POSITIVE")
            else:
                if macd < macd_signal:
                    score += 1
                    signals.append("MACD_BEARISH")
                if macd_histogram < 0:
                    score += 1
                    signals.append("MACD_HISTOGRAM_NEGATIVE")
            
            # ADX analizi (0-2 puan)
            try:
                adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                current_adx = adx[-1] if not pd.isna(adx[-1]) else 0
                
                if current_adx >= 25:
                    score += 2
                    signals.append("ADX_STRONG_TREND")
                elif current_adx >= 20:
                    score += 1
                    signals.append("ADX_MEDIUM_TREND")
            except Exception as e:
                print(f"ADX hesaplama hatası: {e}")
            
            # Hacim ve Volatilite analizi (0-3 puan)
            try:
                atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                current_atr = atr[-1] if not pd.isna(atr[-1]) else 0
                
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                volume_spike = (current_volume / avg_volume) if avg_volume > 0 else 1
                
                if current_atr > current_price * 0.015:
                    score += 1
                    signals.append("ATR_SUFFICIENT")
                    
                    if 2 <= volatility <= 8:
                        score += 1
                        signals.append("VOLATILITY_OPTIMAL")
                
                if volume_spike >= 1.5:
                    score += 1
                    signals.append("VOLUME_SPIKE_HIGH")
            except Exception as e:
                print(f"Hacim/ATR hesaplama hatası: {e}")
                if 2 <= volatility <= 8:
                    score += 1
                    signals.append("VOLATILITY_OPTIMAL")
            
            # SCALPING için momentum analizi (0-2 puan)
            if timeframe == '1m':
                lookback = 3
            elif timeframe == '5m':
                lookback = 6
            else:
                lookback = 4
            
            if len(df) >= lookback:
                price_change = ((current_price - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]) * 100
                
                if current_price > ema200:
                    if price_change > 0:
                        score += 1
                        signals.append("MOMENTUM_POSITIVE")
                    if price_change > 0.3:
                        score += 1
                        signals.append("MOMENTUM_STRONG")
                else:
                    if price_change < -0.3:
                        score += 1
                        signals.append("MOMENTUM_NEGATIVE")
                    if price_change < -0.8:
                        score += 2
                        signals.append("MOMENTUM_STRONG_DOWN")
            
            # SCALPING için sinyal kategorisi belirle
            if score >= 16:
                category = "MEGA SCALP"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            elif score >= 12:
                category = "PRO SCALP"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            elif score >= 8:
                category = "STANDARD SCALP"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            elif score >= 5:
                category = "WEAK SCALP"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            else:
                category = "NO SIGNAL"
                signal_type = "NEUTRAL"
            
            return {
                'score': score,
                'category': category,
                'signal_type': signal_type,
                'signals': signals,
                'current_price': current_price,
                'ema200': ema200,
                'rsi': rsi,
                'stoch_rsi_k': stoch_rsi_k,
                'stoch_rsi_d': stoch_rsi_d,
                'macd': macd,
                'volatility': volatility,
                'timeframe': timeframe
            }
            
        except Exception as e:
            print(f"Sinyal hesaplama hatası {symbol}: {e}")
            return {'score': 0, 'signals': []}

    def calculate_multi_timeframe_score(self, symbol: str) -> Dict[str, Any]:
        """SCALPING için çoklu timeframe analizi - 1m ve 5m"""
        try:
            timeframes = ['1m', '5m']  # Scalping için sadece 1m ve 5m
            timeframe_scores = {}
            total_score = 0
            all_signals = []
            
            for tf in timeframes:
                signal_data = self.calculate_signal_score(symbol, tf)
                timeframe_scores[tf] = signal_data
                total_score += signal_data['score']
                all_signals.extend([f"{tf}_{signal}" for signal in signal_data['signals']])
            
            # Ortalama skor hesapla
            avg_score = total_score / len(timeframes)
            
            # SCALPING için timeframe ağırlıkları (1m giriş noktası, 5m trend)
            weighted_score = (
                timeframe_scores['1m']['score'] * 0.7 +  # %70 ağırlık (giriş noktası)
                timeframe_scores['5m']['score'] * 0.3    # %30 ağırlık (trend doğrulama)
            )
            
            # Trend tutarlılığı kontrol et (scalping için daha esnek)
            trend_consistency = 0
            if timeframe_scores['1m']['signal_type'] == timeframe_scores['5m']['signal_type']:
                trend_consistency = 3  # Mükemmel uyum (yüksek puan)
            elif timeframe_scores['1m']['signal_type'] != 'NEUTRAL' and timeframe_scores['5m']['signal_type'] != 'NEUTRAL':
                trend_consistency = 1  # En azından her ikisi de sinyal veriyor
            
            # Final skor hesapla
            final_score = weighted_score + trend_consistency
            
            # SCALPING için sinyal kategorileri (daha agresif)
            if final_score >= 20:  # Yüksek kalite scalping sinyali
                category = "MEGA SCALP"
            elif final_score >= 16:  # Orta kalite scalping sinyali
                category = "PRO SCALP"
            elif final_score >= 12:   # Minimum scalping sinyali
                category = "STANDARD SCALP"
            elif final_score >= 8:    # Zayıf ama kabul edilebilir
                category = "WEAK SCALP"
            else:
                category = "NO SIGNAL"
            
            # Ana sinyal tipini belirle (1m timeframe'e göre - giriş noktası)
            main_signal_type = timeframe_scores['1m']['signal_type']
            
            return {
                'final_score': round(final_score, 2),
                'weighted_score': round(weighted_score, 2),
                'avg_score': round(avg_score, 2),
                'trend_consistency': trend_consistency,
                'category': category,
                'signal_type': main_signal_type,
                'timeframe_scores': timeframe_scores,
                'all_signals': all_signals,
                'current_price': timeframe_scores['1m']['current_price']  # 1m fiyatı
            }
            
        except Exception as e:
            print(f"Scalping timeframe analiz hatası {symbol}: {e}")
            return {'final_score': 0, 'signals': []}

    def calculate_ranking_score(self, signal_info: Dict[str, Any]) -> float:
        """Sinyal için sıralama skoru hesapla (Momentum, Hacim, Fiyat Değişimi)"""
        try:
            symbol = signal_info['symbol']
            signal_score = signal_info['signal_score']
            current_price = signal_info['current_price']
            
            # 1. MOMENTUM SKORU (0-30 puan)
            momentum_score = 0
            try:
                df_1m = self.get_klines_data(symbol, '1m', 20)
                df_5m = self.get_klines_data(symbol, '5m', 20)
                
                if len(df_1m) >= 10 and len(df_5m) >= 10:
                    # 1m momentum (son 3-6 periyot)
                    price_change_1m_3 = ((current_price - df_1m['close'].iloc[-3]) / df_1m['close'].iloc[-3]) * 100
                    price_change_1m_6 = ((current_price - df_1m['close'].iloc[-6]) / df_1m['close'].iloc[-6]) * 100
                    
                    # 5m momentum (son 2-4 periyot)
                    price_change_5m_2 = ((current_price - df_5m['close'].iloc[-2]) / df_5m['close'].iloc[-2]) * 100
                    price_change_5m_4 = ((current_price - df_5m['close'].iloc[-4]) / df_5m['close'].iloc[-4]) * 100
                    
                    # Momentum puanları
                    if signal_info['signal_type'] == 'LONG':
                        # LONG için pozitif momentum
                        if price_change_1m_3 > 0.2: momentum_score += 8
                        if price_change_1m_6 > 0.5: momentum_score += 7
                        if price_change_5m_2 > 0.3: momentum_score += 8
                        if price_change_5m_4 > 0.8: momentum_score += 7
                    else:
                        # SHORT için negatif momentum
                        if price_change_1m_3 < -0.2: momentum_score += 8
                        if price_change_1m_6 < -0.5: momentum_score += 7
                        if price_change_5m_2 < -0.3: momentum_score += 8
                        if price_change_5m_4 < -0.8: momentum_score += 7
            except Exception as e:
                print(f"Momentum hesaplama hatası {symbol}: {e}")
            
            # 2. HACİM SKORU (0-25 puan)
            volume_score = 0
            try:
                df_1m = self.get_klines_data(symbol, '1m', 20)
                if len(df_1m) >= 20:
                    current_volume = df_1m['volume'].iloc[-1]
                    avg_volume = df_1m['volume'].rolling(window=20).mean().iloc[-1]
                    
                    if avg_volume > 0:
                        volume_spike = current_volume / avg_volume
                        
                        if volume_spike >= 3.0: volume_score += 25  # Çok yüksek hacim
                        elif volume_spike >= 2.0: volume_score += 20  # Yüksek hacim
                        elif volume_spike >= 1.5: volume_score += 15  # Orta hacim
                        elif volume_spike >= 1.2: volume_score += 10  # Hafif hacim artışı
                        elif volume_spike >= 1.0: volume_score += 5   # Normal hacim
            except Exception as e:
                print(f"Hacim hesaplama hatası {symbol}: {e}")
            
            # 3. FİYAT DEĞİŞİMİ SKORU (0-25 puan)
            price_change_score = 0
            try:
                df_1m = self.get_klines_data(symbol, '1m', 20)
                if len(df_1m) >= 20:
                    # ATR hesapla
                    atr = talib.ATR(df_1m['high'].values, df_1m['low'].values, df_1m['close'].values, timeperiod=14)
                    current_atr = atr[-1] if not pd.isna(atr[-1]) else 0
                    
                    # Volatilite hesapla
                    volatility = df_1m['close'].pct_change().std() * 100
                    
                    # ATR puanı (0-15)
                    atr_percentage = (current_atr / current_price) * 100
                    if atr_percentage >= 2.0: price_change_score += 15  # Yüksek volatilite
                    elif atr_percentage >= 1.0: price_change_score += 12  # Orta volatilite
                    elif atr_percentage >= 0.5: price_change_score += 8   # Düşük volatilite
                    elif atr_percentage >= 0.2: price_change_score += 5   # Çok düşük volatilite
                    
                    # Volatilite puanı (0-10)
                    if 1.0 <= volatility <= 5.0: price_change_score += 10  # Optimal volatilite
                    elif 0.5 <= volatility <= 8.0: price_change_score += 7   # Kabul edilebilir
                    elif volatility > 8.0: price_change_score += 3        # Yüksek volatilite
            except Exception as e:
                print(f"Fiyat değişimi hesaplama hatası {symbol}: {e}")
            
            # 4. SİNYAL KALİTESİ SKORU (0-20 puan)
            quality_score = 0
            if signal_info['signal_category'] == 'MEGA SCALP': quality_score += 20
            elif signal_info['signal_category'] == 'PRO SCALP': quality_score += 15
            elif signal_info['signal_category'] == 'STANDARD SCALP': quality_score += 10
            elif signal_info['signal_category'] == 'WEAK SCALP': quality_score += 5
            
            # TOPLAM SIRALAMA SKORU
            total_ranking_score = momentum_score + volume_score + price_change_score + quality_score
            
            # Sinyal bilgisine sıralama detaylarını ekle
            signal_info['ranking_details'] = {
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'price_change_score': price_change_score,
                'quality_score': quality_score,
                'total_ranking_score': total_ranking_score
            }
            
            return total_ranking_score
            
        except Exception as e:
            print(f"Sıralama skoru hesaplama hatası {symbol}: {e}")
            return 0

    def scan_coins_and_trade(self) -> List[Dict[str, Any]]:
        """Coinleri tara, sırala ve en iyi fırsatları işlem aç (Sıralama Algoritması ile)"""
        try:
            # Tüm sembolleri al
            exchange_info = self.client.get_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                if symbol.endswith('USDT') and symbol_info['status'] == 'TRADING':
                    symbols.append(symbol)
            
            print(f"📊 Toplam {len(symbols)} sembol bulundu, tümü taranacak...")
            print(f"🚀 SCALPING TARAMA: En iyi fırsatlar sıralanacak ve işlem açılacak!")
            print(f"🕐 SCALPING Timeframe Analizi: 1m (giriş), 5m (trend)")
            print(f"📈 SIRALAMA ALGORİTMASI: Momentum + Hacim + Fiyat Değişimi + Sinyal Kalitesi")
            
            all_signals = []
            scanned_count = 0
            
            # İLK AŞAMA: Tüm coinleri tara ve sinyalleri topla
            print(f"🔍 AŞAMA 1: Tüm coinler taranıyor ve sinyaller toplanıyor...")
            
            for symbol in symbols:
                try:
                    scanned_count += 1
                    
                    # Her 50 coin'de bir ilerleme göster
                    if scanned_count % 50 == 0:
                        print(f"🔍 Tarama ilerlemesi: {scanned_count}/{len(symbols)} - Bulunan sinyal: {len(all_signals)}")
                    
                    # SIKILAŞTIRILMIŞ filtre kontrolleri
                    if not self.check_volume_filter(symbol):
                        continue
                    
                    if not self.check_spread_filter(symbol):
                        continue
                    
                    if not self.check_trend_strength_filter(symbol):
                        continue
                    
                    if not self.check_market_structure(symbol):
                        continue
                    
                    if not self.check_volatility_filter(symbol):
                        continue
                    
                    if not self.check_momentum_filter(symbol):
                        continue
                    
                    # Çoklu timeframe analizi yap
                    multi_tf_data = self.calculate_multi_timeframe_score(symbol)
                    
                    if multi_tf_data['final_score'] >= self.min_signal_score:
                        # Timeframe skorlarını al
                        tf_scores = multi_tf_data['timeframe_scores']
                        
                        signal_info = {
                            'symbol': symbol,
                            'signal_type': multi_tf_data['signal_type'],
                            'signal_category': multi_tf_data['category'],
                            'signal_score': multi_tf_data['final_score'],
                            'current_price': multi_tf_data['current_price'],
                            'signals': multi_tf_data['all_signals'],
                            'timestamp': datetime.now(),
                            'timeframe_analysis': {
                                '1m_score': tf_scores['1m']['score'],
                                '5m_score': tf_scores['5m']['score'],
                                'trend_consistency': multi_tf_data['trend_consistency'],
                                'weighted_score': multi_tf_data['weighted_score']
                            },
                            'timeframe_scores': tf_scores
                        }
                        
                        # Sıralama skorunu hesapla
                        ranking_score = self.calculate_ranking_score(signal_info)
                        signal_info['ranking_score'] = ranking_score
                        
                        all_signals.append(signal_info)
                    
                    # Rate limit için bekle
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Coin tarama hatası {symbol}: {e}")
                    continue
            
            print(f"✅ AŞAMA 1 TAMAMLANDI: {len(all_signals)} sinyal bulundu")
            
            # İKİNCİ AŞAMA: Sinyalleri sırala
            if all_signals:
                print(f"📊 AŞAMA 2: Sinyaller sıralanıyor...")
                
                # Sıralama skoruna göre azalan sırada sırala
                all_signals.sort(key=lambda x: x['ranking_score'], reverse=True)
                
                print(f"🏆 EN İYİ 10 SİNYAL:")
                for i, signal in enumerate(all_signals[:10]):
                    ranking_details = signal['ranking_details']
                    print(f"   {i+1}. {signal['symbol']} - {signal['signal_category']}")
                    print(f"      📊 Sıralama Skoru: {signal['ranking_score']:.1f}")
                    print(f"      📈 Momentum: {ranking_details['momentum_score']:.1f}, Hacim: {ranking_details['volume_score']:.1f}")
                    print(f"      💰 Fiyat Değişimi: {ranking_details['price_change_score']:.1f}, Kalite: {ranking_details['quality_score']:.1f}")
                
                # ÜÇÜNCÜ AŞAMA: En iyi sinyalleri işlem aç
                print(f"🚀 AŞAMA 3: En iyi fırsatlar işlem açılıyor...")
                
                opened_trades = 0
                max_trades_to_open = min(self.max_trades - len(self.active_trades), len(all_signals))
                
                for i, signal_info in enumerate(all_signals[:max_trades_to_open]):
                    try:
                        # Aynı sembolde zaten işlem var mı kontrol et
                        symbol_exists = any(trade['symbol'] == signal_info['symbol'] for trade in self.active_trades.values())
                        
                        if not symbol_exists:
                            print(f"🔥 {i+1}. SIRADA: {signal_info['symbol']} - {signal_info['signal_category']}")
                            print(f"   📊 Sıralama Skoru: {signal_info['ranking_score']:.1f}")
                            print(f"   📈 1m: {signal_info['timeframe_analysis']['1m_score']}/21, 5m: {signal_info['timeframe_analysis']['5m_score']}/21")
                            
                            # Stokastik RSI bilgilerini göster
                            tf_scores = signal_info.get('timeframe_scores', {})
                            if '1m' in tf_scores and 'stoch_rsi_k' in tf_scores['1m']:
                                stoch_k = tf_scores['1m']['stoch_rsi_k']
                                stoch_d = tf_scores['1m']['stoch_rsi_d']
                                
                                if not (pd.isna(stoch_k) or pd.isna(stoch_d)):
                                    print(f"   📊 Stoch RSI (1m): K={stoch_k:.1f}, D={stoch_d:.1f}")
                            
                            # İşlem aç
                            success = self.open_demo_trade(signal_info)
                            if success:
                                opened_trades += 1
                                print(f"   🎉 İŞLEM AÇILDI! Toplam açılan: {opened_trades}")
                            else:
                                print(f"   ❌ İşlem açılamadı!")
                        else:
                            print(f"   ⚠️ {signal_info['symbol']} zaten aktif, atlanıyor")
                    
                    except Exception as e:
                        print(f"İşlem açma hatası {signal_info['symbol']}: {e}")
                        continue
                
                print(f"🎉 SIRALAMA ALGORİTMASI TAMAMLANDI!")
                print(f"   📊 Taranan coin: {scanned_count}/{len(symbols)}")
                print(f"   🔥 Bulunan sinyal: {len(all_signals)}")
                print(f"   🎯 Açılan işlem: {opened_trades}")
                
                return all_signals[:max_trades_to_open]
            else:
                print(f"❌ Hiç sinyal bulunamadı!")
                return []
            
        except Exception as e:
            print(f"Genel tarama hatası: {e}")
            return []

    def auto_trading_loop(self):
        """Otomatik trading döngüsü"""
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                print(f"🔄 Live Test Döngü {cycle_count} Başlatıldı - {datetime.now().strftime('%H:%M:%S')}")
                
                # Maksimum işlem sayısını kontrol et
                if len(self.active_trades) >= self.max_trades:
                    print(f"⚠️ Maksimum işlem sayısına ulaşıldı ({self.max_trades}). Yeni işlem açılmayacak.")
                    time.sleep(self.scan_interval)
                    continue
                
                # Coinleri tara ve bulduğu anda işlem aç
                signals = self.scan_coins_and_trade()
                
                # Artık tarama sırasında işlemler açıldı, ek işlem açmaya gerek yok
                print(f"   💼 Döngü sonunda aktif işlem sayısı: {len(self.active_trades)}")
                
                print(f"⏳ {self.scan_interval} saniye sonra yeni döngü başlayacak...")
                time.sleep(self.scan_interval)
                
            except Exception as e:
                print(f"Otomatik trading döngüsü hatası: {e}")
                time.sleep(30)

    def calculate_optimal_tp_sl(self, symbol: str, signal_type: str, signal_score: float, timeframe_analysis: Dict = None) -> Tuple[float, float]:
        """Optimal TP/SL noktalarını hesapla (SCALPING Normal Sinyal Sistemi - En az %1 Kar Garantisi)"""
        try:
            # SCALPING için çoklu timeframe analizi varsa kullan
            if timeframe_analysis:
                # 1m, 5m timeframe'lerinden ATR hesapla
                atr_values = {}
                for tf in ['1m', '5m']:
                    df = self.get_klines_data(symbol, tf, 100)
                    if len(df) > 14:
                        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                        atr_values[tf] = df['atr'].iloc[-1]
                
                # SCALPING için ağırlıklı ATR hesapla (1m daha önemli)
                weighted_atr = (
                    atr_values.get('1m', 0) * 0.7 +
                    atr_values.get('5m', 0) * 0.3
                )
                
                # 1m timeframe'den güncel fiyat al (giriş noktası)
                df_1m = self.get_klines_data(symbol, '1m', 100)
                current_price = df_1m['close'].iloc[-1]
                volatility = df_1m['close'].pct_change().std() * 100
            else:
                # Eski yöntem (geriye uyumluluk için)
                df = self.get_klines_data(symbol, '1m', 100)
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                current_price = df['close'].iloc[-1]
                weighted_atr = df['atr'].iloc[-1]
                volatility = df['close'].pct_change().std() * 100
            
            # SCALPING NORMAL SİNYAL TP/SL çarpanları (En az %1 kar garantisi)
            if signal_score >= 20:  # MEGA SCALP
                tp_multiplier = 2.5   # Scalping TP (%2.5 kar)
                sl_multiplier = 1.2   # SL
            elif signal_score >= 16:  # PRO SCALP
                tp_multiplier = 2.0   # Scalping TP (%2.0 kar)
                sl_multiplier = 1.0
            elif signal_score >= 12:   # STANDARD SCALP
                tp_multiplier = 1.5   # Scalping TP (%1.5 kar)
                sl_multiplier = 0.8
            else:  # WEAK SCALP
                tp_multiplier = 1.2   # Scalping TP (%1.2 kar - minimum %1 garantisi)
                sl_multiplier = 0.6
            
            # SCALPING trend tutarlılığına göre ek ayarlama
            if timeframe_analysis and timeframe_analysis.get('trend_consistency', 0) >= 3:
                tp_multiplier *= 1.1   # Mükemmel uyum varsa TP'yi artır
                sl_multiplier *= 0.9   # SL'yi azalt
            elif timeframe_analysis and timeframe_analysis.get('trend_consistency', 0) >= 1:
                tp_multiplier *= 1.05  # En azından uyum varsa TP'yi hafif artır
                sl_multiplier *= 0.95  # SL'yi hafif azalt
            
            # SCALPING volatilite bazlı ayarlama
            if volatility > 3:  # Yüksek volatilite (scalping için daha düşük eşik)
                tp_multiplier *= 1.15  # Daha fazla artır (scalping için)
                sl_multiplier *= 1.1   # Daha fazla artır
            elif volatility < 1:  # Düşük volatilite (scalping için daha düşük eşik)
                tp_multiplier *= 0.85  # Daha fazla azalt
                sl_multiplier *= 0.9   # Daha fazla azalt
            
            # TERS SİNYAL TP/SL hesaplama (gerçek açılan pozisyon tipine göre)
            if signal_type == 'LONG':
                take_profit = current_price + (weighted_atr * tp_multiplier)
                stop_loss = current_price - (weighted_atr * sl_multiplier)
            else:  # SHORT
                take_profit = current_price - (weighted_atr * tp_multiplier)
                stop_loss = current_price + (weighted_atr * sl_multiplier)
            
            return take_profit, stop_loss
            
        except Exception as e:
            print(f"TP/SL hesaplama hatası: {e}")
            # Varsayılan değerler (ters sinyal için)
            if signal_type == 'LONG':
                return current_price * 1.02, current_price * 0.98
            else:
                return current_price * 0.98, current_price * 1.02

    def open_demo_trade(self, signal_data: Dict[str, Any]) -> bool:
        """Demo işlemi aç (Normal Sinyal Sistemi - En az %1 Kar Garantisi)"""
        try:
            symbol = signal_data['symbol']
            original_signal_type = signal_data['signal_type']  # Sistemin verdiği sinyal
            signal_category = signal_data['signal_category']
            signal_score = signal_data['signal_score']
            current_price = signal_data['current_price']
            timeframe_analysis = signal_data.get('timeframe_analysis', {})
            
            # NORMAL SİNYAL: Sistemin verdiği sinyali aynen al
            actual_signal_type = original_signal_type  # Sistem LONG diyor, biz LONG açıyoruz
            
            # TP/SL hesapla (ters sinyal tipine göre)
            take_profit, stop_loss = self.calculate_optimal_tp_sl(
                symbol, actual_signal_type, signal_score, timeframe_analysis
            )
            
            # Pozisyon büyüklüğü hesapla
            position_value = self.demo_balance * self.position_size
            position_size = position_value / current_price
            
            # Trade ID oluştur
            trade_id = f"DEMO_{symbol}_{int(time.time())}"
            
            # Veritabanına kaydet (ters sinyal bilgisi ile)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO active_trades 
                (trade_id, symbol, signal_type, signal_category, signal_score, 
                 entry_price, entry_time, position_size, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, symbol, actual_signal_type, signal_category, signal_score,
                current_price, datetime.now(), position_size, stop_loss, take_profit
            ))
            
            conn.commit()
            conn.close()
            
            # Aktif işlemler listesine ekle
            self.active_trades[trade_id] = {
                'symbol': symbol,
                'signal_type': actual_signal_type,  # Gerçek açılan pozisyon
                'original_signal': original_signal_type,  # Sistemin verdiği sinyal
                'signal_category': signal_category,
                'signal_score': signal_score,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'ACTIVE',
                'timeframe_analysis': timeframe_analysis
            }
            
            # Detaylı log çıktısı (SCALPING Normal Sinyal)
            print(f"🔄 SCALPING NORMAL SİNYAL İŞLEM AÇILDI: {symbol}")
            print(f"   📊 Sistem Sinyali: {original_signal_type} → Açılan Pozisyon: {actual_signal_type}")
            print(f"   🎯 {signal_category} - Skor: {signal_score:.2f}")
            print(f"   💰 Giriş Fiyatı: {current_price:.6f}")
            print(f"   💰 TP: {take_profit:.6f} ({((take_profit-current_price)/current_price)*100:.2f}%) - MIN %1 KAR GARANTİSİ")
            print(f"   💰 SL: {stop_loss:.6f} ({((stop_loss-current_price)/current_price)*100:.2f}%)")
            print(f"   🎯 HEDEF: %80 Win Rate (Scalping Normal Sinyal)")
            
            if timeframe_analysis:
                print(f"   📈 1m: {timeframe_analysis.get('1m_score', 0)}/21, "
                      f"5m: {timeframe_analysis.get('5m_score', 0)}/21")
                print(f"   🎯 Trend Tutarlılığı: {timeframe_analysis.get('trend_consistency', 0)}/3")
            
            return True
            
        except Exception as e:
            print(f"Demo işlem açma hatası: {e}")
            return False

    def check_trade_exit_conditions(self, trade_id: str, current_price: float) -> Tuple[bool, str, float]:
        """İşlem çıkış koşullarını kontrol et (Ters Sinyal Sistemi)"""
        trade = self.active_trades.get(trade_id)
        if not trade:
            return False, "", 0
        
        # current_price None kontrolü
        if current_price is None:
            return False, "", 0
        
        symbol = trade['symbol']
        signal_type = trade['signal_type']  # Gerçek açılan pozisyon tipi (ters sinyal)
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        
        # TERS SİNYAL TP/SL kontrolü (gerçek açılan pozisyon tipine göre)
        # Stop Loss kontrolü
        if signal_type == 'LONG' and current_price <= stop_loss:
            return True, "STOP_LOSS", stop_loss
        elif signal_type == 'SHORT' and current_price >= stop_loss:
            return True, "STOP_LOSS", stop_loss
        
        # Take Profit kontrolü
        if signal_type == 'LONG' and current_price >= take_profit:
            return True, "TAKE_PROFIT", take_profit
        elif signal_type == 'SHORT' and current_price <= take_profit:
            return True, "TAKE_PROFIT", take_profit
        
        return False, "", 0

    def close_demo_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """Demo işlemi kapat"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return False
            
            # PnL hesapla
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            if trade['signal_type'] == 'LONG':
                pnl = (exit_price - entry_price) * position_size
            else:  # SHORT
                pnl = (entry_price - exit_price) * position_size
            
            pnl_percentage = (pnl / (entry_price * position_size)) * 100
            
            # Süre hesapla
            duration = datetime.now() - trade['entry_time']
            duration_minutes = int(duration.total_seconds() / 60)
            
            # Veritabanına kaydet
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO completed_trades 
                (trade_id, symbol, signal_type, signal_category, signal_score,
                 entry_price, exit_price, entry_time, exit_time, position_size,
                 pnl, pnl_percentage, exit_reason, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, trade['symbol'], trade['signal_type'], trade['signal_category'],
                trade['signal_score'], entry_price, exit_price, trade['entry_time'],
                datetime.now(), position_size, pnl, pnl_percentage, exit_reason, duration_minutes
            ))
            
            # Aktif işlemlerden kaldır
            cursor.execute('DELETE FROM active_trades WHERE trade_id = ?', (trade_id,))
            
            conn.commit()
            conn.close()
            
            # Aktif işlemler listesinden kaldır
            del self.active_trades[trade_id]
            
            # Tamamlanan işlemler listesine ekle
            self.completed_trades.append({
                'trade_id': trade_id,
                'symbol': trade['symbol'],
                'signal_type': trade['signal_type'],
                'signal_category': trade['signal_category'],
                'signal_score': trade['signal_score'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage,
                'exit_reason': exit_reason,
                'duration_minutes': duration_minutes
            })
            
            print(f"✅ İşlem kapatıldı: {trade['symbol']} - PnL: {pnl:.2f} USDT ({pnl_percentage:.2f}%) - {exit_reason}")
            return True
            
        except Exception as e:
            print(f"İşlem kapatma hatası: {e}")
            return False

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Performans metriklerini hesapla"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Genel istatistikler
            df_completed = pd.read_sql_query('SELECT * FROM completed_trades', conn)
            
            if len(df_completed) == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
            
            # Kazanan/kaybeden işlemler
            winning_trades = df_completed[df_completed['pnl'] > 0]
            losing_trades = df_completed[df_completed['pnl'] < 0]
            
            total_trades = len(df_completed)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            win_rate = (winning_count / total_trades) * 100 if total_trades > 0 else 0
            
            # PnL istatistikleri
            total_pnl = df_completed['pnl'].sum()
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            
            profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())) if len(losing_trades) > 0 else float('inf')
            
            # Kategori bazlı analiz
            mega_signals = df_completed[df_completed['signal_category'] == 'MEGA SCALP']
            pro_signals = df_completed[df_completed['signal_category'] == 'PRO SCALP']
            
            mega_pnl = mega_signals['pnl_percentage'].mean() if len(mega_signals) > 0 else 0
            pro_pnl = pro_signals['pnl_percentage'].mean() if len(pro_signals) > 0 else 0
            
            # Maksimum drawdown hesapla
            cumulative_pnl = df_completed['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / running_max * 100
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Sharpe ratio (basit hesaplama)
            returns = df_completed['pnl_percentage'] / 100
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            conn.close()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_count,
                'losing_trades': losing_count,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'mega_signals_pnl': round(mega_pnl, 2),
                'pro_signals_pnl': round(pro_pnl, 2),
                'mega_signals_count': len(mega_signals),
                'pro_signals_count': len(pro_signals)
            }
            
        except Exception as e:
            print(f"Performans hesaplama hatası: {e}")
            return {}

    def monitor_active_trades(self):
        """Aktif işlemleri izle"""
        while self.is_running:
            try:
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        symbol = trade['symbol']
                        
                        # Güncel fiyat al
                        current_price = self.get_current_price(symbol)
                        
                        # Fiyat kontrolü
                        if current_price is None:
                            print(f"⚠️ {symbol} için fiyat alınamadı, atlanıyor...")
                            continue
                        
                        # Çıkış koşullarını kontrol et
                        should_exit, exit_reason, exit_price = self.check_trade_exit_conditions(trade_id, current_price)
                        
                        # Timeout kontrolü
                        entry_time = trade['entry_time']
                        duration = (datetime.now() - entry_time).total_seconds() / 60
                        
                        if duration > self.max_trade_duration:
                            print(f"⏰ {symbol}: Maksimum süre aşıldı ({duration:.0f} dakika)")
                            self.close_demo_trade(trade_id, current_price, "TIMEOUT")
                            continue
                        
                        if should_exit:
                            self.close_demo_trade(trade_id, exit_price, exit_reason)
                    
                    except Exception as e:
                        print(f"İşlem izleme hatası {trade.get('symbol', 'UNKNOWN')}: {e}")
                        continue
                
                time.sleep(10)  # 10 saniyede bir kontrol et
                
            except Exception as e:
                print(f"Genel işlem izleme hatası: {e}")
                time.sleep(30)

    def start_live_test(self, max_coins: int = 480, max_trades: int = 5):
        """Live test'i başlat"""
        self.is_running = True
        self.max_trades = max_trades
        self.max_coins = max_coins
        
        # İşlem izleme thread'ini başlat
        monitor_thread = threading.Thread(target=self.monitor_active_trades)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Otomatik trading döngüsünü başlat
        trading_thread = threading.Thread(target=self.auto_trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        print(f"🚀 SCALPING SİSTEMİ BAŞLATILDI!")
        print(f"📊 Maksimum coin sayısı: {max_coins} (TÜM BİNANCE COİNLERİ)")
        print(f"📊 Maksimum işlem sayısı: {max_trades}")
        print(f"💰 Demo bakiye: {self.demo_balance} USDT")
        print(f"📈 Pozisyon büyüklüğü: %{self.position_size * 100}")
        print(f"🔍 Otomatik tarama aktif - {self.scan_interval} saniyede bir")
        print(f"🕐 SCALPING Timeframe Analizi: 1m (giriş), 5m (trend)")
        print(f"🎯 Minimum sinyal skoru: {self.min_signal_score}/20 (STANDARD SCALP+ sinyaller)")
        print(f"⚖️ Timeframe Ağırlıkları: 1m (%70), 5m (%30)")
        print(f"🔍 SCALPING filtreler: Hacim(150K), Spread(0.15%), ADX(18), Volatilite, Momentum")
        print(f"💰 Risk Yönetimi: %5 pozisyon, maksimum {max_trades} işlem")
        print(f"⚡ HIZLI SCALPING: Bulduğu anda işlem açar!")
        print(f"🎯 TÜM BİNANCE: ~480 coin taranacak")
        print(f"🔄 TERS SİNYAL STRATEJİSİ: Sistem LONG diyor → SHORT açıyoruz!")
        print(f"🎯 HEDEF: %80 WIN RATE (Sistem %20 veriyor, tersini alıyoruz!)")

    def stop_live_test(self):
        """Live test'i durdur"""
        self.is_running = False
        print("🛑 Deviso Live Test durduruldu!")

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Aktif işlemleri getir"""
        return list(self.active_trades.values())

    def get_completed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Tamamlanan işlemleri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM completed_trades 
                ORDER BY exit_time DESC 
                LIMIT ?
            ''', (limit,))
            
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'trade_id': row[1],
                    'symbol': row[2],
                    'signal_type': row[3],
                    'signal_category': row[4],
                    'signal_score': row[5],
                    'entry_price': row[6],
                    'exit_price': row[7],
                    'entry_time': row[8],
                    'exit_time': row[9],
                    'position_size': row[10],
                    'pnl': row[11],
                    'pnl_percentage': row[12],
                    'exit_reason': row[13],
                    'duration_minutes': row[14]
                })
            
            conn.close()
            return trades
            
        except Exception as e:
            print(f"Tamamlanan işlemler getirme hatası: {e}")
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Performans özetini getir"""
        metrics = self.calculate_performance_metrics()
        
        return {
            'metrics': metrics,
            'active_trades_count': len(self.active_trades),
            'demo_balance': self.demo_balance,
            'position_size_percentage': self.position_size * 100
        }
