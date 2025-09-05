import pandas as pd
import numpy as np
import time
import threading
import sqlite3
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Any
from binance.client import Client
import talib
warnings.filterwarnings('ignore')

class DevisoSuperSignalScanner:
    def __init__(self):
        # Binance client başlat (API key olmadan da çalışır)
        self.client = None
        try:
            self.client = Client("", "")
            # Test bağlantısı - timeout ile
            import requests
            requests.adapters.DEFAULT_RETRIES = 1
            # Ping yerine basit bir istek yap
            self.client.get_exchange_info()
        except Exception as e:
            print(f"Binance client başlatma hatası: {e}")
            # Fallback olarak None bırak
            self.client = None
        self.db_path = 'super_signal.db'
        self.init_database()
        self.is_running = False
        self.scan_results = []
        self.current_cycle = 0
        self.ema_filtered_coins = []
        
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Eski tabloları sil
        cursor.execute('DROP TABLE IF EXISTS ema_filtered_coins')
        cursor.execute('DROP TABLE IF EXISTS super_signals')
        
        # EMA filtrelenmiş coinler tablosu
        cursor.execute('''
            CREATE TABLE ema_filtered_coins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                ratio REAL,
                ema200 REAL,
                current_price REAL,
                timestamp TIMESTAMP,
                timeframe TEXT
            )
        ''')
        
        # Super sinyaller tablosu
        cursor.execute('''
            CREATE TABLE super_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal TEXT,
                bars_ago INTEGER,
                price REAL,
                current_price REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                timestamp TIMESTAMP,
                timeframe TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def get_klines_data(self, symbol: str, interval: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Binance'den kline verisi al"""
        try:
            # Binance interval formatını dönüştür
            interval_map = {
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_5MINUTE)
            
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
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
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
    
    def check_ema_condition(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """EMA200 koşullarını kontrol et"""
        try:
            if len(df) < 200:
                return {'signal': False, 'reason': 'Yetersiz veri'}
            
            # EMA200 hesapla
            df['ema200'] = self.calculate_ema(df, 200)
            
            # Son değerler
            current_price = df['close'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            
            # NaN kontrolü
            if pd.isna(ema200):
                return {'signal': False, 'reason': 'EMA200 hesaplanamadı'}
            
            # Ratio hesapla
            ratio = ((current_price - ema200) / ema200) * 100
            
            # Sinyal koşulları
            if ratio >= 0:  # Fiyat EMA200'ün üstünde
                return {
                    'signal': True,
                    'ratio': ratio,
                    'ema200': ema200,
                    'current_price': current_price
                }
            else:
                return {'signal': False, 'reason': f'Fiyat EMA200 altında (Ratio: {ratio:.2f}%)'}
                
        except Exception as e:
            print(f"EMA koşul kontrolü hatası {symbol}: {e}")
            return {'signal': False, 'reason': f'Hata: {e}'}
    
    def check_rsi_macd_condition(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """RSI ve MACD sinyallerini kontrol et"""
        try:
            if len(df) < 50:
                return []
            
            # RSI hesapla
            df['rsi'] = self.calculate_rsi(df, 14)
            
            # MACD hesapla
            df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df)
            
            signals = []
            current_price = df['close'].iloc[-1]
            
            # Son 20 barı kontrol et
            for i in range(min(20, len(df) - 1)):
                idx = -(i + 1)
                
                if pd.isna(df['rsi'].iloc[idx]) or pd.isna(df['macd'].iloc[idx]):
                    continue
                
                rsi = df['rsi'].iloc[idx]
                macd = df['macd'].iloc[idx]
                macd_signal = df['macd_signal'].iloc[idx]
                macd_histogram = df['macd_histogram'].iloc[idx]
                price = df['close'].iloc[idx]
                
                # C20L Sinyali: RSI < 30 ve MACD > MACD Signal
                if rsi < 30 and macd > macd_signal:
                    signal_info = {
                        'symbol': symbol,
                        'signal': 'C20L',
                        'bars_ago': i + 1,
                        'price': round(price, 6),
                        'current_price': round(current_price, 6),
                        'rsi': round(rsi, 2),
                        'macd': round(macd, 6),
                        'macd_signal': round(macd_signal, 6),
                        'macd_histogram': round(macd_histogram, 6),
                        'timestamp': datetime.now(),
                        'timeframe': timeframe
                    }
                    signals.append(signal_info)
                    print(f"🔥 C20L Sinyali: {symbol} - RSI: {rsi:.2f}, MACD: {macd:.6f}")
                    break
                
                # C20S Sinyali: RSI > 70 ve MACD < MACD Signal
                elif rsi > 70 and macd < macd_signal:
                    signal_info = {
                        'symbol': symbol,
                        'signal': 'C20S',
                        'bars_ago': i + 1,
                        'price': round(price, 6),
                        'current_price': round(current_price, 6),
                        'rsi': round(rsi, 2),
                        'macd': round(macd, 6),
                        'macd_signal': round(macd_signal, 6),
                        'macd_histogram': round(macd_histogram, 6),
                        'timestamp': datetime.now(),
                        'timeframe': timeframe
                    }
                    signals.append(signal_info)
                    print(f"🔥 C20S Sinyali: {symbol} - RSI: {rsi:.2f}, MACD: {macd:.6f}")
                    break
                
                # M5L Sinyali: MACD > MACD Signal ve MACD Histogram > 0
                elif macd > macd_signal and macd_histogram > 0:
                    signal_info = {
                        'symbol': symbol,
                        'signal': 'M5L',
                        'bars_ago': i + 1,
                        'price': round(price, 6),
                        'current_price': round(current_price, 6),
                        'rsi': round(rsi, 2),
                        'macd': round(macd, 6),
                        'macd_signal': round(macd_signal, 6),
                        'macd_histogram': round(macd_histogram, 6),
                        'timestamp': datetime.now(),
                        'timeframe': timeframe
                    }
                    signals.append(signal_info)
                    print(f"🔥 M5L Sinyali: {symbol} - MACD: {macd:.6f}, Histogram: {macd_histogram:.6f}")
                    break
                
                # M5S Sinyali: MACD < MACD Signal ve MACD Histogram < 0
                elif macd < macd_signal and macd_histogram < 0:
                    signal_info = {
                        'symbol': symbol,
                        'signal': 'M5S',
                        'bars_ago': i + 1,
                        'price': round(price, 6),
                        'current_price': round(current_price, 6),
                        'rsi': round(rsi, 2),
                        'macd': round(macd, 6),
                        'macd_signal': round(macd_signal, 6),
                        'macd_histogram': round(macd_histogram, 6),
                        'timestamp': datetime.now(),
                        'timeframe': timeframe
                    }
                    signals.append(signal_info)
                    print(f"🔥 M5S Sinyali: {symbol} - MACD: {macd:.6f}, Histogram: {macd_histogram:.6f}")
                    break
            
            return signals
            
        except Exception as e:
            print(f"RSI/MACD sinyal kontrolü hatası {symbol}: {e}")
            return []
    
    def calculate_ranking_score(self, signal_info: Dict[str, Any]) -> float:
        """Sinyal için sıralama skoru hesapla (Momentum, Hacim, Fiyat Değişimi, Super Signal Kalitesi)"""
        try:
            symbol = signal_info['symbol']
            current_price = signal_info['current_price']
            
            # 1. MOMENTUM SKORU (0-30 puan)
            momentum_score = 0
            try:
                df_5m = self.get_klines_data(symbol, '5m', 20)
                df_15m = self.get_klines_data(symbol, '15m', 20)
                
                if len(df_5m) >= 10 and len(df_15m) >= 10:
                    # 5m momentum (son 3-6 periyot)
                    price_change_5m_3 = ((current_price - df_5m['close'].iloc[-3]) / df_5m['close'].iloc[-3]) * 100
                    price_change_5m_6 = ((current_price - df_5m['close'].iloc[-6]) / df_5m['close'].iloc[-6]) * 100
                    
                    # 15m momentum (son 2-4 periyot)
                    price_change_15m_2 = ((current_price - df_15m['close'].iloc[-2]) / df_15m['close'].iloc[-2]) * 100
                    price_change_15m_4 = ((current_price - df_15m['close'].iloc[-4]) / df_15m['close'].iloc[-4]) * 100
                    
                    # Momentum puanları
                    if signal_info['signal'] in ['C20L', 'M5L']:  # LONG sinyaller
                        # LONG için pozitif momentum
                        if price_change_5m_3 > 0.3: momentum_score += 8
                        if price_change_5m_6 > 0.8: momentum_score += 7
                        if price_change_15m_2 > 0.5: momentum_score += 8
                        if price_change_15m_4 > 1.2: momentum_score += 7
                    else:  # SHORT sinyaller
                        # SHORT için negatif momentum
                        if price_change_5m_3 < -0.3: momentum_score += 8
                        if price_change_5m_6 < -0.8: momentum_score += 7
                        if price_change_15m_2 < -0.5: momentum_score += 8
                        if price_change_15m_4 < -1.2: momentum_score += 7
            except Exception as e:
                print(f"Momentum hesaplama hatası {symbol}: {e}")
            
            # 2. HACİM SKORU (0-25 puan)
            volume_score = 0
            try:
                df_5m = self.get_klines_data(symbol, '5m', 20)
                if len(df_5m) >= 20:
                    current_volume = df_5m['volume'].iloc[-1]
                    avg_volume = df_5m['volume'].rolling(window=20).mean().iloc[-1]
                    
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
                df_5m = self.get_klines_data(symbol, '5m', 20)
                if len(df_5m) >= 20:
                    # ATR hesapla
                    atr = talib.ATR(df_5m['high'].values, df_5m['low'].values, df_5m['close'].values, timeperiod=14)
                    current_atr = atr[-1] if not pd.isna(atr[-1]) else 0
                    
                    # Volatilite hesapla
                    volatility = df_5m['close'].pct_change().std() * 100
                    
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
            
            # 4. SUPER SIGNAL KALİTESİ SKORU (0-20 puan)
            quality_score = 0
            try:
                # EMA200 kalitesi (0-10 puan)
                if 'ema200' in signal_info:
                    ema200 = signal_info['ema200']
                    price_distance = abs(current_price - ema200) / ema200 * 100
                    if price_distance <= 0.5: quality_score += 10  # Çok yakın
                    elif price_distance <= 1.0: quality_score += 8   # Yakın
                    elif price_distance <= 2.0: quality_score += 5   # Orta
                    elif price_distance <= 5.0: quality_score += 2   # Uzak
                
                # RSI/MACD kalitesi (0-10 puan)
                if 'rsi' in signal_info and 'macd' in signal_info:
                    rsi = signal_info['rsi']
                    macd = signal_info['macd']
                    macd_signal = signal_info.get('macd_signal', 0)
                    macd_histogram = signal_info.get('macd_histogram', 0)
                    
                    # RSI kalitesi
                    if signal_info['signal'] in ['C20L', 'M5L']:  # LONG sinyaller
                        if rsi <= 20: quality_score += 5  # Çok güçlü oversold
                        elif rsi <= 30: quality_score += 3   # Güçlü oversold
                        elif rsi <= 40: quality_score += 1   # Orta oversold
                    else:  # SHORT sinyaller
                        if rsi >= 80: quality_score += 5  # Çok güçlü overbought
                        elif rsi >= 70: quality_score += 3   # Güçlü overbought
                        elif rsi >= 60: quality_score += 1   # Orta overbought
                    
                    # MACD kalitesi
                    if signal_info['signal'] in ['C20L', 'M5L']:  # LONG sinyaller
                        if macd > macd_signal and macd_histogram > 0: quality_score += 5  # Güçlü bullish
                        elif macd > macd_signal: quality_score += 3  # Orta bullish
                        elif macd_histogram > 0: quality_score += 1  # Hafif bullish
                    else:  # SHORT sinyaller
                        if macd < macd_signal and macd_histogram < 0: quality_score += 5  # Güçlü bearish
                        elif macd < macd_signal: quality_score += 3  # Orta bearish
                        elif macd_histogram < 0: quality_score += 1  # Hafif bearish
                        
            except Exception as e:
                print(f"Super Signal kalite hesaplama hatası {symbol}: {e}")
            
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

    def scan_coins(self, max_coins: int = 30, timeframe: str = '5m') -> List[Dict[str, Any]]:
        """Coinleri tara ve sırala (Sıralama Algoritması ile)"""
        try:
            # Tüm sembolleri al
            exchange_info = self.client.get_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                if symbol.endswith('USDT') and symbol_info['status'] == 'TRADING':
                    symbols.append(symbol)
            
            print(f"📊 Toplam {len(symbols)} sembol bulundu, tümü taranacak...")
            print(f"🚀 SUPER SIGNAL TARAMA: En iyi fırsatlar sıralanacak!")
            print(f"📈 SIRALAMA ALGORİTMASI: Momentum + Hacim + Fiyat Değişimi + Super Signal Kalitesi")
            
            all_signals = []
            scanned_count = 0
            
            # İLK AŞAMA: EMA200 Filtresi
            print("🔍 AŞAMA 1: EMA200 Filtresi başlatıldı...")
            ema_filtered = []
            
            for symbol in symbols[:max_coins]:
                try:
                    scanned_count += 1
                    
                    # Her 10 coin'de bir ilerleme göster
                    if scanned_count % 10 == 0:
                        print(f"🔍 Tarama ilerlemesi: {scanned_count}/{min(max_coins, len(symbols))} - Bulunan EMA: {len(ema_filtered)}")
                    
                    # Kline verisi al
                    df = self.get_klines_data(symbol, timeframe, 200)
                    
                    if len(df) < 200:
                        continue
                    
                    # EMA koşullarını kontrol et
                    result = self.check_ema_condition(df, symbol, timeframe)
                    
                    if result['signal']:
                        ema_info = {
                            'symbol': symbol,
                            'ratio': round(result['ratio'], 2),
                            'ema200': round(result['ema200'], 6),
                            'current_price': round(result['current_price'], 6),
                            'timestamp': datetime.now(),
                            'timeframe': timeframe
                        }
                        ema_filtered.append(ema_info)
                        
                        # Veritabanına kaydet
                        self.save_ema_filtered_to_db(ema_info)
                    
                    # Rate limit için bekle
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"EMA tarama hatası {symbol}: {e}")
                    continue
            
            print(f"✅ EMA200 Filtresi tamamlandı! {len(ema_filtered)} coin bulundu.")
            
            # İKİNCİ AŞAMA: RSI/MACD Taraması
            print("🔍 AŞAMA 2: RSI/MACD Taraması başlatıldı...")
            
            for ema_coin in ema_filtered:
                symbol = ema_coin['symbol']
                
                try:
                    # Kline verisi al
                    df = self.get_klines_data(symbol, timeframe, 100)
                    
                    if len(df) < 50:
                        continue
                    
                    # RSI/MACD sinyallerini kontrol et
                    signals = self.check_rsi_macd_condition(df, symbol, timeframe)
                    
                    for signal in signals:
                        # EMA bilgilerini ekle
                        signal['ema200'] = ema_coin['ema200']
                        signal['ratio'] = ema_coin['ratio']
                        
                        # Sıralama skorunu hesapla
                        ranking_score = self.calculate_ranking_score(signal)
                        signal['ranking_score'] = ranking_score
                        
                        all_signals.append(signal)
                    
                    # Rate limit için bekle
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"RSI/MACD tarama hatası {symbol}: {e}")
                    continue
            
            print(f"✅ AŞAMA 2 TAMAMLANDI: {len(all_signals)} sinyal bulundu")
            
            # ÜÇÜNCÜ AŞAMA: Sinyalleri sırala
            if all_signals:
                print(f"📊 AŞAMA 3: Sinyaller sıralanıyor...")
                
                # Sıralama skoruna göre azalan sırada sırala
                all_signals.sort(key=lambda x: x['ranking_score'], reverse=True)
                
                print(f"🏆 EN İYİ 10 SUPER SİNYAL:")
                for i, signal in enumerate(all_signals[:10]):
                    ranking_details = signal['ranking_details']
                    print(f"   {i+1}. {signal['symbol']} - {signal['signal']}")
                    print(f"      📊 Sıralama Skoru: {signal['ranking_score']:.1f}")
                    print(f"      📈 Momentum: {ranking_details['momentum_score']:.1f}, Hacim: {ranking_details['volume_score']:.1f}")
                    print(f"      💰 Fiyat Değişimi: {ranking_details['price_change_score']:.1f}, Super Signal Kalitesi: {ranking_details['quality_score']:.1f}")
                    print(f"      📊 RSI: {signal['rsi']:.1f}, MACD: {signal['macd']:.6f}, EMA200 Ratio: {signal['ratio']:.2f}%")
                
                # Veritabanına kaydet
                for signal in all_signals:
                    self.save_super_signal_to_db(signal)
                
                # Sonuçları güncelle
                self.ema_filtered_coins = ema_filtered
                self.scan_results = all_signals
                
                print(f"🎉 SUPER SIGNAL SIRALAMA ALGORİTMASI TAMAMLANDI!")
                print(f"   📊 Taranan coin: {scanned_count}/{min(max_coins, len(symbols))}")
                print(f"   🔥 Bulunan sinyal: {len(all_signals)}")
                
                return all_signals
            else:
                print(f"❌ Hiç sinyal bulunamadı!")
                return []
            
        except Exception as e:
            print(f"Genel tarama hatası: {e}")
            return []
    
    def save_ema_filtered_to_db(self, ema_info: Dict[str, Any]):
        """EMA filtrelenmiş coini veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ema_filtered_coins 
                (symbol, ratio, ema200, current_price, timestamp, timeframe)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                ema_info['symbol'],
                ema_info['ratio'],
                ema_info['ema200'],
                ema_info['current_price'],
                ema_info['timestamp'],
                ema_info['timeframe']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"EMA filtrelenmiş coin kaydetme hatası: {e}")
    
    def save_super_signal_to_db(self, signal_info: Dict[str, Any]):
        """Super sinyali veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO super_signals 
                (symbol, signal, bars_ago, price, current_price, rsi, macd, macd_signal, macd_histogram, timestamp, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_info['symbol'],
                signal_info['signal'],
                signal_info['bars_ago'],
                signal_info['price'],
                signal_info['current_price'],
                signal_info['rsi'],
                signal_info['macd'],
                signal_info['macd_signal'],
                signal_info['macd_histogram'],
                signal_info['timestamp'],
                signal_info['timeframe']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Super sinyal kaydetme hatası: {e}")
    
    def start_super_signal_scan(self, max_coins: int = 30, timeframe: str = '5m'):
        """Super Signal taramasını başlat"""
        self.is_running = True
        self.max_coins = max_coins
        self.timeframe = timeframe
        
        def scan_loop():
            cycle_count = 0
            while self.is_running:
                try:
                    cycle_count += 1
                    self.current_cycle = cycle_count
                    print(f"🔄 Super Signal Döngü {cycle_count} Başlatıldı - {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Coinleri tara
                    self.scan_coins(max_coins, timeframe)
                    
                    print(f"⏳ {timeframe} sonra yeni döngü başlayacak...")
                    
                    # Timeframe'e göre bekleme süresi
                    if timeframe == '5m':
                        time.sleep(300)  # 5 dakika
                    elif timeframe == '15m':
                        time.sleep(900)  # 15 dakika
                    elif timeframe == '1h':
                        time.sleep(3600)  # 1 saat
                    else:
                        time.sleep(300)  # Varsayılan 5 dakika
                    
                except Exception as e:
                    print(f"Super Signal tarama döngüsü hatası: {e}")
                    time.sleep(60)
        
        # Tarama thread'ini başlat
        scan_thread = threading.Thread(target=scan_loop)
        scan_thread.daemon = True
        scan_thread.start()
        
        print(f"🚀 Super Signal Scanner başlatıldı!")
        print(f"📊 Maksimum Coin: {max_coins}")
        print(f"⏰ Timeframe: {timeframe}")
    
    def stop_scan(self):
        """Taramayı durdur"""
        self.is_running = False
        print("🛑 Super Signal Scanner durduruldu!")
    
    def get_latest_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Son sinyalleri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Super sinyalleri ve EMA ratio bilgisini birleştir
            cursor.execute('''
                SELECT s.*, e.ratio as ema_ratio 
                FROM super_signals s
                LEFT JOIN ema_filtered_coins e ON s.symbol = e.symbol
                ORDER BY s.timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            signals = []
            for row in cursor.fetchall():
                signals.append({
                    'symbol': row[1],
                    'signal': row[2],
                    'bars_ago': row[3],
                    'price': row[4],
                    'current_price': row[5],
                    'rsi': row[6],
                    'macd': row[7],
                    'macd_signal': row[8],
                    'macd_histogram': row[9],
                    'timestamp': row[10],
                    'timeframe': row[11],
                    'ema_ratio': row[12] if row[12] is not None else 0.0
                })
            
            conn.close()
            return signals
            
        except Exception as e:
            print(f"Sinyal getirme hatası: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Toplam sinyal sayısı
            cursor.execute('SELECT COUNT(*) FROM super_signals')
            total_signals = cursor.fetchone()[0]
            
            # Bugünkü sinyal sayısı
            today = datetime.now().date()
            cursor.execute('SELECT COUNT(*) FROM super_signals WHERE DATE(timestamp) = ?', (today,))
            today_signals = cursor.fetchone()[0]
            
            # Sinyal türü dağılımı
            cursor.execute('SELECT signal, COUNT(*) FROM super_signals GROUP BY signal')
            signal_distribution = dict(cursor.fetchall())
            
            # EMA filtrelenmiş coin sayısı
            cursor.execute('SELECT COUNT(*) FROM ema_filtered_coins')
            ema_filtered_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_signals': total_signals,
                'today_signals': today_signals,
                'signal_distribution': signal_distribution,
                'ema_filtered_count': ema_filtered_count,
                'current_cycle': self.current_cycle,
                'is_running': self.is_running
            }
            
        except Exception as e:
            print(f"İstatistik getirme hatası: {e}")
            return {}

def main():
    scanner = DevisoSuperSignalScanner()
    
    print("🚀 Deviso Super Signal Scanner Test")
    print("=" * 50)
    
    # Test taraması
    results = scanner.scan_coins(max_coins=10, timeframe='5m')
    
    if results:
        print(f"\n🎯 {len(results)} coin bulundu:")
        for result in results:
            print(f"  • {result['symbol']} - Ratio: {result['ratio']}%")
        
        # İlk coin için RSI/MACD test
        if results:
            test_symbol = f"{results[0]['symbol']}/USDT:USDT"
            print(f"\n🔍 RSI/MACD test: {test_symbol}")
            # The original code had rsi_macd_scan_symbol which is removed.
            # This part of the test will need to be adapted or removed if no replacement is provided.
            # For now, we'll just print a message indicating the function is removed.
            print("RSI/MACD tarama fonksiyonu kaldırıldı.")
            # signal = scanner.rsi_macd_scan_symbol(test_symbol, '5m')
            
            # if signal:
            #     print(f"  ✅ Sinyal bulundu: {signal['signal']} ({signal['category']})")
            # else:
            #     print("  ❌ Sinyal bulunamadı")
    else:
        print("❌ Test taramasında coin bulunamadı")

if __name__ == "__main__":
    main()
