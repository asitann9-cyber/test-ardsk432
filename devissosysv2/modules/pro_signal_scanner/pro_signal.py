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

class DevisoProSignalScanner:
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
        self.db_path = 'pro_signal.db'
        self.init_database()
        self.is_running = False
        self.scan_results = []
        self.current_cycle = 0
        
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Eski tabloyu sil
        cursor.execute('DROP TABLE IF EXISTS pro_signals')
        
        cursor.execute('''
            CREATE TABLE pro_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal_type TEXT,
                signal_category TEXT,
                signal_score REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                ema200 REAL,
                ema50 REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                volatility REAL,
                momentum REAL,
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
    
    def calculate_signal_score(self, symbol: str, timeframe: str = '5m') -> Dict[str, Any]:
        """Sinyal skorunu hesapla"""
        try:
            df = self.get_klines_data(symbol, timeframe, 100)
            if len(df) < 50:
                return {'score': 0, 'signals': []}

            # EMA200 ve EMA50 hesapla
            df['ema200'] = self.calculate_ema(df, 200)
            df['ema50'] = self.calculate_ema(df, 50)
            
            # RSI hesapla
            df['rsi'] = self.calculate_rsi(df, 14)
            
            # MACD hesapla
            df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df)
            
            # Son değerler
            current_price = df['close'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_histogram = df['macd_histogram'].iloc[-1]
            
            # Volatilite hesapla
            volatility = df['close'].pct_change().std() * 100
            
            # Momentum hesapla (5 bar öncesi ile karşılaştırma)
            momentum = ((current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
            
            # Sinyal puanları
            score = 0
            signals = []
            
            # EMA200 analizi (0-3 puan)
            if current_price > ema200:
                score += 1
                signals.append("EMA200_ABOVE")
            if current_price > ema50:
                score += 1
                signals.append("EMA50_ABOVE")
            if ema50 > ema200:
                score += 1
                signals.append("EMA50_ABOVE_EMA200")
            
            # RSI analizi (0-2 puan)
            if 30 <= rsi <= 70:
                score += 1
                signals.append("RSI_NEUTRAL")
            if rsi > 50:
                score += 1
                signals.append("RSI_BULLISH")
            
            # MACD analizi (0-2 puan)
            if macd > macd_signal:
                score += 1
                signals.append("MACD_BULLISH")
            if macd_histogram > 0:
                score += 1
                signals.append("MACD_HISTOGRAM_POSITIVE")
            
            # Volatilite analizi (0-1 puan)
            if 2 <= volatility <= 8:
                score += 1
                signals.append("VOLATILITY_OPTIMAL")
            
            # Momentum analizi (0-2 puan)
            if momentum > 0:
                score += 1
                signals.append("MOMENTUM_POSITIVE")
            if momentum > 1:
                score += 1
                signals.append("MOMENTUM_STRONG")
            
            # Sinyal kategorisi belirle
            if score >= 9:
                category = "MEGA SIGNAL"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            elif score >= 7:
                category = "PRO SIGNAL"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            elif score >= 5:
                category = "STANDARD SIGNAL"
                signal_type = "LONG" if current_price > ema200 else "SHORT"
            else:
                category = "WEAK SIGNAL"
                signal_type = "NEUTRAL"
            
            return {
                'score': score,
                'category': category,
                'signal_type': signal_type,
                'signals': signals,
                'current_price': current_price,
                'ema200': ema200,
                'ema50': ema50,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_histogram,
                'volatility': volatility,
                'momentum': momentum
            }
            
        except Exception as e:
            print(f"Sinyal hesaplama hatası {symbol}: {e}")
            return {'score': 0, 'signals': []}

    def calculate_ranking_score(self, signal_info: Dict[str, Any]) -> float:
        """Sinyal için sıralama skoru hesapla (Momentum, Hacim, Fiyat Değişimi, Pro Signal Kalitesi)"""
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
                    if signal_info['signal_type'] == 'LONG':
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
            
            # 4. PRO SIGNAL KALİTESİ SKORU (0-20 puan)
            quality_score = 0
            try:
                # Sinyal skoru kalitesi (0-10 puan)
                signal_score = signal_info['signal_score']
                if signal_score >= 8: quality_score += 10  # Çok yüksek kalite
                elif signal_score >= 6: quality_score += 8   # Yüksek kalite
                elif signal_score >= 5: quality_score += 5   # Orta kalite
                
                # Teknik indikatör kalitesi (0-10 puan)
                rsi = signal_info['rsi']
                macd = signal_info['macd']
                macd_signal = signal_info['macd_signal']
                macd_histogram = signal_info['macd_histogram']
                ema200 = signal_info['ema200']
                ema50 = signal_info['ema50']
                
                # EMA kalitesi
                if signal_info['signal_type'] == 'LONG':
                    if current_price > ema200 and ema50 > ema200: quality_score += 5  # Güçlü bullish trend
                    elif current_price > ema200: quality_score += 3  # Orta bullish trend
                    elif current_price > ema50: quality_score += 1  # Hafif bullish trend
                else:  # SHORT
                    if current_price < ema200 and ema50 < ema200: quality_score += 5  # Güçlü bearish trend
                    elif current_price < ema200: quality_score += 3  # Orta bearish trend
                    elif current_price < ema50: quality_score += 1  # Hafif bearish trend
                
                # RSI/MACD kalitesi
                if signal_info['signal_type'] == 'LONG':
                    if rsi <= 30 and macd > macd_signal: quality_score += 5  # Güçlü oversold + bullish MACD
                    elif rsi <= 40 or macd > macd_signal: quality_score += 3  # Orta kalite
                    elif rsi <= 50 or macd_histogram > 0: quality_score += 1  # Hafif kalite
                else:  # SHORT
                    if rsi >= 70 and macd < macd_signal: quality_score += 5  # Güçlü overbought + bearish MACD
                    elif rsi >= 60 or macd < macd_signal: quality_score += 3  # Orta kalite
                    elif rsi >= 50 or macd_histogram < 0: quality_score += 1  # Hafif kalite
                        
            except Exception as e:
                print(f"Pro Signal kalite hesaplama hatası {symbol}: {e}")
            
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

    def scan_coins(self, max_coins: int = 30) -> List[Dict[str, Any]]:
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
            print(f"🚀 PRO SIGNAL TARAMA: En iyi fırsatlar sıralanacak!")
            print(f"📈 SIRALAMA ALGORİTMASI: Momentum + Hacim + Fiyat Değişimi + Pro Signal Kalitesi")
            
            all_signals = []
            scanned_count = 0
            
            # İLK AŞAMA: Tüm coinleri tara ve sinyalleri topla
            print(f"🔍 AŞAMA 1: Tüm coinler taranıyor ve sinyaller toplanıyor...")
            
            for symbol in symbols[:max_coins]:
                try:
                    scanned_count += 1
                    
                    # Her 10 coin'de bir ilerleme göster
                    if scanned_count % 10 == 0:
                        print(f"🔍 Tarama ilerlemesi: {scanned_count}/{min(max_coins, len(symbols))} - Bulunan sinyal: {len(all_signals)}")
                    
                    # Sinyal skorunu hesapla
                    signal_data = self.calculate_signal_score(symbol, '5m')
                    
                    if signal_data['score'] >= 5:  # Minimum 5 puan
                        signal_info = {
                            'symbol': symbol,
                            'signal_type': signal_data['signal_type'],
                            'signal_category': signal_data['category'],
                            'signal_score': signal_data['score'],
                            'current_price': signal_data['current_price'],
                            'ema200': signal_data['ema200'],
                            'ema50': signal_data['ema50'],
                            'rsi': signal_data['rsi'],
                            'macd': signal_data['macd'],
                            'macd_signal': signal_data['macd_signal'],
                            'macd_histogram': signal_data['macd_histogram'],
                            'volatility': signal_data['volatility'],
                            'momentum': signal_data['momentum'],
                            'signals': signal_data['signals'],
                            'timestamp': datetime.now(),
                            'timeframe': '5m'
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
                
                print(f"🏆 EN İYİ 10 PRO SİNYAL:")
                for i, signal in enumerate(all_signals[:10]):
                    ranking_details = signal['ranking_details']
                    print(f"   {i+1}. {signal['symbol']} - {signal['signal_category']}")
                    print(f"      📊 Sıralama Skoru: {signal['ranking_score']:.1f}")
                    print(f"      📈 Momentum: {ranking_details['momentum_score']:.1f}, Hacim: {ranking_details['volume_score']:.1f}")
                    print(f"      💰 Fiyat Değişimi: {ranking_details['price_change_score']:.1f}, Pro Signal Kalitesi: {ranking_details['quality_score']:.1f}")
                    print(f"      📊 Sinyal Skoru: {signal['signal_score']}/10, RSI: {signal['rsi']:.1f}, MACD: {signal['macd']:.6f}")
                
                # Veritabanına kaydet
                for signal in all_signals:
                    self.save_signal_to_db(signal)
                
                print(f"🎉 PRO SIGNAL SIRALAMA ALGORİTMASI TAMAMLANDI!")
                print(f"   📊 Taranan coin: {scanned_count}/{min(max_coins, len(symbols))}")
                print(f"   🔥 Bulunan sinyal: {len(all_signals)}")
                
                return all_signals
            else:
                print(f"❌ Hiç sinyal bulunamadı!")
                return []
            
        except Exception as e:
            print(f"Genel tarama hatası: {e}")
            return []
    
    def save_signal_to_db(self, signal_info: Dict[str, Any]):
        """Sinyali veritabanına kaydet"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pro_signals 
                (symbol, signal_type, signal_category, signal_score, entry_price, stop_loss, take_profit, ema200, ema50, rsi, macd, macd_signal, macd_histogram, volatility, momentum, timestamp, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_info['symbol'],
                signal_info['signal_type'],
                signal_info['signal_category'],
                signal_info['signal_score'],
                signal_info['current_price'],
                0,  # stop_loss - hesaplanacak
                0,  # take_profit - hesaplanacak
                signal_info['ema200'],
                signal_info['ema50'],
                signal_info['rsi'],
                signal_info['macd'],
                signal_info['macd_signal'],
                signal_info['macd_histogram'],
                signal_info['volatility'],
                signal_info['momentum'],
                signal_info['timestamp'],
                signal_info['timeframe']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Veritabanı kaydetme hatası: {e}")
    
    def start_pro_signal_scan(self, max_coins: int = 30):
        """Pro Signal taramasını başlat"""
        self.is_running = True
        self.max_coins = max_coins
        
        def scan_loop():
            cycle_count = 0
            while self.is_running:
                try:
                    cycle_count += 1
                    self.current_cycle = cycle_count
                    print(f"🔄 Pro Signal Döngü {cycle_count} Başlatıldı - {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Coinleri tara
                    self.scan_results = self.scan_coins(max_coins)
                    
                    print(f"⏳ 5 dakika sonra yeni döngü başlayacak...")
                    time.sleep(300)  # 5 dakika bekle
                    
                except Exception as e:
                    print(f"Pro Signal tarama döngüsü hatası: {e}")
                    time.sleep(60)
        
        # Tarama thread'ini başlat
        scan_thread = threading.Thread(target=scan_loop)
        scan_thread.daemon = True
        scan_thread.start()
        
        print(f"🚀 Pro Signal Scanner başlatıldı!")
        print(f"📊 Maksimum Coin: {max_coins}")
    
    def stop_scan(self):
        """Taramayı durdur"""
        self.is_running = False
        print("🛑 Pro Signal Scanner durduruldu!")
    
    def get_latest_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Son sinyalleri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM pro_signals 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            signals = []
            for row in cursor.fetchall():
                # Güvenli indeks erişimi
                signals.append({
                    'symbol': row[1] if len(row) > 1 else 'N/A',
                    'signal_type': row[2] if len(row) > 2 else 'N/A',
                    'signal_category': row[3] if len(row) > 3 else 'N/A',
                    'signal_score': row[4] if len(row) > 4 else 0.0,
                    'entry_price': row[5] if len(row) > 5 else 0.0,
                    'stop_loss': row[6] if len(row) > 6 else 0.0,
                    'take_profit': row[7] if len(row) > 7 else 0.0,
                    'ema200': row[8] if len(row) > 8 else 0.0,
                    'ema50': row[9] if len(row) > 9 else 0.0,
                    'rsi': row[10] if len(row) > 10 else 0.0,
                    'macd': row[11] if len(row) > 11 else 0.0,
                    'macd_signal': row[12] if len(row) > 12 else 0.0,
                    'macd_histogram': row[13] if len(row) > 13 else 0.0,
                    'volatility': row[14] if len(row) > 14 else 0.0,
                    'momentum': row[15] if len(row) > 15 else 0.0,
                    'timestamp': row[16] if len(row) > 16 else None,
                    'timeframe': row[17] if len(row) > 17 else 'N/A'
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
            cursor.execute('SELECT COUNT(*) FROM pro_signals')
            total_signals = cursor.fetchone()[0]
            
            # Bugünkü sinyal sayısı
            today = datetime.now().date()
            cursor.execute('SELECT COUNT(*) FROM pro_signals WHERE DATE(timestamp) = ?', (today,))
            today_signals = cursor.fetchone()[0]
            
            # Sinyal kategorisi dağılımı
            cursor.execute('SELECT signal_category, COUNT(*) FROM pro_signals GROUP BY signal_category')
            category_distribution = dict(cursor.fetchall())
            
            # Ortalama sinyal skoru
            cursor.execute('SELECT AVG(signal_score) FROM pro_signals')
            avg_score = cursor.fetchone()[0] or 0
            
            # En yüksek skorlu sinyaller
            cursor.execute('SELECT signal_category, COUNT(*) FROM pro_signals WHERE signal_score >= 9 GROUP BY signal_category')
            mega_signals = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_signals': total_signals,
                'today_signals': today_signals,
                'category_distribution': category_distribution,
                'avg_score': round(avg_score, 2),
                'mega_signals': mega_signals,
                'current_cycle': self.current_cycle,
                'is_running': self.is_running
            }
            
        except Exception as e:
            print(f"İstatistik getirme hatası: {e}")
            return {}

def main():
    scanner = DevisoProSignalScanner()
    
    print("🚀 Deviso Pro Signal Scanner Test")
    print("=" * 50)
    
    # Test taraması
    scanner.start_pro_signal_scan(max_coins=5) # Test için 5 coin
    
    # Taramayı 10 saniye beklet
    time.sleep(10)
    
    # Taramayı durdur
    scanner.stop_scan()
    
    # Son sinyalleri getir
    latest_signals = scanner.get_latest_signals(limit=10)
    print("\nSon Sinyaller:")
    for signal in latest_signals:
        print(f"  {signal['symbol']} - {signal['signal_category']} (Skor: {signal['signal_score']})")

if __name__ == "__main__":
    main()
