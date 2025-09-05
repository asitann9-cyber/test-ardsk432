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

class DevisoEmaScanner:
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
        self.db_path = 'ema_scanner.db'
        self.init_database()
        self.is_running = False
        self.scan_results = []
        self.current_cycle = 0
        
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ema_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                signal_type TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                ratio REAL,
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
                signal_type = "LONG"
                stop_loss = ema200 * 0.98  # EMA200'ün %2 altı
                take_profit = current_price * 1.03  # Giriş fiyatının %3 üstü
                
                return {
                    'signal': True,
                    'signal_type': signal_type,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'ratio': ratio,
                    'ema200': ema200,
                    'reason': f'Fiyat EMA200 üstünde (Ratio: {ratio:.2f}%)'
                }
            else:
                return {'signal': False, 'reason': f'Fiyat EMA200 altında (Ratio: {ratio:.2f}%)'}
                
        except Exception as e:
            print(f"EMA koşul kontrolü hatası {symbol}: {e}")
            return {'signal': False, 'reason': f'Hata: {e}'}
    
    def calculate_ranking_score(self, signal_info: Dict[str, Any]) -> float:
        """Sinyal için sıralama skoru hesapla (Momentum, Hacim, Fiyat Değişimi)"""
        try:
            symbol = signal_info['symbol']
            current_price = signal_info['entry_price']
            
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
                    else:
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
                    
                    if avg_volume > 0 and current_volume > 0:
                        volume_spike = current_volume / avg_volume
                        
                        if volume_spike >= 3.0: volume_score += 25  # Çok yüksek hacim
                        elif volume_spike >= 2.0: volume_score += 20  # Yüksek hacim
                        elif volume_spike >= 1.5: volume_score += 15  # Orta hacim
                        elif volume_spike >= 1.2: volume_score += 10  # Hafif hacim artışı
                        elif volume_spike >= 1.0: volume_score += 5   # Normal hacim
                    else:
                        # Hacim verisi yoksa veya sıfırsa minimum puan
                        volume_score += 2
            except Exception as e:
                print(f"Hacim hesaplama hatası {symbol}: {e}")
                volume_score += 1  # Hata durumunda minimum puan
            
            # 3. FİYAT DEĞİŞİMİ SKORU (0-25 puan)
            price_change_score = 0
            bar_price_change_percent = 0  # Bar içi fiyat değişimi yüzdesi
            try:
                df_5m = self.get_klines_data(symbol, '5m', 20)
                if len(df_5m) >= 20:
                    # Mevcut bar'ın açılış fiyatından şu anki fiyata değişim
                    current_bar_open = df_5m['open'].iloc[-1]
                    bar_price_change_percent = ((current_price - current_bar_open) / current_bar_open) * 100
                    
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
            
            # 4. EMA KALİTESİ SKORU (0-20 puan)
            quality_score = 0
            try:
                df_5m = self.get_klines_data(symbol, '5m', 200)
                if len(df_5m) >= 200:
                    df_5m['ema200'] = self.calculate_ema(df_5m, 200)
                    current_price = df_5m['close'].iloc[-1]
                    ema200 = df_5m['ema200'].iloc[-1]
                    
                    # EMA200'a yakınlık
                    price_distance = abs(current_price - ema200) / ema200 * 100
                    if price_distance <= 0.5: quality_score += 20  # Çok yakın
                    elif price_distance <= 1.0: quality_score += 15  # Yakın
                    elif price_distance <= 2.0: quality_score += 10  # Orta
                    elif price_distance <= 5.0: quality_score += 5   # Uzak
            except Exception as e:
                print(f"EMA kalite hesaplama hatası {symbol}: {e}")
            
            # TOPLAM SIRALAMA SKORU
            total_ranking_score = momentum_score + volume_score + price_change_score + quality_score
            
            # Sinyal bilgisine sıralama detaylarını ekle
            signal_info['ranking_details'] = {
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'price_change_score': price_change_score,
                'quality_score': quality_score,
                'total_ranking_score': total_ranking_score,
                'bar_price_change_percent': bar_price_change_percent
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
            print(f"🚀 EMA TARAMA: En iyi fırsatlar sıralanacak!")
            print(f"📈 SIRALAMA ALGORİTMASI: Momentum + Hacim + Fiyat Değişimi + EMA Kalitesi")
            
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
                    
                    # EMA analizi yap
                    df = self.get_klines_data(symbol, timeframe, 200)
                    signal_data = self.check_ema_condition(df, symbol, timeframe)
                    
                    if signal_data['signal']:
                        signal_info = {
                            'symbol': symbol,
                            'signal_type': signal_data['signal_type'],
                            'entry_price': signal_data['entry_price'],
                            'stop_loss': signal_data['stop_loss'],
                            'take_profit': signal_data['take_profit'],
                            'ratio': signal_data['ratio'],
                            'timeframe': timeframe,
                            'timestamp': datetime.now(),
                            'reason': signal_data['reason']
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
                    print(f"   {i+1}. {signal['symbol']} - {signal['signal_type']}")
                    print(f"      📊 Sıralama Skoru: {signal['ranking_score']:.1f}")
                    print(f"      📈 Momentum: {ranking_details['momentum_score']:.1f}, Hacim: {ranking_details['volume_score']:.1f}")
                    print(f"      💰 Fiyat Değişimi: {ranking_details['price_change_score']:.1f}, EMA Kalitesi: {ranking_details['quality_score']:.1f}")
                    print(f"      📊 Risk/Ödül: {signal['ratio']:.2f}")
                
                # Sonuçları veritabanına kaydet
                for signal in all_signals:
                    self.save_signal_to_db(signal)
                
                print(f"🎉 EMA SIRALAMA ALGORİTMASI TAMAMLANDI!")
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
                INSERT INTO ema_signals 
                (symbol, signal_type, entry_price, stop_loss, take_profit, ratio, timestamp, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_info['symbol'],
                signal_info['signal_type'],
                signal_info['entry_price'],
                signal_info['stop_loss'],
                signal_info['take_profit'],
                signal_info['ratio'],
                signal_info['timestamp'],
                signal_info['timeframe']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Veritabanı kaydetme hatası: {e}")
    
    def start_scan(self, max_coins: int = 30, timeframe: str = '5m'):
        """Taramayı başlat"""
        self.is_running = True
        self.max_coins = max_coins
        self.timeframe = timeframe
        
        def scan_loop():
            cycle_count = 0
            while self.is_running:
                try:
                    cycle_count += 1
                    self.current_cycle = cycle_count
                    print(f"🔄 EMA200 Döngü {cycle_count} Başlatıldı - {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Coinleri tara
                    self.scan_results = self.scan_coins(max_coins, timeframe)
                    
                    print(f"⏳ 5 dakika sonra yeni döngü başlayacak...")
                    time.sleep(300)  # 5 dakika bekle
                    
                except Exception as e:
                    print(f"Tarama döngüsü hatası: {e}")
                    time.sleep(60)
        
        # Tarama thread'ini başlat
        scan_thread = threading.Thread(target=scan_loop)
        scan_thread.daemon = True
        scan_thread.start()
        
        print(f"🚀 EMA200 Scanner başlatıldı!")
        print(f"📊 Maksimum Coin: {max_coins}")
        print(f"⏰ Timeframe: {timeframe}")
    
    def stop_scan(self):
        """Taramayı durdur"""
        self.is_running = False
        print("🛑 EMA200 Scanner durduruldu!")
    
    def get_latest_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Son sinyalleri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM ema_signals 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            signals = []
            for row in cursor.fetchall():
                signals.append({
                    'symbol': row[1],
                    'signal_type': row[2],
                    'entry_price': row[3],
                    'stop_loss': row[4],
                    'take_profit': row[5],
                    'ratio': row[6],
                    'timestamp': row[7],
                    'timeframe': row[8]
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
            cursor.execute('SELECT COUNT(*) FROM ema_signals')
            total_signals = cursor.fetchone()[0]
            
            # Bugünkü sinyal sayısı
            today = datetime.now().date()
            cursor.execute('SELECT COUNT(*) FROM ema_signals WHERE DATE(timestamp) = ?', (today,))
            today_signals = cursor.fetchone()[0]
            
            # LONG/SHORT dağılımı
            cursor.execute('SELECT signal_type, COUNT(*) FROM ema_signals GROUP BY signal_type')
            signal_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_signals': total_signals,
                'today_signals': today_signals,
                'signal_distribution': signal_distribution,
                'current_cycle': self.current_cycle,
                'is_running': self.is_running
            }
            
        except Exception as e:
            print(f"İstatistik getirme hatası: {e}")
            return {}