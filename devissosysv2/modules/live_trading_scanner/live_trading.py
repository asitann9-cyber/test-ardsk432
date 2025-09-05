import pandas as pd
import numpy as np
import time
import threading
import sqlite3
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Any, Tuple
from binance.client import Client
from binance.enums import *
import talib
warnings.filterwarnings('ignore')

class DevisoLiveTradingScanner:
    def __init__(self):
        # Binance client başlat (API key gerekli)
        self.api_key = ""  # API key buraya eklenecek
        self.api_secret = ""  # API secret buraya eklenecek
        self.client = Client(self.api_key, self.api_secret)
        self.db_path = 'live_trading.db'
        self.init_database()
        self.is_running = False
        self.active_trades = {}
        self.completed_trades = []
        self.performance_metrics = {}
        
        # Futures Kasa Yönetimi Ayarları
        self.futures_balance = 1000      # Futures hesabında toplam bakiye (USDT)
        self.risk_per_trade = 0.02       # İşlem başına risk (%2)
        self.max_risk_per_day = 0.1      # Günlük maksimum risk (%10)
        self.leverage = 10               # Varsayılan kaldıraç
        self.margin_type = "ISOLATED"    # Margin tipi (ISOLATED/CROSSED)
        self.min_position_value = 10     # Minimum pozisyon değeri (USDT)
        self.max_position_value = 100    # Maksimum pozisyon değeri (USDT)
        self.auto_adjust_leverage = True # Otomatik kaldıraç ayarlama
        self.use_trailing_stop = False   # Trailing stop kullanımı
        self.trailing_stop_distance = 0.02  # Trailing stop mesafesi (%2)
        
        # Eski ayarlar (geriye uyumluluk için)
        self.position_size = 0.1         # Bakiyenin %10'u (eski sistem)
        self.scan_interval = 60          # 60 saniyede bir tarama
        self.max_coins = 30              # Maksimum coin sayısı
        self.min_signal_score = 9        # Minimum sinyal skoru (PRO ve üzeri)
        self.max_trades = 10             # Maksimum işlem sayısı
        self.use_testnet = True          # Testnet kullan (güvenlik için)
        
        # Günlük risk takibi
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Otomatik bakiye güncelleme
        self.auto_update_balance = True
        self.balance_update_interval = 30  # 30 saniyede bir güncelle
        self.last_balance_update = datetime.now()
        self.cached_account_info = {}
        
        # Testnet ayarları
        if self.use_testnet:
            self.client.API_URL = 'https://testnet.binancefuture.com/fapi'
        
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
                status TEXT DEFAULT 'ACTIVE',
                binance_order_id TEXT
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
                duration_minutes INTEGER,
                binance_order_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_account_balance(self) -> float:
        """Hesap bakiyesini al"""
        try:
            account = self.client.futures_account()
            for balance in account['assets']:
                if balance['asset'] == 'USDT':
                    return float(balance['balance'])
            return 0.0
        except Exception as e:
            print(f"Bakiye alma hatası: {e}")
            return 0.0

    def get_futures_account_info(self, force_update: bool = False) -> Dict[str, Any]:
        """Futures hesap bilgilerini al (cache ile)"""
        try:
            current_time = datetime.now()
            
            # Cache kontrolü - eğer son güncelleme üzerinden yeterli süre geçmediyse cache'den döndür
            if (not force_update and 
                self.cached_account_info and 
                (current_time - self.last_balance_update).total_seconds() < self.balance_update_interval):
                return self.cached_account_info
            
            # API'den yeni veri çek
            account = self.client.futures_account()
            positions = self.client.futures_position_information()
            
            # Toplam bakiye bilgileri (hesap seviyesinde)
            total_balance = float(account.get('totalWalletBalance', 0))
            available_balance = float(account.get('availableBalance', 0))
            total_margin_balance = float(account.get('totalMarginBalance', 0))
            
            # USDT asset bilgilerini de al
            usdt_asset = None
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    usdt_asset = asset
                    break
            
            # Aktif pozisyonları hesapla
            active_positions = []
            total_unrealized_pnl = 0.0
            
            for position in positions:
                if float(position['positionAmt']) != 0:
                    unrealized_pnl = float(position['unRealizedProfit'])
                    total_unrealized_pnl += unrealized_pnl
                    active_positions.append({
                        'symbol': position['symbol'],
                        'side': 'LONG' if float(position['positionAmt']) > 0 else 'SHORT',
                        'size': abs(float(position['positionAmt'])),
                        'entry_price': float(position['entryPrice']),
                        'unrealized_pnl': unrealized_pnl,
                        'leverage': int(position['leverage'])
                    })
            
            # Cache'e kaydet
            account_info = {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'total_margin_balance': total_margin_balance,
                'total_unrealized_pnl': total_unrealized_pnl,
                'active_positions': active_positions,
                'position_count': len(active_positions),
                'last_update': current_time.isoformat(),
                'update_timestamp': current_time.timestamp()
            }
            
            self.cached_account_info = account_info
            self.last_balance_update = current_time
            
            return account_info
            
        except Exception as e:
            print(f"Futures hesap bilgisi alma hatası: {e}")
            # Hata durumunda cache'den döndür (eğer varsa)
            if self.cached_account_info:
                return self.cached_account_info
            return {}

    def calculate_position_size(self, symbol: str, signal_score: float, current_price: float) -> Dict[str, Any]:
        """Risk bazlı pozisyon büyüklüğü hesapla"""
        try:
            # Günlük risk kontrolü
            self.check_daily_risk_reset()
            
            # Mevcut bakiye
            account_info = self.get_futures_account_info()
            available_balance = account_info.get('available_balance', 0)
            
            if available_balance < self.min_position_value:
                return {'error': f'Yetersiz bakiye: {available_balance} USDT'}
            
            # Günlük risk kontrolü
            daily_risk_limit = self.futures_balance * self.max_risk_per_day
            if abs(self.daily_pnl) >= daily_risk_limit:
                return {'error': f'Günlük risk limiti aşıldı: {self.daily_pnl:.2f} USDT'}
            
            # Sinyal skoruna göre risk ayarlama
            risk_multiplier = 1.0
            if signal_score >= 12:  # MEGA SIGNAL
                risk_multiplier = 1.5
            elif signal_score >= 9:  # PRO SIGNAL
                risk_multiplier = 1.0
            elif signal_score >= 6:  # STANDARD SIGNAL
                risk_multiplier = 0.7
            else:  # WEAK SIGNAL
                risk_multiplier = 0.5
            
            # Risk bazlı pozisyon değeri hesapla
            risk_amount = self.futures_balance * self.risk_per_trade * risk_multiplier
            position_value = min(risk_amount, self.max_position_value)
            position_value = max(position_value, self.min_position_value)
            
            # Kaldıraç hesaplama
            leverage = self.leverage
            if self.auto_adjust_leverage:
                # Volatiliteye göre kaldıraç ayarlama
                volatility = self.get_symbol_volatility(symbol)
                if volatility > 5:
                    leverage = min(leverage, 5)  # Yüksek volatilitede düşük kaldıraç
                elif volatility < 2:
                    leverage = min(leverage, 20)  # Düşük volatilitede yüksek kaldıraç
            
            # Pozisyon büyüklüğü (quantity) hesapla
            margin_required = position_value / leverage
            if margin_required > available_balance:
                # Bakiye yetersizse, mevcut bakiye ile maksimum pozisyon
                position_value = available_balance * leverage * 0.95  # %5 güvenlik payı
                margin_required = position_value / leverage
            
            quantity = position_value / current_price
            
            # Precision'a göre quantity'yi ayarla
            adjusted_quantity = self.adjust_quantity_precision(quantity, symbol)
            
            return {
                'position_value': round(position_value, 2),
                'quantity': adjusted_quantity,
                'leverage': leverage,
                'margin_required': round(margin_required, 2),
                'risk_amount': round(risk_amount, 2),
                'risk_multiplier': risk_multiplier
            }
            
        except Exception as e:
            print(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return {'error': str(e)}

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Sembol bilgilerini al (precision, min quantity, etc.)"""
        try:
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    # Quantity precision
                    quantity_precision = 0
                    for filter in s['filters']:
                        if filter['filterType'] == 'LOT_SIZE':
                            step_size = float(filter['stepSize'])
                            quantity_precision = len(str(step_size).split('.')[-1].rstrip('0'))
                            min_qty = float(filter['minQty'])
                            max_qty = float(filter['maxQty'])
                            break
                    
                    # Price precision
                    price_precision = 0
                    for filter in s['filters']:
                        if filter['filterType'] == 'PRICE_FILTER':
                            tick_size = float(filter['tickSize'])
                            price_precision = len(str(tick_size).split('.')[-1].rstrip('0'))
                            break
                    
                    return {
                        'symbol': symbol,
                        'quantity_precision': quantity_precision,
                        'price_precision': price_precision,
                        'min_qty': min_qty,
                        'max_qty': max_qty,
                        'step_size': step_size,
                        'tick_size': tick_size
                    }
            
            # Varsayılan değerler
            return {
                'symbol': symbol,
                'quantity_precision': 6,
                'price_precision': 2,
                'min_qty': 0.001,
                'max_qty': 1000000,
                'step_size': 0.001,
                'tick_size': 0.01
            }
            
        except Exception as e:
            print(f"Sembol bilgisi alma hatası {symbol}: {e}")
            # Varsayılan değerler
            return {
                'symbol': symbol,
                'quantity_precision': 6,
                'price_precision': 2,
                'min_qty': 0.001,
                'max_qty': 1000000,
                'step_size': 0.001,
                'tick_size': 0.01
            }

    def adjust_quantity_precision(self, quantity: float, symbol: str) -> float:
        """Miktarı sembol precision kurallarına göre ayarla"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            precision = symbol_info['quantity_precision']
            step_size = symbol_info['step_size']
            min_qty = symbol_info['min_qty']
            max_qty = symbol_info['max_qty']
            
            # Step size'a göre yuvarla
            adjusted_quantity = round(quantity / step_size) * step_size
            
            # Precision'a göre yuvarla
            adjusted_quantity = round(adjusted_quantity, precision)
            
            # Min/max kontrolü
            if adjusted_quantity < min_qty:
                adjusted_quantity = min_qty
            elif adjusted_quantity > max_qty:
                adjusted_quantity = max_qty
            
            return adjusted_quantity
            
        except Exception as e:
            print(f"Miktar precision ayarlama hatası: {e}")
            return round(quantity, 6)

    def get_symbol_volatility(self, symbol: str) -> float:
        """Sembol volatilitesini hesapla"""
        try:
            df = self.get_klines_data(symbol, '5m', 100)
            if len(df) < 20:
                return 3.0  # Varsayılan volatilite
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * 100
            return round(volatility, 2)
            
        except Exception as e:
            print(f"Volatilite hesaplama hatası: {e}")
            return 3.0

    def check_daily_risk_reset(self):
        """Günlük risk takibini sıfırla"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            print(f"📅 Günlük risk takibi sıfırlandı: {current_date}")

    def set_symbol_leverage(self, symbol: str, leverage: int) -> bool:
        """Sembol için kaldıraç ayarla"""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"⚙️ {symbol} kaldıracı {leverage}x olarak ayarlandı")
            return True
        except Exception as e:
            print(f"Kaldıraç ayarlama hatası {symbol}: {e}")
            return False

    def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Margin tipini ayarla"""
        try:
            if margin_type.upper() == "ISOLATED":
                self.client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            else:
                self.client.futures_change_margin_type(symbol=symbol, marginType="CROSSED")
            print(f"⚙️ {symbol} margin tipi {margin_type} olarak ayarlandı")
            return True
        except Exception as e:
            print(f"Margin tipi ayarlama hatası {symbol}: {e}")
            return False

    def setup_position_mode(self) -> bool:
        """Position mode'u ayarla (Hedge Mode için)"""
        try:
            # Mevcut position mode'u kontrol et
            account_info = self.client.futures_account()
            dual_side_position = account_info.get('dualSidePosition', False)
            
            if not dual_side_position:
                # Hedge Mode'u aktif et
                self.client.futures_change_position_mode(dualSidePosition=True)
                print("⚙️ Hedge Mode aktif edildi")
            else:
                print("⚙️ Hedge Mode zaten aktif")
            
            return True
            
        except Exception as e:
            if "No need to change" in str(e):
                print("⚙️ Position mode zaten doğru ayarlanmış")
                return True
            else:
                print(f"Position mode ayarlama hatası: {e}")
                return False

    def get_current_price(self, symbol: str) -> float:
        """Sembolün güncel fiyatını al"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Fiyat alma hatası {symbol}: {e}")
            return None

    def get_klines_data(self, symbol: str, interval: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Binance'den kline verisi al"""
        try:
            interval_map = {
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }
            
            binance_interval = interval_map.get(interval, Client.KLINE_INTERVAL_5MINUTE)
            
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
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

    def calculate_signal_score(self, symbol: str, timeframe: str = '5m') -> Dict[str, Any]:
        """Sinyal skorunu hesapla"""
        try:
            df = self.get_klines_data(symbol, timeframe, 100)
            if len(df) < 50:
                return {'score': 0, 'signals': []}

            df['ema200'] = self.calculate_ema(df, 200)
            df['ema50'] = self.calculate_ema(df, 50)
            df['rsi'] = self.calculate_rsi(df, 14)
            df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df)
            
            current_price = df['close'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]
            ema50 = df['ema50'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_histogram = df['macd_histogram'].iloc[-1]
            
            volatility = df['close'].pct_change().std() * 100
            
            score = 0
            signals = []
            
            if current_price > ema200:
                score += 1
                signals.append("EMA200_ABOVE")
            if current_price > ema50:
                score += 1
                signals.append("EMA50_ABOVE")
            if ema50 > ema200:
                score += 1
                signals.append("EMA50_ABOVE_EMA200")
            
            if 30 <= rsi <= 70:
                score += 1
                signals.append("RSI_NEUTRAL")
            if rsi > 50:
                score += 1
                signals.append("RSI_BULLISH")
            
            if macd > macd_signal:
                score += 1
                signals.append("MACD_BULLISH")
            if macd_histogram > 0:
                score += 1
                signals.append("MACD_HISTOGRAM_POSITIVE")
            
            if 2 <= volatility <= 8:
                score += 1
                signals.append("VOLATILITY_OPTIMAL")
            
            if timeframe == '5m':
                lookback = 6
            elif timeframe == '15m':
                lookback = 4
            else:
                lookback = 6
            
            if len(df) >= lookback:
                price_change = ((current_price - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]) * 100
                if price_change > 0:
                    score += 1
                    signals.append("MOMENTUM_POSITIVE")
                if price_change > 1:
                    score += 1
                    signals.append("MOMENTUM_STRONG")
            
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
                'rsi': rsi,
                'macd': macd,
                'volatility': volatility,
                'timeframe': timeframe
            }
            
        except Exception as e:
            print(f"Sinyal hesaplama hatası {symbol}: {e}")
            return {'score': 0, 'signals': []}

    def calculate_multi_timeframe_score(self, symbol: str) -> Dict[str, Any]:
        """Çoklu timeframe analizi yap"""
        try:
            timeframes = ['5m', '15m', '1h']
            timeframe_scores = {}
            total_score = 0
            all_signals = []
            
            for tf in timeframes:
                signal_data = self.calculate_signal_score(symbol, tf)
                timeframe_scores[tf] = signal_data
                total_score += signal_data['score']
                all_signals.extend([f"{tf}_{signal}" for signal in signal_data['signals']])
            
            avg_score = total_score / len(timeframes)
            
            weighted_score = (
                timeframe_scores['5m']['score'] * 0.5 +
                timeframe_scores['15m']['score'] * 0.3 +
                timeframe_scores['1h']['score'] * 0.2
            )
            
            trend_consistency = 0
            if (timeframe_scores['5m']['signal_type'] == timeframe_scores['15m']['signal_type'] == 
                timeframe_scores['1h']['signal_type']):
                trend_consistency = 2
            elif (timeframe_scores['5m']['signal_type'] == timeframe_scores['15m']['signal_type'] or
                  timeframe_scores['15m']['signal_type'] == timeframe_scores['1h']['signal_type']):
                trend_consistency = 1
            
            final_score = weighted_score + trend_consistency
            
            if final_score >= 12:
                category = "MEGA SIGNAL"
            elif final_score >= 9:
                category = "PRO SIGNAL"
            elif final_score >= 6:
                category = "STANDARD SIGNAL"
            else:
                category = "WEAK SIGNAL"
            
            main_signal_type = timeframe_scores['5m']['signal_type']
            
            return {
                'final_score': round(final_score, 2),
                'weighted_score': round(weighted_score, 2),
                'avg_score': round(avg_score, 2),
                'trend_consistency': trend_consistency,
                'category': category,
                'signal_type': main_signal_type,
                'timeframe_scores': timeframe_scores,
                'all_signals': all_signals,
                'current_price': timeframe_scores['5m']['current_price']
            }
            
        except Exception as e:
            print(f"Çoklu timeframe analiz hatası {symbol}: {e}")
            return {'final_score': 0, 'signals': []}

    def scan_coins(self) -> List[Dict[str, Any]]:
        """Coinleri tara ve sinyalleri bul"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = []
            
            for symbol_info in exchange_info['symbols']:
                symbol = symbol_info['symbol']
                if symbol.endswith('USDT') and symbol_info['status'] == 'TRADING':
                    symbols.append(symbol)
            
            print(f"📊 Toplam {len(symbols)} sembol bulundu, ilk {self.max_coins} tanesi taranacak...")
            print(f"🕐 Çoklu Timeframe Analizi: 5m, 15m, 1h")
            
            signals = []
            scanned_count = 0
            
            for symbol in symbols[:self.max_coins]:
                try:
                    scanned_count += 1
                    print(f"🔍 Live Trading Analizi: {symbol} ({scanned_count}/{self.max_coins})")
                    
                    multi_tf_data = self.calculate_multi_timeframe_score(symbol)
                    
                    if multi_tf_data['final_score'] >= self.min_signal_score:
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
                                '5m_score': tf_scores['5m']['score'],
                                '15m_score': tf_scores['15m']['score'],
                                '1h_score': tf_scores['1h']['score'],
                                'trend_consistency': multi_tf_data['trend_consistency'],
                                'weighted_score': multi_tf_data['weighted_score']
                            }
                        }
                        signals.append(signal_info)
                        
                        print(f"🔥 {multi_tf_data['category']}: {symbol}")
                        print(f"   📊 Final Skor: {multi_tf_data['final_score']:.2f}")
                        print(f"   📈 5m: {tf_scores['5m']['score']}/10, 15m: {tf_scores['15m']['score']}/10, 1h: {tf_scores['1h']['score']}/10")
                        print(f"   🎯 Trend Tutarlılığı: {multi_tf_data['trend_consistency']}/2")
                    
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"Coin tarama hatası {symbol}: {e}")
                    continue
            
            print(f"🎉 Bu taramada {len(signals)} sinyal bulundu!")
            return signals
            
        except Exception as e:
            print(f"Genel tarama hatası: {e}")
            return []

    def calculate_optimal_tp_sl(self, symbol: str, signal_type: str, signal_score: float, timeframe_analysis: Dict = None) -> Tuple[float, float]:
        """Optimal TP/SL noktalarını hesapla"""
        try:
            if timeframe_analysis:
                atr_values = {}
                for tf in ['5m', '15m', '1h']:
                    df = self.get_klines_data(symbol, tf, 100)
                    if len(df) > 14:
                        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                        atr_values[tf] = df['atr'].iloc[-1]
                
                weighted_atr = (
                    atr_values.get('5m', 0) * 0.5 +
                    atr_values.get('15m', 0) * 0.3 +
                    atr_values.get('1h', 0) * 0.2
                )
                
                df_5m = self.get_klines_data(symbol, '5m', 100)
                current_price = df_5m['close'].iloc[-1]
                volatility = df_5m['close'].pct_change().std() * 100
            else:
                df = self.get_klines_data(symbol, '5m', 100)
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                current_price = df['close'].iloc[-1]
                weighted_atr = df['atr'].iloc[-1]
                volatility = df['close'].pct_change().std() * 100
            
            if signal_score >= 12:
                tp_multiplier = 3.5
                sl_multiplier = 1.8
            elif signal_score >= 9:
                tp_multiplier = 3.0
                sl_multiplier = 1.5
            elif signal_score >= 6:
                tp_multiplier = 2.5
                sl_multiplier = 1.2
            else:
                tp_multiplier = 2.0
                sl_multiplier = 1.0
            
            if timeframe_analysis and timeframe_analysis.get('trend_consistency', 0) >= 2:
                tp_multiplier *= 1.1
                sl_multiplier *= 0.9
            
            if volatility > 5:
                tp_multiplier *= 1.2
                sl_multiplier *= 1.1
            elif volatility < 2:
                tp_multiplier *= 0.8
                sl_multiplier *= 0.9
            
            if signal_type == 'LONG':
                take_profit = current_price + (weighted_atr * tp_multiplier)
                stop_loss = current_price - (weighted_atr * sl_multiplier)
            else:
                take_profit = current_price - (weighted_atr * tp_multiplier)
                stop_loss = current_price + (weighted_atr * sl_multiplier)
            
            return take_profit, stop_loss
            
        except Exception as e:
            print(f"TP/SL hesaplama hatası: {e}")
            if signal_type == 'LONG':
                return current_price * 1.02, current_price * 0.98
            else:
                return current_price * 0.98, current_price * 1.02

    def open_live_trade(self, signal_data: Dict[str, Any]) -> bool:
        """Canlı işlem aç"""
        try:
            symbol = signal_data['symbol']
            signal_type = signal_data['signal_type']
            signal_category = signal_data['signal_category']
            signal_score = signal_data['signal_score']
            current_price = signal_data['current_price']
            timeframe_analysis = signal_data.get('timeframe_analysis', {})
            
            # Futures hesap bilgilerini al
            account_info = self.get_futures_account_info()
            available_balance = account_info.get('available_balance', 0)
            
            if available_balance < self.min_position_value:
                print(f"⚠️ Yetersiz bakiye: {available_balance} USDT")
                return False
            
            # Risk bazlı pozisyon büyüklüğü hesapla
            position_calc = self.calculate_position_size(symbol, signal_score, current_price)
            if 'error' in position_calc:
                print(f"⚠️ Pozisyon hesaplama hatası: {position_calc['error']}")
                return False
            
            position_value = position_calc['position_value']
            quantity = position_calc['quantity']
            leverage = position_calc['leverage']
            margin_required = position_calc['margin_required']
            
            # TP/SL hesapla
            take_profit, stop_loss = self.calculate_optimal_tp_sl(
                symbol, signal_type, signal_score, timeframe_analysis
            )
            
            # Trade ID oluştur
            trade_id = f"LIVE_{symbol}_{int(time.time())}"
            
            # Kaldıraç ve margin tipini ayarla
            self.set_symbol_leverage(symbol, leverage)
            self.set_margin_type(symbol, self.margin_type)
            
            # Binance'de işlem aç
            side = SIDE_BUY if signal_type == 'LONG' else SIDE_SELL
            
            try:
                # Quantity'yi tekrar kontrol et ve ayarla
                final_quantity = self.adjust_quantity_precision(quantity, symbol)
                
                print(f"🔧 İşlem detayları:")
                print(f"   📊 Sembol: {symbol}")
                print(f"   📈 Yön: {signal_type} ({side})")
                print(f"   📊 Miktar: {final_quantity} (orijinal: {quantity})")
                print(f"   💰 Pozisyon Değeri: {position_value} USDT")
                
                # Position side belirle (Hedge Mode için)
                position_side = "LONG" if signal_type == "LONG" else "SHORT"
                print(f"⚙️ Position Side: {position_side}")
                
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=final_quantity,
                    positionSide=position_side
                )
                
                binance_order_id = order['orderId']
                
                # Veritabanına kaydet
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO active_trades 
                    (trade_id, symbol, signal_type, signal_category, signal_score, 
                     entry_price, entry_time, position_size, stop_loss, take_profit, binance_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id, symbol, signal_type, signal_category, signal_score,
                    current_price, datetime.now(), final_quantity, stop_loss, take_profit, binance_order_id
                ))
                
                conn.commit()
                conn.close()
                
                # Aktif işlemler listesine ekle
                self.active_trades[trade_id] = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'signal_category': signal_category,
                    'signal_score': signal_score,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'position_size': final_quantity,
                    'position_value': position_value,
                    'leverage': leverage,
                    'margin_required': margin_required,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'status': 'ACTIVE',
                    'binance_order_id': binance_order_id,
                    'timeframe_analysis': timeframe_analysis,
                    'risk_multiplier': position_calc['risk_multiplier']
                }
                
                # Günlük işlem sayısını artır
                self.daily_trades += 1
                
                print(f"🔥 Canlı işlem açıldı: {symbol} {signal_type}")
                print(f"   📊 {signal_category} - Skor: {signal_score:.2f}")
                print(f"   💰 Pozisyon Değeri: {position_value} USDT")
                print(f"   📈 Kaldıraç: {leverage}x, Margin: {margin_required} USDT")
                print(f"   🎯 Risk Çarpanı: {position_calc['risk_multiplier']:.2f}")
                print(f"   💰 TP: {take_profit:.6f}, SL: {stop_loss:.6f}")
                print(f"   📈 Binance Order ID: {binance_order_id}")
                print(f"   📊 Günlük İşlem: {self.daily_trades}, Günlük PnL: {self.daily_pnl:.2f} USDT")
                
                if timeframe_analysis:
                    print(f"   📈 5m: {timeframe_analysis.get('5m_score', 0)}/10, "
                          f"15m: {timeframe_analysis.get('15m_score', 0)}/10, "
                          f"1h: {timeframe_analysis.get('1h_score', 0)}/10")
                    print(f"   🎯 Trend Tutarlılığı: {timeframe_analysis.get('trend_consistency', 0)}/2")
                
                return True
                
            except Exception as e:
                print(f"Binance işlem açma hatası: {e}")
                return False
            
        except Exception as e:
            print(f"Canlı işlem açma hatası: {e}")
            return False

    def close_live_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """Canlı işlemi kapat"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return False
            
            symbol = trade['symbol']
            signal_type = trade['signal_type']
            position_size = trade['position_size']
            binance_order_id = trade.get('binance_order_id')
            
            # Binance'de pozisyonu kapat
            side = SIDE_SELL if signal_type == 'LONG' else SIDE_BUY
            
            try:
                # Position side belirle (Hedge Mode için)
                position_side = "LONG" if signal_type == "LONG" else "SHORT"
                
                close_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=position_size,
                    positionSide=position_side
                )
                
                # PnL hesapla
                entry_price = trade['entry_price']
                position_size = trade['position_size']
                if signal_type == 'LONG':
                    pnl = (exit_price - entry_price) * position_size
                else:
                    pnl = (entry_price - exit_price) * position_size
                
                pnl_percentage = (pnl / (entry_price * position_size)) * 100
                
                # Günlük PnL'i güncelle
                self.daily_pnl += pnl
                
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
                     pnl, pnl_percentage, exit_reason, duration_minutes, binance_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id, trade['symbol'], trade['signal_type'], trade['signal_category'],
                    trade['signal_score'], entry_price, exit_price, trade['entry_time'],
                    datetime.now(), position_size, pnl, pnl_percentage, exit_reason, duration_minutes, binance_order_id
                ))
                
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
                    'duration_minutes': duration_minutes,
                    'binance_order_id': binance_order_id
                })
                
                print(f"✅ Canlı işlem kapatıldı: {trade['symbol']} - PnL: {pnl:.2f} USDT ({pnl_percentage:.2f}%) - {exit_reason}")
                print(f"   📈 Binance Close Order ID: {close_order['orderId']}")
                return True
                
            except Exception as e:
                print(f"Binance işlem kapatma hatası: {e}")
                return False
            
        except Exception as e:
            print(f"İşlem kapatma hatası: {e}")
            return False

    def check_trade_exit_conditions(self, trade_id: str, current_price: float) -> Tuple[bool, str, float]:
        """İşlem çıkış koşullarını kontrol et"""
        trade = self.active_trades.get(trade_id)
        if not trade or current_price is None:
            return False, "", 0
        
        symbol = trade['symbol']
        signal_type = trade['signal_type']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        
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

    def auto_trading_loop(self):
        """Otomatik trading döngüsü"""
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                print(f"🔄 Live Trading Döngü {cycle_count} Başlatıldı - {datetime.now().strftime('%H:%M:%S')}")
                
                # Maksimum işlem sayısını kontrol et
                if len(self.active_trades) >= self.max_trades:
                    print(f"⚠️ Maksimum işlem sayısına ulaşıldı ({self.max_trades}). Yeni işlem açılmayacak.")
                    time.sleep(self.scan_interval)
                    continue
                
                # Coinleri tara
                signals = self.scan_coins()
                
                # Sinyallere göre işlem aç
                for signal in signals:
                    if len(self.active_trades) >= self.max_trades:
                        break
                    
                    # Aynı sembolde zaten işlem var mı kontrol et
                    symbol_exists = any(trade['symbol'] == signal['symbol'] for trade in self.active_trades.values())
                    
                    if not symbol_exists and signal['signal_score'] >= self.min_signal_score:
                        # Canlı işlemi aç
                        self.open_live_trade(signal)
                        time.sleep(1)
                
                print(f"⏳ {self.scan_interval} saniye sonra yeni döngü başlayacak...")
                time.sleep(self.scan_interval)
                
            except Exception as e:
                print(f"Otomatik trading döngüsü hatası: {e}")
                time.sleep(30)

    def monitor_active_trades(self):
        """Aktif işlemleri izle"""
        while self.is_running:
            try:
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        symbol = trade['symbol']
                        current_price = self.get_current_price(symbol)
                        
                        if current_price is None:
                            print(f"⚠️ {symbol} için fiyat alınamadı, atlanıyor...")
                            continue
                        
                        should_exit, exit_reason, exit_price = self.check_trade_exit_conditions(trade_id, current_price)
                        
                        if should_exit:
                            self.close_live_trade(trade_id, exit_price, exit_reason)
                    
                    except Exception as e:
                        print(f"İşlem izleme hatası {trade.get('symbol', 'UNKNOWN')}: {e}")
                        continue
                
                time.sleep(10)
                
            except Exception as e:
                print(f"Genel işlem izleme hatası: {e}")
                time.sleep(30)

    def auto_balance_update_loop(self):
        """Otomatik bakiye güncelleme döngüsü"""
        while self.is_running:
            try:
                if self.auto_update_balance:
                    # Bakiye bilgilerini güncelle
                    account_info = self.get_futures_account_info(force_update=True)
                    
                    if account_info:
                        print(f"💰 Bakiye Güncellendi - {datetime.now().strftime('%H:%M:%S')}")
                        print(f"   💵 Toplam: {account_info.get('total_balance', 0):.2f} USDT")
                        print(f"   💰 Kullanılabilir: {account_info.get('available_balance', 0):.2f} USDT")
                        print(f"   📊 Gerçekleşmemiş PnL: {account_info.get('total_unrealized_pnl', 0):.2f} USDT")
                        print(f"   📈 Aktif Pozisyon: {account_info.get('position_count', 0)}")
                
                time.sleep(self.balance_update_interval)
                
            except Exception as e:
                print(f"Otomatik bakiye güncelleme hatası: {e}")
                time.sleep(60)  # Hata durumunda 1 dakika bekle

    def start_live_trading(self, max_coins: int = 30, max_trades: int = 10):
        """Live trading'i başlat"""
        if not self.api_key or not self.api_secret:
            print("❌ API Key ve Secret gerekli! Lütfen config dosyasında ayarlayın.")
            return
        
        self.is_running = True
        self.max_trades = max_trades
        self.max_coins = max_coins
        
        # Futures hesap bilgilerini al
        account_info = self.get_futures_account_info()
        total_balance = account_info.get('total_balance', 0)
        available_balance = account_info.get('available_balance', 0)
        total_unrealized_pnl = account_info.get('total_unrealized_pnl', 0)
        position_count = account_info.get('position_count', 0)
        
        print(f"💰 Futures Hesap Bilgileri:")
        print(f"   💵 Toplam Bakiye: {total_balance} USDT")
        print(f"   💰 Kullanılabilir Bakiye: {available_balance} USDT")
        print(f"   📊 Gerçekleşmemiş PnL: {total_unrealized_pnl} USDT")
        print(f"   📈 Aktif Pozisyon Sayısı: {position_count}")
        
        if available_balance < self.min_position_value:
            print(f"❌ Yetersiz bakiye! Minimum {self.min_position_value} USDT gerekli.")
            return
        
        # Position mode'u ayarla (Hedge Mode için)
        self.setup_position_mode()
        
        # İşlem izleme thread'ini başlat
        monitor_thread = threading.Thread(target=self.monitor_active_trades)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Otomatik trading döngüsünü başlat
        trading_thread = threading.Thread(target=self.auto_trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        # Otomatik bakiye güncelleme thread'ini başlat
        balance_thread = threading.Thread(target=self.auto_balance_update_loop)
        balance_thread.daemon = True
        balance_thread.start()
        
        print(f"🚀 Deviso Live Trading başlatıldı!")
        print(f"📊 Maksimum coin sayısı: {max_coins}")
        print(f"📊 Maksimum işlem sayısı: {max_trades}")
        print(f"💰 Futures Bakiyesi: {self.futures_balance} USDT")
        print(f"🎯 Risk Yönetimi:")
        print(f"   📊 İşlem Başına Risk: %{self.risk_per_trade * 100}")
        print(f"   📈 Günlük Maksimum Risk: %{self.max_risk_per_day * 100}")
        print(f"   💰 Minimum Pozisyon: {self.min_position_value} USDT")
        print(f"   💰 Maksimum Pozisyon: {self.max_position_value} USDT")
        print(f"   ⚙️ Varsayılan Kaldıraç: {self.leverage}x")
        print(f"   🔒 Margin Tipi: {self.margin_type}")
        print(f"   🔄 Otomatik Kaldıraç: {'Aktif' if self.auto_adjust_leverage else 'Deaktif'}")
        print(f"   📈 Trailing Stop: {'Aktif' if self.use_trailing_stop else 'Deaktif'}")
        print(f"🔍 Otomatik tarama aktif - {self.scan_interval} saniyede bir")
        print(f"🕐 Çoklu Timeframe Analizi: 5m, 15m, 1h")
        print(f"🎯 Minimum sinyal skoru: {self.min_signal_score}/15 (PRO ve üzeri)")
        print(f"⚖️ Timeframe Ağırlıkları: 5m (%50), 15m (%30), 1h (%20)")
        print(f"🔒 Testnet: {'Aktif' if self.use_testnet else 'Deaktif'}")
    
    def stop_live_trading(self):
        """Live trading'i durdur"""
        self.is_running = False
        print("🛑 Deviso Live Trading durduruldu!")

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
                    'duration_minutes': row[14],
                    'binance_order_id': row[15]
                })
            
            conn.close()
            return trades
            
        except Exception as e:
            print(f"Tamamlanan işlemler getirme hatası: {e}")
            return []

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Performans metriklerini hesapla"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            
            winning_trades = df_completed[df_completed['pnl'] > 0]
            losing_trades = df_completed[df_completed['pnl'] < 0]
            
            total_trades = len(df_completed)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            win_rate = (winning_count / total_trades) * 100 if total_trades > 0 else 0
            total_pnl = df_completed['pnl'].sum()
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            
            profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())) if len(losing_trades) > 0 else float('inf')
            
            mega_signals = df_completed[df_completed['signal_category'] == 'MEGA SIGNAL']
            pro_signals = df_completed[df_completed['signal_category'] == 'PRO SIGNAL']
            
            mega_pnl = mega_signals['pnl_percentage'].mean() if len(mega_signals) > 0 else 0
            pro_pnl = pro_signals['pnl_percentage'].mean() if len(pro_signals) > 0 else 0
            
            cumulative_pnl = df_completed['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / running_max * 100
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
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

    def get_performance_summary(self) -> Dict[str, Any]:
        """Performans özetini getir"""
        metrics = self.calculate_performance_metrics()
        account_info = self.get_futures_account_info()
        
        return {
            'metrics': metrics,
            'active_trades_count': len(self.active_trades),
            'futures_account': account_info,
            'risk_management': {
                'futures_balance': self.futures_balance,
                'risk_per_trade': self.risk_per_trade * 100,
                'max_risk_per_day': self.max_risk_per_day * 100,
                'leverage': self.leverage,
                'margin_type': self.margin_type,
                'min_position_value': self.min_position_value,
                'max_position_value': self.max_position_value,
                'auto_adjust_leverage': self.auto_adjust_leverage,
                'use_trailing_stop': self.use_trailing_stop,
                'trailing_stop_distance': self.trailing_stop_distance * 100
            },
            'daily_tracking': {
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_trades': self.daily_trades,
                'last_reset_date': self.last_reset_date.isoformat()
            }
        }

    def test_manual_trade(self, symbol: str = "BTCUSDT", signal_type: str = "LONG") -> bool:
        """Test için manuel işlem aç"""
        try:
            print(f"🧪 Test işlemi açılıyor: {symbol} {signal_type}")
            
            # Sembol bilgilerini kontrol et
            symbol_info = self.get_symbol_info(symbol)
            print(f"📊 Sembol bilgileri: {symbol_info}")
            
            # Test sinyal verisi oluştur
            test_signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_category': 'TEST SIGNAL',
                'signal_score': 8.0,  # Test skoru
                'current_price': self.get_current_price(symbol),
                'signals': ['TEST_SIGNAL'],
                'timestamp': datetime.now(),
                'timeframe_analysis': {
                    '5m_score': 8,
                    '15m_score': 7,
                    '1h_score': 6,
                    'trend_consistency': 1,
                    'weighted_score': 7.5
                }
            }
            
            if test_signal['current_price'] is None:
                print(f"❌ {symbol} için fiyat alınamadı!")
                return False
            
            print(f"💰 Test fiyatı: {test_signal['current_price']}")
            
            # İşlemi aç
            success = self.open_live_trade(test_signal)
            
            if success:
                print(f"✅ Test işlemi başarıyla açıldı!")
                print(f"📊 Aktif işlem sayısı: {len(self.active_trades)}")
                return True
            else:
                print(f"❌ Test işlemi açılamadı!")
                return False
                
        except Exception as e:
            print(f"Test işlemi hatası: {e}")
            return False

    def close_all_trades(self) -> bool:
        """Tüm aktif işlemleri kapat"""
        try:
            print(f"🛑 Tüm işlemler kapatılıyor... ({len(self.active_trades)} aktif işlem)")
            
            closed_count = 0
            for trade_id, trade in list(self.active_trades.items()):
                try:
                    current_price = self.get_current_price(trade['symbol'])
                    if current_price:
                        success = self.close_live_trade(trade_id, current_price, "MANUAL_CLOSE")
                        if success:
                            closed_count += 1
                            print(f"✅ {trade['symbol']} işlemi kapatıldı")
                        else:
                            print(f"❌ {trade['symbol']} işlemi kapatılamadı")
                except Exception as e:
                    print(f"İşlem kapatma hatası {trade.get('symbol', 'UNKNOWN')}: {e}")
            
            print(f"📊 Toplam {closed_count} işlem kapatıldı")
            return closed_count > 0
            
        except Exception as e:
            print(f"Toplu işlem kapatma hatası: {e}")
            return False
