"""
Binance API servisleri
Binance vadeli işlemler API'si ile veri çekme ve sembol yönetimi
Örnek koddaki daha doğru Supertrend hesaplama için optimize edilmiş
"""

import requests
import pandas as pd
import logging
import numpy as np
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class BinanceService:
    """Binance API işlemleri için service sınıfı"""
    
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    # Timeframe limitleri - Örnek koddan alınan daha yüksek limitler
    # Yeni emtialar için daha doğru Supertrend hesaplama
    TIMEFRAME_LIMITS = {
        '1m': 500, '5m': 500, '15m': 500, '30m': 500,
        '1h': 500, '2h': 500, '4h': 500, '1d': 500
    }
    
    @classmethod
    def fetch_symbols(cls) -> List[str]:
        """
        Binance USDT vadeli işlem sembollerini çek
        
        Returns:
            List[str]: Aktif USDT vadeli işlem sembolleri listesi
        """
        try:
            url = f"{cls.BASE_URL}/exchangeInfo"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            symbols = []
            for symbol_info in data['symbols']:
                if (symbol_info.get('quoteAsset') == 'USDT' and 
                    symbol_info.get('status') == 'TRADING' and
                    symbol_info.get('contractType') == 'PERPETUAL'):
                    symbols.append(symbol_info['symbol'])
            
            logger.info(f"Binance: {len(symbols)} USDT vadeli sembolü bulundu")
            return sorted(symbols)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API bağlantı hatası: {e}")
            raise
        except Exception as e:
            logger.error(f"Binance sembol listesi çekme hatası: {e}")
            raise
    
    @classmethod
    def fetch_klines_data(cls, symbol: str, timeframe: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Belirli sembol ve timeframe için OHLCV verilerini çek
        Örnek koddan optimize edilmiş - Daha fazla veri çeker
        
        Args:
            symbol (str): Sembol adı (ör: BTCUSDT)
            timeframe (str): Zaman dilimi (ör: 4h)
            limit (Optional[int]): Veri limiti, None ise otomatik belirlenir
            
        Returns:
            Optional[pd.DataFrame]: OHLCV verileri içeren DataFrame
        """
        try:
            if limit is None:
                limit = cls.TIMEFRAME_LIMITS.get(timeframe, 500)  # Varsayılan 500
            
            # Maksimum 1000'e sınırla (Binance limiti)
            limit = min(limit, 1000)
            
            url = f"{cls.BASE_URL}/klines"
            
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"Boş veri: {symbol} {timeframe}")
                return None
            
            # DataFrame oluştur
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Timestamp'i datetime'a çevir (UTC -> Istanbul)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Istanbul')
            
            # Numerik kolonları float'a çevir
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Tarih sırasına göre sırala
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            
            logger.debug(f"Veri çekildi: {symbol} {timeframe} - {len(df)} mum (limit: {limit})")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API bağlantı hatası {symbol}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Binance veri çekme hatası {symbol}: {e}")
            return None
    
    @classmethod
    def calculate_required_candles(cls, time_range_days: int, timeframe: str) -> int:
        """
        Örnek koddan alınan: Gerekli mum sayısını hesapla
        Daha uzun veri aralıkları için optimize edilmiş
        
        Args:
            time_range_days (int): Gün cinsinden zaman aralığı
            timeframe (str): Timeframe
            
        Returns:
            int: Gerekli mum sayısı
        """
        if timeframe == '1m':
            return min(1000, 24 * 60 * time_range_days)
        elif timeframe == '5m':
            return min(1000, 24 * 12 * time_range_days)
        elif timeframe == '15m':
            return min(1000, 24 * 4 * max(15, time_range_days))
        elif timeframe == '30m':
            return min(1000, 24 * 2 * time_range_days)
        elif timeframe == '1h':
            return min(1000, 24 * max(20, time_range_days))
        elif timeframe == '2h':
            return min(1000, 12 * max(25, time_range_days))
        elif timeframe == '4h':
            return min(1000, 6 * max(30, time_range_days))
        elif timeframe == '1d':
            return min(1000, max(100, time_range_days))
        else:
            return min(1000, 24 * 4 * time_range_days)
    
    @classmethod
    def fetch_enhanced_klines_data(cls, symbol: str, timeframe: str, time_range_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Örnek koddan esinlenilmiş: Gelişmiş veri çekme
        Supertrend hesaplama için optimize edilmiş daha fazla veri
        
        Args:
            symbol (str): Sembol adı
            timeframe (str): Zaman dilimi
            time_range_days (int): Kaç günlük veri
            
        Returns:
            Optional[pd.DataFrame]: Genişletilmiş OHLCV verileri
        """
        try:
            # Gerekli mum sayısını hesapla
            required_candles = cls.calculate_required_candles(time_range_days, timeframe)
            
            # Veri çek
            df = cls.fetch_klines_data(symbol, timeframe, required_candles)
            
            if df is None or len(df) < 50:
                logger.warning(f"Yetersiz veri: {symbol} {timeframe} - {len(df) if df is not None else 0} mum")
                return None
            
            # Ek bilgiler ekle
            df['price_change'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['hl_avg'] = (df['high'] + df['low']) / 2
            
            logger.debug(f"Gelişmiş veri çekildi: {symbol} {timeframe} - {len(df)} mum ({time_range_days} gün)")
            return df
            
        except Exception as e:
            logger.debug(f"Gelişmiş veri çekme hatası {symbol}: {e}")
            return None
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        Sembolün geçerli olup olmadığını kontrol et
        
        Args:
            symbol (str): Kontrol edilecek sembol
            
        Returns:
            bool: Sembol geçerli ise True
        """
        try:
            all_symbols = cls.fetch_symbols()
            return symbol in all_symbols
        except Exception:
            return False
    
    @classmethod
    def get_current_price(cls, symbol: str) -> Optional[float]:
        """
        Sembolün güncel fiyatını getir
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Optional[float]: Güncel fiyat
        """
        try:
            url = f"{cls.BASE_URL}/ticker/price"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return float(data['price'])
            
        except Exception as e:
            logger.debug(f"Fiyat çekme hatası {symbol}: {e}")
            return None
    
    @classmethod
    def get_market_info(cls, symbol: str) -> Dict[str, Any]:
        """
        Sembol hakkında pazar bilgilerini getir
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Dict[str, Any]: Pazar bilgileri
        """
        try:
            url = f"{cls.BASE_URL}/ticker/24hr"
            params = {'symbol': symbol}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price_change': float(data['priceChange']),
                'price_change_percent': float(data['priceChangePercent']),
                'last_price': float(data['lastPrice']),
                'volume': float(data['volume']),
                'high_price': float(data['highPrice']),
                'low_price': float(data['lowPrice'])
            }
            
        except Exception as e:
            logger.debug(f"Pazar bilgisi çekme hatası {symbol}: {e}")
            return {}
    
    @classmethod
    def get_bulk_prices(cls, symbols: List[str]) -> Dict[str, float]:
        """
        Birden fazla sembolün güncel fiyatlarını toplu olarak getir
        
        Args:
            symbols (List[str]): Sembol listesi
            
        Returns:
            Dict[str, float]: Sembol -> fiyat mapping
        """
        try:
            url = f"{cls.BASE_URL}/ticker/price"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Sadece istenen sembollerin fiyatlarını döndür
            price_dict = {}
            for item in data:
                symbol = item['symbol']
                if symbol in symbols:
                    price_dict[symbol] = float(item['price'])
            
            logger.debug(f"Toplu fiyat çekimi: {len(price_dict)} sembol")
            return price_dict
            
        except Exception as e:
            logger.debug(f"Toplu fiyat çekme hatası: {e}")
            return {}
    
    @classmethod
    def get_symbol_precision(cls, symbol: str) -> Dict[str, int]:
        """
        Sembolün fiyat ve miktar hassasiyetini getir
        
        Args:
            symbol (str): Sembol adı
            
        Returns:
            Dict[str, int]: Hassasiyet bilgileri
        """
        try:
            url = f"{cls.BASE_URL}/exchangeInfo"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == symbol:
                    price_precision = symbol_info.get('pricePrecision', 2)
                    quantity_precision = symbol_info.get('quantityPrecision', 3)
                    
                    return {
                        'price_precision': price_precision,
                        'quantity_precision': quantity_precision
                    }
            
            # Varsayılan değerler
            return {
                'price_precision': 2,
                'quantity_precision': 3
            }
            
        except Exception as e:
            logger.debug(f"Hassasiyet bilgisi çekme hatası {symbol}: {e}")
            return {
                'price_precision': 2,
                'quantity_precision': 3
            }
    
    @classmethod
    def get_data_quality_info(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Çekilen verinin kalitesi hakkında bilgi ver
        
        Args:
            df (pd.DataFrame): OHLCV verisi
            
        Returns:
            Dict[str, Any]: Veri kalite bilgileri
        """
        if df is None or len(df) == 0:
            return {
                'candle_count': 0,
                'has_gaps': True,
                'data_quality': 'POOR',
                'suitable_for_supertrend': False
            }
        
        try:
            # Eksik veri kontrolü
            missing_data = df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any()
            
            # Zaman boşlukları kontrolü
            time_diffs = df['timestamp'].diff().dropna()
            expected_diff = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(hours=4)
            has_gaps = (time_diffs > expected_diff * 1.5).any()
            
            # Supertrend için uygunluk
            suitable_for_supertrend = len(df) >= 100 and not missing_data
            
            if suitable_for_supertrend and not has_gaps:
                quality = 'EXCELLENT'
            elif suitable_for_supertrend:
                quality = 'GOOD'
            elif len(df) >= 50:
                quality = 'FAIR'
            else:
                quality = 'POOR'
            
            return {
                'candle_count': len(df),
                'has_missing_data': missing_data,
                'has_gaps': has_gaps,
                'data_quality': quality,
                'suitable_for_supertrend': suitable_for_supertrend,
                'time_range': {
                    'start': df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M') if len(df) > 0 else None,
                    'end': df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M') if len(df) > 0 else None
                }
            }
            
        except Exception as e:
            logger.debug(f"Veri kalite analizi hatası: {e}")
            return {
                'candle_count': len(df),
                'data_quality': 'UNKNOWN',
                'suitable_for_supertrend': len(df) >= 50
            }