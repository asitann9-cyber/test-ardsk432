"""
Binance API servisleri
Binance vadeli işlemler API'si ile veri çekme ve sembol yönetimi
"""

import requests
import pandas as pd
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class BinanceService:
    """Binance API işlemleri için service sınıfı"""
    
    BASE_URL = "https://fapi.binance.com/fapi/v1"
    
    # Timeframe limitleri - SUPERTREND İÇİN ARTTIRILDI
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
        
        Args:
            symbol (str): Sembol adı (ör: BTCUSDT)
            timeframe (str): Zaman dilimi (ör: 4h)
            limit (Optional[int]): Veri limiti, None ise otomatik belirlenir
            
        Returns:
            Optional[pd.DataFrame]: OHLCV verileri içeren DataFrame
        """
        try:
            if limit is None:
                limit = cls.TIMEFRAME_LIMITS.get(timeframe, 500)
            
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