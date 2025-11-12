"""
🌐 Veri Çekme Modülü
Binance API'den kripto veri çekme ve yönetimi
🔥 YENİ: Veri kaynağı seçilebilir (Mainnet/Testnet)
"""

import time
import logging
import pandas as pd
from typing import List, Optional
import requests

from config import (
    LOCAL_TZ, LIMIT, SYMBOL_LIMIT, TIMEOUT, REQ_SLEEP, create_session,
    DATA_BASE, DATA_ENVIRONMENT  # 🔥 YENİ: Veri kaynağı için
)

logger = logging.getLogger("crypto-analytics")

# Global session
session = create_session()


def get_data_url() -> str:
    """
    🔥 YENİ: Güncel veri kaynağı URL'ini al
    Runtime'da config.DATA_BASE değişebilir
    
    Returns:
        str: Güncel DATA_BASE URL'i
    """
    from config import DATA_BASE
    return DATA_BASE


def get_usdt_perp_symbols() -> List[str]:
    """
    🔥 GÜNCELLEME: Seçilen veri kaynağından USDT-M perpetual sembollerini al
    
    Returns:
        List[str]: Aktif USDT perpetual sembolleri
    """
    try:
        data_url = get_data_url()
        exchange_info_url = f"{data_url}/fapi/v1/exchangeInfo"
        
        logger.debug(f"📡 Sembol listesi çekiliyor: {exchange_info_url}")
        
        with session.get(exchange_info_url, timeout=TIMEOUT) as r:
            r.raise_for_status()
            data = r.json()
        
        symbols = []
        for s in data.get('symbols', []):
            if (
                s.get('quoteAsset') == 'USDT' and
                s.get('status') == 'TRADING' and
                s.get('contractType') == 'PERPETUAL'
            ):
                symbols.append(s['symbol'])
        
        symbols = sorted(set(symbols))
        if SYMBOL_LIMIT:
            symbols = symbols[:SYMBOL_LIMIT]
        
        source_tag = "🚀 MAINNET" if DATA_ENVIRONMENT == 'mainnet' else "🧪 TESTNET"
        logger.info(f"📋 {len(symbols)} USDT perpetual sembol yüklendi ({source_tag})")
        return symbols
        
    except Exception as e:
        logger.error(f"Sembol listesi alma hatası: {e}")
        return []


def fetch_klines(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """
    🔥 GÜNCELLEME: Seçilen veri kaynağından kline verisi çek
    
    Args:
        symbol (str): Trading sembolü (örn: BTCUSDT)
        interval (str): Zaman dilimi (1m, 5m, 15m, 1h, 4h)
        
    Returns:
        Optional[pd.DataFrame]: OHLCV verileri veya None
    """
    try:
        data_url = get_data_url()
        klines_url = f"{data_url}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': LIMIT
        }
        
        with session.get(klines_url, params=params, timeout=TIMEOUT) as r:
            r.raise_for_status()
            arr = r.json()
        
        if not arr:
            logger.warning(f"⚠️ {symbol} için veri alınamadı")
            return None
        
        cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'taker_base', 'taker_quote', 'ignore'
        ]
        
        df = pd.DataFrame(arr, columns=cols)
        
        # Numerik dönüşümler
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Zaman dönüşümleri
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ)
        
        # Sadece gerekli sütunları döndür
        result_df = df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
        
        # Veri doğrulama
        if result_df.empty or result_df.isnull().all().any():
            logger.warning(f"⚠️ {symbol} için geçersiz veri")
            return None
            
        return result_df
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"API hatası {symbol}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Veri çekme hatası {symbol}: {e}")
        return None


def get_current_price(symbol: str) -> Optional[float]:
    """
    🔥 GÜNCELLEME: Seçilen veri kaynağından fiyat al
    
    Args:
        symbol (str): Trading sembolü
        
    Returns:
        Optional[float]: Mevcut fiyat veya None
    """
    try:
        time.sleep(0.02)  # Rate limit koruması
        
        data_url = get_data_url()
        price_url = f"{data_url}/fapi/v1/ticker/price"
        
        params = {'symbol': symbol}
        
        response = requests.get(price_url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        price = float(data['price'])
        
        if price <= 0:
            logger.warning(f"⚠️ {symbol} için geçersiz fiyat: {price}")
            return None
            
        return price
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Fiyat API hatası {symbol}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Fiyat parse hatası {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Fiyat alma hatası {symbol}: {e}")
        return None


def fetch_multiple_symbols(symbols: List[str], interval: str) -> dict:
    """
    🔥 GÜNCELLEME: Birden fazla sembol için veri çek (seçilen kaynaktan)
    
    Args:
        symbols (List[str]): Sembol listesi
        interval (str): Zaman dilimi
        
    Returns:
        dict: Sembol -> DataFrame mapping
    """
    results = {}
    failed_count = 0
    
    source_tag = "🚀 MAINNET" if DATA_ENVIRONMENT == 'mainnet' else "🧪 TESTNET"
    logger.info(f"📡 {len(symbols)} sembol için veri çekiliyor ({source_tag})...")
    
    for i, symbol in enumerate(symbols):
        try:
            # Rate limiting
            time.sleep(REQ_SLEEP)
            
            df = fetch_klines(symbol, interval)
            if df is not None:
                results[symbol] = df
            else:
                failed_count += 1
            
            # Progress log
            if (i + 1) % 100 == 0:
                success_rate = ((i + 1 - failed_count) / (i + 1)) * 100
                logger.info(f"📊 İlerleme: {i+1}/{len(symbols)} - Başarı: {success_rate:.1f}%")
                
        except Exception as e:
            logger.debug(f"Sembol hatası {symbol}: {e}")
            failed_count += 1
    
    success_count = len(results)
    success_rate = (success_count / len(symbols)) * 100 if symbols else 0
    
    logger.info(f"✅ Veri çekme tamamlandı: {success_count}/{len(symbols)} başarılı ({success_rate:.1f}%) ({source_tag})")
    
    if failed_count > 0:
        logger.warning(f"⚠️ {failed_count} sembol için veri alınamadı")
    
    return results


def validate_ohlcv_data(df: pd.DataFrame, symbol: str = "") -> bool:
    """
    OHLCV verisinin geçerliliğini kontrol et
    
    Args:
        df (pd.DataFrame): Kontrol edilecek DataFrame
        symbol (str): Sembol adı (log için)
        
    Returns:
        bool: True eğer veri geçerliyse
    """
    if df is None or df.empty:
        return False
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Sütun kontrolü
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"⚠️ {symbol} eksik sütun: {col}")
            return False
    
    # Veri tipi kontrolü
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"⚠️ {symbol} geçersiz veri tipi: {col}")
            return False
    
    # OHLC mantık kontrolü
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).any()
    
    if invalid_ohlc:
        logger.warning(f"⚠️ {symbol} OHLC mantık hatası")
        return False
    
    # Negatif volume kontrolü
    if (df['volume'] < 0).any():
        logger.warning(f"⚠️ {symbol} negatif volume")
        return False
    
    return True


def get_market_summary() -> dict:
    """
    🔥 GÜNCELLEME: Seçilen veri kaynağından piyasa özetini al
    
    Returns:
        dict: Piyasa özet bilgileri
    """
    try:
        symbols = get_usdt_perp_symbols()
        
        if not symbols:
            return {'error': 'Sembol listesi alınamadı'}
        
        # İlk 50 sembol için fiyat al (test amaçlı)
        test_symbols = symbols[:50]
        price_data = {}
        
        for symbol in test_symbols:
            price = get_current_price(symbol)
            if price:
                price_data[symbol] = price
        
        source_tag = "mainnet" if DATA_ENVIRONMENT == 'mainnet' else "testnet"
        
        return {
            'total_symbols': len(symbols),
            'active_symbols': len(price_data),
            'sample_prices': dict(list(price_data.items())[:10]),
            'data_source': source_tag,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Piyasa özeti hatası: {e}")
        return {'error': str(e), 'status': 'failed'}


def get_data_source_info() -> dict:
    """
    🔥 YENİ: Mevcut veri kaynağı bilgisini al
    
    Returns:
        dict: Veri kaynağı bilgileri
    """
    data_url = get_data_url()
    
    return {
        'environment': DATA_ENVIRONMENT,
        'base_url': data_url,
        'is_mainnet': DATA_ENVIRONMENT == 'mainnet',
        'is_testnet': DATA_ENVIRONMENT == 'testnet',
        'display_name': '🚀 Binance Mainnet' if DATA_ENVIRONMENT == 'mainnet' else '🧪 Binance Testnet'
    }