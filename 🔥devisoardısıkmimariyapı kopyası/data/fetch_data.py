"""
🌐 Veri Çekme Modülü - DÜZELTME (HATASIZ)
✅ Testnet/Mainnet endpoint birleştirmesi
✅ Live trading ile aynı environment kullanımı
✅ Tüm type hint'ler düzeltildi
"""

import time
import logging
import pandas as pd
from typing import List, Optional, Dict
import requests

from config import (
    LOCAL_TZ, LIMIT, SYMBOL_LIMIT, EXCHANGE_INFO, KLINES, 
    TIMEOUT, REQ_SLEEP, create_session, BASE, ENVIRONMENT
)

logger = logging.getLogger("crypto-analytics")

# Global session
session = create_session()


def get_usdt_perp_symbols() -> List[str]:
    """
    🔥 DÜZELTME: USDT-M perpetual sembollerini aynı environment'tan al
    
    Returns:
        List[str]: Aktif USDT perpetual sembolleri
    """
    try:
        logger.info(f"📡 Sembol listesi alınıyor - Environment: {ENVIRONMENT}")
        logger.info(f"🌐 Exchange info URL: {EXCHANGE_INFO}")
        
        with session.get(EXCHANGE_INFO, timeout=TIMEOUT) as r:
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
            
        logger.info(f"📋 {len(symbols)} USDT perpetual sembol yüklendi ({ENVIRONMENT})")
        
        # Örnek sembolleri logla
        if symbols:
            sample = symbols[:10]
            logger.info(f"📋 Örnek semboller: {', '.join(sample)}")
            
        return symbols
        
    except Exception as e:
        logger.error(f"Sembol listesi alma hatası: {e}")
        return []


def fetch_klines(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """
    🔥 DÜZELTME: Belirtilen sembol için kline verisi çek (unified endpoint)
    
    Args:
        symbol (str): Trading sembolü (örn: BTCUSDT)
        interval (str): Zaman dilimi (1m, 5m, 15m, 1h, 4h)
        
    Returns:
        Optional[pd.DataFrame]: OHLCV verileri veya None
    """
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': LIMIT
        }
        
        logger.debug(f"📊 Kline verisi çekiliyor: {symbol} {interval} - URL: {KLINES}")
        
        with session.get(KLINES, params=params, timeout=TIMEOUT) as r:
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
    🔥 DÜZELTME: Unified endpoint ile fiyat alma
    
    Args:
        symbol (str): Trading sembolü
        
    Returns:
        Optional[float]: Mevcut fiyat veya None
    """
    try:
        time.sleep(0.02)  # Rate limit koruması
        
        # 🔥 DÜZELTME: Config'ten BASE URL kullan (auto testnet/mainnet)
        url = f"{BASE}/fapi/v1/ticker/price?symbol={symbol}"
        
        response = requests.get(url, timeout=10)
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


def fetch_multiple_symbols(symbols: List[str], interval: str) -> Dict[str, pd.DataFrame]:
    """
    🔥 DÜZELTME: Birden fazla sembol için veri çek (unified endpoint)
    
    Args:
        symbols (List[str]): Sembol listesi
        interval (str): Zaman dilimi
        
    Returns:
        Dict[str, pd.DataFrame]: Sembol -> DataFrame mapping
    """
    results: Dict[str, pd.DataFrame] = {}
    failed_count = 0
    
    logger.info(f"📡 {len(symbols)} sembol için veri çekiliyor ({ENVIRONMENT})...")
    logger.info(f"🌐 Base URL: {BASE}")
    
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
    
    logger.info(f"✅ Veri çekme tamamlandı ({ENVIRONMENT}): {success_count}/{len(symbols)} başarılı ({success_rate:.1f}%)")
    
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


def get_market_summary() -> Dict[str, any]:
    """
    🔥 DÜZELTME: Genel piyasa özetini al (unified endpoint)
    
    Returns:
        Dict[str, any]: Piyasa özet bilgileri
    """
    try:
        logger.info(f"📊 Piyasa özeti alınıyor ({ENVIRONMENT})...")
        
        symbols = get_usdt_perp_symbols()
        
        if not symbols:
            return {
                'error': 'Sembol listesi alınamadı',
                'environment': ENVIRONMENT,
                'base_url': BASE
            }
        
        # İlk 50 sembol için fiyat al (test amaçlı)
        test_symbols = symbols[:50]
        price_data: Dict[str, float] = {}
        
        logger.info(f"💰 {len(test_symbols)} sembol için fiyat kontrolü...")
        
        for symbol in test_symbols:
            price = get_current_price(symbol)
            if price:
                price_data[symbol] = price
        
        success_rate = (len(price_data) / len(test_symbols)) * 100 if test_symbols else 0
        
        logger.info(f"✅ Piyasa özeti tamamlandı: {len(price_data)}/{len(test_symbols)} fiyat alındı ({success_rate:.1f}%)")
        
        return {
            'total_symbols': len(symbols),
            'active_symbols': len(price_data),
            'sample_prices': dict(list(price_data.items())[:10]),
            'success_rate': success_rate,
            'environment': ENVIRONMENT,
            'base_url': BASE,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Piyasa özeti hatası: {e}")
        return {
            'error': str(e), 
            'environment': ENVIRONMENT,
            'base_url': BASE,
            'status': 'failed'
        }


def test_api_connectivity() -> Dict[str, any]:
    """
    🔥 YENİ: API bağlantısını test et
    
    Returns:
        Dict[str, any]: Bağlantı test sonuçları
    """
    test_results: Dict[str, any] = {
        'environment': ENVIRONMENT,
        'base_url': BASE,
        'exchange_info': False,
        'klines': False,
        'ticker': False,
        'sample_symbol': None,
        'errors': []
    }
    
    try:
        # Exchange info testi
        logger.info("🧪 Exchange info testi...")
        response = requests.get(EXCHANGE_INFO, timeout=TIMEOUT)
        if response.status_code == 200:
            test_results['exchange_info'] = True
            logger.info("✅ Exchange info başarılı")
        else:
            test_results['errors'].append(f"Exchange info failed: {response.status_code}")
            
    except Exception as e:
        test_results['errors'].append(f"Exchange info error: {e}")
        logger.error(f"❌ Exchange info hatası: {e}")
    
    try:
        # Sample symbol al
        symbols = get_usdt_perp_symbols()
        if symbols:
            sample_symbol = symbols[0]  # İlk sembol
            test_results['sample_symbol'] = sample_symbol
            
            # Klines testi
            logger.info(f"🧪 Klines testi: {sample_symbol}")
            df = fetch_klines(sample_symbol, "1h")
            if df is not None and not df.empty:
                test_results['klines'] = True
                logger.info("✅ Klines başarılı")
            else:
                test_results['errors'].append("Klines returned empty data")
            
            # Ticker testi
            logger.info(f"🧪 Ticker testi: {sample_symbol}")
            price = get_current_price(sample_symbol)
            if price is not None:
                test_results['ticker'] = True
                test_results['sample_price'] = price
                logger.info(f"✅ Ticker başarılı: ${price}")
            else:
                test_results['errors'].append("Ticker returned no price")
                
    except Exception as e:
        test_results['errors'].append(f"Symbol test error: {e}")
        logger.error(f"❌ Sembol testi hatası: {e}")
    
    # Genel durum
    all_tests = [test_results['exchange_info'], test_results['klines'], test_results['ticker']]
    test_results['overall_success'] = all(all_tests)
    test_results['success_rate'] = sum(all_tests) / len(all_tests) * 100
    
    logger.info(f"🧪 API test tamamlandı: {test_results['success_rate']:.0f}% başarı")
    
    return test_results


def check_symbol_availability(symbols: List[str]) -> Dict[str, bool]:
    """
    🔥 YENİ: Sembollerin mevcut endpoint'te kullanılabilir olup olmadığını kontrol et
    
    Args:
        symbols (List[str]): Kontrol edilecek sembol listesi
        
    Returns:
        Dict[str, bool]: Sembol -> Mevcut mapping
    """
    logger.info(f"🔍 {len(symbols)} sembol kullanılabilirlik kontrolü ({ENVIRONMENT})...")
    
    availability: Dict[str, bool] = {}
    available_count = 0
    
    # Exchange info'dan mevcut sembolleri al
    try:
        available_symbols = set(get_usdt_perp_symbols())
    except Exception as e:
        logger.error(f"❌ Kullanılabilir semboller alınamadı: {e}")
        return {symbol: False for symbol in symbols}
    
    for symbol in symbols:
        is_available = symbol in available_symbols
        availability[symbol] = is_available
        if is_available:
            available_count += 1
    
    success_rate = (available_count / len(symbols)) * 100 if symbols else 0
    
    logger.info(f"✅ Kullanılabilirlik kontrolü: {available_count}/{len(symbols)} mevcut ({success_rate:.1f}%)")
    
    # Mevcut olmayanları logla
    unavailable = [s for s, avail in availability.items() if not avail]
    if unavailable:
        logger.warning(f"⚠️ Mevcut olmayan semboller: {', '.join(unavailable[:10])}")
        if len(unavailable) > 10:
            logger.warning(f"⚠️ ... ve {len(unavailable) - 10} sembol daha")
    
    return availability


def get_environment_info() -> Dict[str, any]:
    """
    🔥 YENİ: Environment bilgilerini döndür
    
    Returns:
        Dict[str, any]: Environment detayları
    """
    return {
        'environment': ENVIRONMENT,
        'base_url': BASE,
        'exchange_info_url': EXCHANGE_INFO,
        'klines_url': KLINES,
        'is_testnet': ENVIRONMENT == 'testnet',
        'is_mainnet': ENVIRONMENT == 'mainnet',
        'timeout': TIMEOUT,
        'request_sleep': REQ_SLEEP,
        'limit': LIMIT
    }