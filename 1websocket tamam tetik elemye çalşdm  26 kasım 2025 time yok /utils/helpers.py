"""
Helper Functions
Genel yardımcı fonksiyonlar ve utilities
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
import re

logger = logging.getLogger(__name__)

def create_tradingview_link(symbol: str, timeframe: str) -> str:
    """
    TradingView grafik linki oluştur
    
    Args:
        symbol (str): Sembol adı (örn: BTCUSDT)
        timeframe (str): Zaman dilimi (örn: 4h)
        
    Returns:
        str: TradingView chart URL
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

def format_time(timestamp: Union[str, datetime], format_type: str = 'full') -> str:
    """
    Zaman formatla
    
    Args:
        timestamp: Tarih/saat verisi
        format_type: Format tipi ('full', 'time', 'date')
        
    Returns:
        str: Formatlanmış tarih/saat
    """
    try:
        if isinstance(timestamp, str):
            # String ise parse et
            if ' ' in timestamp:
                dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                return timestamp
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            dt = datetime.now()
        
        if format_type == 'time':
            return dt.strftime('%H:%M:%S')
        elif format_type == 'date':
            return dt.strftime('%Y-%m-%d')
        else:  # 'full'
            return dt.strftime('%Y-%m-%d %H:%M:%S')
            
    except Exception as e:
        logger.debug(f"Zaman formatlama hatası: {e}")
        return datetime.now().strftime('%H:%M:%S')

def validate_symbol(symbol: str) -> bool:
    """
    Sembol formatının geçerliliğini kontrol et
    
    Args:
        symbol (str): Kontrol edilecek sembol
        
    Returns:
        bool: Geçerli ise True
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Temel format kontrolü
    if len(symbol) < 4 or len(symbol) > 12:
        return False
    
    # USDT ile biten semboller için
    if symbol.endswith('USDT'):
        base_symbol = symbol[:-4]  # USDT'yi çıkar
        if len(base_symbol) < 2:
            return False
        
        # Sadece harfler ve rakamlar
        if not re.match(r'^[A-Z0-9]+$', symbol):
            return False
        
        return True
    
    return False

def validate_timeframe(timeframe: str) -> bool:
    """
    Timeframe geçerliliğini kontrol et
    
    Args:
        timeframe (str): Kontrol edilecek timeframe
        
    Returns:
        bool: Geçerli ise True
    """
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
    return timeframe in valid_timeframes

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Yüzdelik değişim hesapla
    
    Args:
        old_value (float): Eski değer
        new_value (float): Yeni değer
        
    Returns:
        float: Yüzdelik değişim
    """
    try:
        if old_value == 0:
            return 0.0
        
        percentage_change = ((new_value - old_value) / old_value) * 100
        return round(percentage_change, 2)
        
    except Exception:
        return 0.0

def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """
    Sayıyı formatla
    
    Args:
        number: Formatlanacak sayı
        decimal_places: Ondalık basamak sayısı
        
    Returns:
        str: Formatlanmış sayı
    """
    try:
        if isinstance(number, (int, float)):
            return f"{number:.{decimal_places}f}"
        return str(number)
    except Exception:
        return "0.00"

def format_currency(amount: float, currency: str = 'USDT', decimal_places: int = 4) -> str:
    """
    Para birimi formatla
    
    Args:
        amount (float): Miktar
        currency (str): Para birimi
        decimal_places (int): Ondalık basamak
        
    Returns:
        str: Formatlanmış miktar
    """
    try:
        formatted_amount = f"{amount:.{decimal_places}f}"
        return f"{formatted_amount} {currency}"
    except Exception:
        return f"0.00 {currency}"

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Güvenli float dönüşümü
    
    Args:
        value: Dönüştürülecek değer
        default: Hata durumunda döndürülecek değer
        
    Returns:
        float: Dönüştürülmüş değer
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Güvenli int dönüşümü
    
    Args:
        value: Dönüştürülecek değer
        default: Hata durumunda döndürülecek değer
        
    Returns:
        int: Dönüştürülmüş değer
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    String'i belirtilen uzunlukta kes
    
    Args:
        text (str): Kesilecek metin
        max_length (int): Maksimum uzunluk
        suffix (str): Ek (üç nokta vs)
        
    Returns:
        str: Kesilmiş metin
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def get_signal_strength_emoji(strength: str) -> str:
    """
    Sinyal gücü için emoji getir
    
    Args:
        strength (str): Sinyal gücü
        
    Returns:
        str: Emoji
    """
    emoji_map = {
        'Strong': '🔥🔥🔥',
        'Medium': '🔥🔥',
        'Weak': '🔥',
        'None': '⚪'
    }
    
    return emoji_map.get(strength, '⚡')

def get_trend_emoji(trend_type: str) -> str:
    """
    Trend tipi için emoji getir
    
    Args:
        trend_type (str): Trend tipi
        
    Returns:
        str: Emoji
    """
    emoji_map = {
        'Long': '🟢',
        'Short': '🔴', 
        'C↑': '📺',
        'C↓': '📻',
        'None': '⚪'
    }
    
    return emoji_map.get(trend_type, '⚪')

def create_analysis_summary(results: list, timeframe: str) -> Dict[str, Any]:
    """
    Analiz özeti oluştur
    
    Args:
        results (list): Analiz sonuçları
        timeframe (str): Zaman dilimi
        
    Returns:
        Dict[str, Any]: Özet bilgileri
    """
    if not results:
        return {
            'total_count': 0,
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
        'total_count': len(results),
        'long_count': long_count,
        'short_count': short_count,
        'high_consecutive_count': high_consecutive_count,
        'max_consecutive': max_consecutive,
        'timeframe': timeframe,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def sanitize_telegram_message(message: str) -> str:
    """
    Telegram mesajını temizle ve güvenli hale getir
    
    Args:
        message (str): Ham mesaj
        
    Returns:
        str: Temizlenmiş mesaj
    """
    if not message:
        return ""
    
    # HTML karakterlerini escape et
    message = message.replace('&', '&amp;')
    message = message.replace('<', '&lt;')
    message = message.replace('>', '&gt;')
    
    # Çok uzun mesajları kes
    if len(message) > 4000:  # Telegram limiti 4096
        message = message[:3900] + "...\n\n[Mesaj çok uzun, kısaltıldı]"
    
    return message

def generate_unique_id() -> str:
    """
    Benzersiz ID oluştur
    
    Returns:
        str: Unique ID
    """
    from datetime import datetime
    import hashlib
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]

def is_market_hours() -> bool:
    """
    Kripto pazar 7/24 açık olduğu için her zaman True
    Gelecekte farklı pazar saatleri eklenebilir
    
    Returns:
        bool: Pazar açık ise True
    """
    return True

def format_large_number(number: Union[int, float]) -> str:
    """
    Büyük sayıları K, M, B formatında göster
    
    Args:
        number: Formatlanacak sayı
        
    Returns:
        str: Formatlanmış sayı (örn: 1.5K, 2.3M)
    """
    try:
        num = float(number)
        
        if abs(num) >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif abs(num) >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:.1f}"
            
    except (TypeError, ValueError):
        return "0"

def calculate_time_difference(start_time: str, end_time: Optional[str] = None) -> str:
    """
    İki zaman arasındaki farkı hesapla
    
    Args:
        start_time (str): Başlangıç zamanı
        end_time (Optional[str]): Bitiş zamanı (None ise şu anki zaman)
        
    Returns:
        str: Zaman farkı açıklaması
    """
    try:
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') if end_time else datetime.now()
        
        diff = end_dt - start_dt
        
        if diff.days > 0:
            return f"{diff.days} gün önce"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} saat önce"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} dakika önce"
        else:
            return "Az önce"
            
    except Exception:
        return "Bilinmiyor"

def validate_percentage(percentage: float, min_val: float = -100.0, max_val: float = 1000.0) -> bool:
    """
    Yüzdelik değerin makul aralıkta olup olmadığını kontrol et
    
    Args:
        percentage (float): Kontrol edilecek yüzde
        min_val (float): Minimum değer
        max_val (float): Maksimum değer
        
    Returns:
        bool: Makul aralıkta ise True
    """
    try:
        return min_val <= float(percentage) <= max_val
    except (TypeError, ValueError):
        return False

def get_consecutive_color_class(consecutive_type: str, consecutive_count: int) -> str:
    """
    Ardışık mum tipi ve sayısına göre CSS class döndür
    
    Args:
        consecutive_type (str): Ardışık tip (Long/Short)
        consecutive_count (int): Ardışık sayı
        
    Returns:
        str: CSS class adı
    """
    base_class = 'consecutive-long' if consecutive_type == 'Long' else 'consecutive-short'
    
    if consecutive_count >= 7:
        return f"{base_class} high-priority"
    elif consecutive_count >= 5:
        return f"{base_class} medium-priority"
    else:
        return base_class

def create_error_response(error_message: str, error_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Standart hata response'u oluştur
    
    Args:
        error_message (str): Hata mesajı
        error_code (Optional[str]): Hata kodu
        
    Returns:
        Dict[str, Any]: Hata response'u
    """
    response = {
        'success': False,
        'error': error_message,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if error_code:
        response['error_code'] = error_code
    
    return response

def create_success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
    """
    Standart başarı response'u oluştur
    
    Args:
        data: Response verisi
        message (Optional[str]): Başarı mesajı
        
    Returns:
        Dict[str, Any]: Başarı response'u
    """
    response = {
        'success': True,
        'data': data,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if message:
        response['message'] = message
    
    return response