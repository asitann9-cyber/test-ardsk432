"""
🛠️ Yardımcı Fonksiyonlar

"""

import math
import pandas as pd
import numpy as np


def gauss_sum(n: float) -> float:
    """
    Gauss toplamı hesaplama: n*(n+1)/2
    
    Args:
        n (float): Toplam sayısı
        
    Returns:
        float: Gauss toplamı
    """
    return n * (n + 1) / 2.0


def safe_log(value, base=math.e, min_value=1e-10):
    """
    Güvenli logaritma hesaplama - sıfır ve negatif değerler için koruma
    
    Args:
        value: Logaritması alınacak değer
        base: Logaritma tabanı (varsayılan: e)
        min_value: Minimum değer koruması
        
    Returns:
        float: Logaritma değeri
    """
    if value <= 0:
        return math.log(min_value, base) if base != math.e else math.log(min_value)
    return math.log(value, base) if base != math.e else math.log(value)


def pine_sma(src, length):
    """
    Pine Script ta.sma eşdeğeri - Basit hareketli ortalama
    
    Args:
        src (pd.Series): Kaynak veri serisi
        length (int): Ortalama periyodu
        
    Returns:
        pd.Series: Hareketli ortalama serisi
    """
    result = []
    valid_values = []
    
    for i in range(len(src)):
        if not pd.isna(src.iloc[i]):
            valid_values.append(src.iloc[i])
        
        if len(valid_values) > length:
            valid_values = valid_values[-length:]
        
        if len(valid_values) > 0:
            result.append(np.mean(valid_values))
        else:
            result.append(np.nan)
    
    return pd.Series(result, index=src.index)


def crossunder(a, b): 
    """
    Pine Script crossunder fonksiyonu
    a serisinin b serisini aşağı doğru kesmesi
    """
    return (a < b) & (a.shift(1) >= b.shift(1))


def crossover(a, b):  
    """
    Pine Script crossover fonksiyonu
    a serisinin b serisini yukarı doğru kesmesi
    """
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross(a, b):
    """
    Pine Script cross fonksiyonu
    a ve b serilerinin kesişmesi (her iki yönde)
    """
    return ((a > b) & (a.shift(1) <= b.shift(1))) | ((a < b) & (a.shift(1) >= b.shift(1)))


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    DataFrame'in gerekli sütunlara sahip olup olmadığını kontrol et
    
    Args:
        df (pd.DataFrame): Kontrol edilecek DataFrame
        required_columns (list): Gerekli sütun listesi
        
    Returns:
        bool: True eğer tüm sütunlar mevcutsa
    """
    if df is None or df.empty:
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0


def safe_division(numerator, denominator, default=0.0):
    """
    Güvenli bölme işlemi - sıfıra bölme koruması
    
    Args:
        numerator: Bölünen
        denominator: Bölen
        default: Sıfıra bölme durumunda döndürülecek değer
        
    Returns:
        float: Bölme sonucu veya default değer
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default


def format_number(number, decimals=2):
    """
    Sayıyı belirtilen ondalık basamakla formatla
    
    Args:
        number: Formatlanacak sayı
        decimals: Ondalık basamak sayısı
        
    Returns:
        str: Formatlanmış sayı string'i
    """
    try:
        if pd.isna(number) or number is None:
            return "N/A"
        return f"{float(number):.{decimals}f}"
    except:
        return "N/A"


def calculate_percentage_change(old_value, new_value):
    """
    İki değer arasındaki yüzde değişimi hesapla
    
    Args:
        old_value: Eski değer
        new_value: Yeni değer
        
    Returns:
        float: Yüzde değişim
    """
    if old_value == 0 or pd.isna(old_value) or pd.isna(new_value):
        return 0.0
    
    return ((new_value - old_value) / old_value) * 100


def clamp(value, min_value, max_value):
    """
    Değeri belirtilen aralıkta sınırla
    
    Args:
        value: Sınırlanacak değer
        min_value: Minimum değer
        max_value: Maksimum değer
        
    Returns:
        Sınırlanmış değer
    """
    return max(min_value, min(max_value, value))


def is_valid_symbol(symbol: str) -> bool:
    """
    Kripto sembolünün geçerli olup olmadığını kontrol et
    
    Args:
        symbol (str): Kontrol edilecek sembol
        
    Returns:
        bool: True eğer geçerliyse
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Temel kontroller
    if len(symbol) < 6 or len(symbol) > 15:
        return False
    
    # USDT ile bitip bitmediğini kontrol et
    if not symbol.endswith('USDT'):
        return False
    
    return True


def clean_numeric_value(value, default=0.0):
    """
    Numerik değeri temizle ve geçerli hale getir
    
    Args:
        value: Temizlenecek değer
        default: Geçersiz durumda döndürülecek default değer
        
    Returns:
        float: Temizlenmiş numerik değer
    """
    try:
        if pd.isna(value) or value is None:
            return default
        
        if isinstance(value, str):
            # String'den sayısal olmayan karakterleri temizle
            cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
            return float(cleaned) if cleaned else default
        
        return float(value)
    except:
        return default