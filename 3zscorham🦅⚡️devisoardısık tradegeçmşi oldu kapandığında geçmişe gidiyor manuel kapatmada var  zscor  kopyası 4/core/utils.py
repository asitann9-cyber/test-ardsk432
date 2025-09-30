"""
🛠️ Yardımcı Fonksiyonlar
Pine Script uyumlu fonksiyonlar ve genel yardımcılar
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
    Handles NaN values properly like Pine Script
    
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
    
    Args:
        a (pd.Series): Birinci seri
        b (pd.Series): İkinci seri
        
    Returns:
        pd.Series: Boolean seri (True = crossunder oluştu)
    """
    return (a < b) & (a.shift(1) >= b.shift(1))


def crossover(a, b):  
    """
    Pine Script crossover fonksiyonu
    a serisinin b serisini yukarı doğru kesmesi
    
    Args:
        a (pd.Series): Birinci seri
        b (pd.Series): İkinci seri
        
    Returns:
        pd.Series: Boolean seri (True = crossover oluştu)
    """
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross(a, b):
    """
    🔥 YENİ: Pine Script cross fonksiyonu
    a ve b serilerinin kesişmesi (her iki yönde)
    
    Args:
        a (pd.Series): Birinci seri
        b (pd.Series): İkinci seri
        
    Returns:
        pd.Series: Boolean seri (True = herhangi bir yönde kesişme oluştu)
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


# 🔥 YENİ: Pine Script yardımcı fonksiyonları
def na_values(series: pd.Series) -> bool:
    """
    Pine Script na() kontrolü
    
    Args:
        series (pd.Series): Kontrol edilecek seri
        
    Returns:
        bool: True eğer son değer NaN ise
    """
    return pd.isna(series.iloc[-1]) if len(series) > 0 else True


def nz(value, replacement=0.0):
    """
    Pine Script nz() fonksiyonu - NaN değerleri değiştir
    
    Args:
        value: Kontrol edilecek değer
        replacement: NaN durumunda kullanılacak değer
        
    Returns:
        Temizlenmiş değer
    """
    return replacement if pd.isna(value) else value


def highest(series: pd.Series, length: int) -> float:
    """
    Pine Script ta.highest() fonksiyonu
    Belirtilen periyottaki en yüksek değeri döndürür
    
    Args:
        series (pd.Series): Veri serisi
        length (int): Periyot uzunluğu
        
    Returns:
        float: En yüksek değer
    """
    if len(series) < length:
        return series.max() if len(series) > 0 else 0.0
    
    return series.tail(length).max()


def lowest(series: pd.Series, length: int) -> float:
    """
    Pine Script ta.lowest() fonksiyonu
    Belirtilen periyottaki en düşük değeri döndürür
    
    Args:
        series (pd.Series): Veri serisi
        length (int): Periyot uzunluğu
        
    Returns:
        float: En düşük değer
    """
    if len(series) < length:
        return series.min() if len(series) > 0 else 0.0
    
    return series.tail(length).min()


def change(series: pd.Series, length: int = 1) -> pd.Series:
    """
    Pine Script ta.change() fonksiyonu
    Belirtilen periyot öncesine göre değişim
    
    Args:
        series (pd.Series): Veri serisi  
        length (int): Geriye bakılacak periyot
        
    Returns:
        pd.Series: Değişim serisi
    """
    return series - series.shift(length)


def sma(series: pd.Series, length: int) -> pd.Series:
    """
    Pine Script ta.sma() fonksiyonu alternatifi
    Standart pandas rolling mean kullanır
    
    Args:
        series (pd.Series): Veri serisi
        length (int): Periyot uzunluğu
        
    Returns:
        pd.Series: Hareketli ortalama serisi
    """
    return series.rolling(window=length, min_periods=1).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """
    Pine Script ta.ema() fonksiyonu
    Exponential Moving Average
    
    Args:
        series (pd.Series): Veri serisi
        length (int): Periyot uzunluğu
        
    Returns:
        pd.Series: Exponential hareketli ortalama serisi
    """
    return series.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Pine Script ta.rsi() fonksiyonu
    Relative Strength Index
    
    Args:
        close (pd.Series): Kapanış fiyatları
        length (int): RSI periyodu
        
    Returns:
        pd.Series: RSI değerleri
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def stdev(series: pd.Series, length: int) -> pd.Series:
    """
    Pine Script ta.stdev() fonksiyonu
    Standard Deviation
    
    Args:
        series (pd.Series): Veri serisi
        length (int): Periyot uzunluğu
        
    Returns:
        pd.Series: Standart sapma serisi
    """
    return series.rolling(window=length).std()


def correlation(x: pd.Series, y: pd.Series, length: int) -> pd.Series:
    """
    Pine Script ta.correlation() fonksiyonu
    İki seri arasındaki korelasyon
    
    Args:
        x (pd.Series): Birinci seri
        y (pd.Series): İkinci seri
        length (int): Periyot uzunluğu
        
    Returns:
        pd.Series: Korelasyon serisi
    """
    return x.rolling(window=length).corr(y)


# 🔥 YENİ: Deviso spesifik yardımcı fonksiyonlar
def is_bullish_candle(open_price: float, close_price: float) -> bool:
    """
    Mum yükseliş mumunu mu kontrol et
    
    Args:
        open_price (float): Açılış fiyatı
        close_price (float): Kapanış fiyatı
        
    Returns:
        bool: True eğer yükseliş mumu ise
    """
    return close_price > open_price


def is_bearish_candle(open_price: float, close_price: float) -> bool:
    """
    Mum düşüş mumunu mu kontrol et
    
    Args:
        open_price (float): Açılış fiyatı
        close_price (float): Kapanış fiyatı
        
    Returns:
        bool: True eğer düşüş mumu ise
    """
    return close_price < open_price


def candle_body_size(open_price: float, close_price: float) -> float:
    """
    Mum gövde boyutunu hesapla
    
    Args:
        open_price (float): Açılış fiyatı
        close_price (float): Kapanış fiyatı
        
    Returns:
        float: Gövde boyutu (mutlak değer)
    """
    return abs(close_price - open_price)


def candle_range(high_price: float, low_price: float) -> float:
    """
    Mum toplam aralığını hesapla
    
    Args:
        high_price (float): En yüksek fiyat
        low_price (float): En düşük fiyat
        
    Returns:
        float: Toplam aralık
    """
    return high_price - low_price