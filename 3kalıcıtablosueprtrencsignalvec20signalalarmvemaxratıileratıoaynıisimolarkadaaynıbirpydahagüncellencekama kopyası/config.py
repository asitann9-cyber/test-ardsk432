"""
Configuration settings for Supertrend Analysis System
🆕 YENİ: Dinamik C-Signal Threshold - Panel'den Ayarlanabilir Long/Short Sinyal Tespiti
✅ GÜNCELLEME: C-Signal threshold artık class attribute olarak dinamik ayarlanabilir
"""

import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class Config:
    """Ana konfigürasyon sınıfı"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production-2024')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Binance Configuration
    BINANCE_BASE_URL = "https://fapi.binance.com/fapi/v1"
    BINANCE_TIMEOUT = 15
    
    # Timeframe Limits for API calls
    TIMEFRAME_LIMITS = {
        '1m': 500, '5m': 500, '15m': 500, '30m': 500,
        '1h': 500, '2h': 500, '4h': 500, '1d': 500
    }
    
    # TradingView Configuration
    TRADINGVIEW_BASE_URL = "https://tr.tradingview.com/chart/"
    TRADINGVIEW_TIMEFRAME_MAP = {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240', '1d': '1D'
    }
    
    # Analysis Configuration
    PARALLEL_WORKERS = 5  # Paralel işleme worker sayısı
    RSI_PERIOD = 14  # RSI hesaplama periyodu
    MINIMUM_DATA_LENGTH = 15  # C-Signal için minimum veri uzunluğu
    
    # SUPERTREND PARAMETRELERI
    SUPERTREND_PARAMS = {
        'atr_period': 10,
        'multiplier': 3.0,
        'z_score_length': 14,
        'use_z_score': True,
        'momentum_rsi_period': 14,
        'top_symbols_count': 20,
    }
    
    # 🆕 DİNAMİK C-SIGNAL ALERT CONFIGURATION
    # Bu değerler artık panel üzerinden değiştirilebilir!
    C_SIGNAL_ALERT_THRESHOLD = 20.0  # Varsayılan: ±20 veya üzeri için alert
    
    # LONG ve SHORT threshold'lar otomatik olarak C_SIGNAL_ALERT_THRESHOLD'dan türetilir
    @classmethod
    def get_c_signal_long_threshold(cls):
        """LONG sinyal threshold'u döndür (pozitif değer)"""
        return abs(cls.C_SIGNAL_ALERT_THRESHOLD)
    
    @classmethod
    def get_c_signal_short_threshold(cls):
        """SHORT sinyal threshold'u döndür (negatif değer)"""
        return -abs(cls.C_SIGNAL_ALERT_THRESHOLD)
    
    @classmethod
    def update_c_signal_threshold(cls, new_threshold: float) -> bool:
        """
        C-Signal threshold'unu güncelle
        
        Args:
            new_threshold (float): Yeni threshold değeri (mutlak değer)
            
        Returns:
            bool: Güncelleme başarılı ise True
        """
        try:
            threshold_value = abs(float(new_threshold))
            
            # Validasyon: 0-100 arasında olmalı
            if threshold_value < 0 or threshold_value > 100:
                return False
            
            cls.C_SIGNAL_ALERT_THRESHOLD = threshold_value
            return True
            
        except (TypeError, ValueError):
            return False
    
    # Telegram Spam Prevention
    MIN_TELEGRAM_INTERVAL = 300  # 5 dakika (saniye cinsinden)
    
    # Auto Update Configuration
    AUTO_UPDATE_INTERVAL = 30  # 30 saniye
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    LOG_FILE = os.getenv('LOG_FILE', 'supertrend_system.log')
    
    # Database Configuration (gelecekte kullanılabilir)
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    # Cache Configuration (gelecekte kullanılabilir)
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'memory')
    CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 dakika
    
    @classmethod
    def validate_config(cls):
        """Konfigürasyon doğrulaması"""
        errors = []
        
        # Telegram konfigürasyonu isteğe bağlı ama uyarı ver
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("WARNING: TELEGRAM_BOT_TOKEN tanımlanmamış - Telegram bildirimleri çalışmayacak")
        
        if not cls.TELEGRAM_CHAT_ID:
            errors.append("WARNING: TELEGRAM_CHAT_ID tanımlanmamış - Telegram bildirimleri çalışmayacak")
        
        # Worker sayısı makul aralıkta olmalı
        if cls.PARALLEL_WORKERS < 1 or cls.PARALLEL_WORKERS > 20:
            errors.append("ERROR: PARALLEL_WORKERS 1-20 arasında olmalı")
        
        # Supertrend parametrelerini kontrol et
        if cls.SUPERTREND_PARAMS['atr_period'] <= 0:
            errors.append("ERROR: SUPERTREND_PARAMS atr_period pozitif olmalı")
        
        if cls.SUPERTREND_PARAMS['multiplier'] <= 0:
            errors.append("ERROR: SUPERTREND_PARAMS multiplier pozitif olmalı")
        
        if cls.SUPERTREND_PARAMS['z_score_length'] <= 0:
            errors.append("ERROR: SUPERTREND_PARAMS z_score_length pozitif olmalı")
        
        # 🆕 C-Signal threshold kontrolü - Dinamik olarak ayarlanabilir
        if cls.C_SIGNAL_ALERT_THRESHOLD < 0 or cls.C_SIGNAL_ALERT_THRESHOLD > 100:
            errors.append("ERROR: C_SIGNAL_ALERT_THRESHOLD 0-100 arasında olmalı")
        
        return errors
    
    @classmethod
    def get_summary(cls):
        """Konfigürasyon özetini döndür"""
        return {
            'telegram_configured': bool(cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID),
            'debug_mode': cls.DEBUG,
            'parallel_workers': cls.PARALLEL_WORKERS,
            'auto_update_interval': cls.AUTO_UPDATE_INTERVAL,
            'timeframe_limits': cls.TIMEFRAME_LIMITS,
            'supertrend_params': cls.SUPERTREND_PARAMS,
            'c_signal_threshold': cls.C_SIGNAL_ALERT_THRESHOLD,  # 🆕 Dinamik
            'c_signal_long_threshold': cls.get_c_signal_long_threshold(),  # 🆕
            'c_signal_short_threshold': cls.get_c_signal_short_threshold(),  # 🆕
            'log_level': cls.LOG_LEVEL
        }
    
    @classmethod
    def get_c_signal_config(cls):
        """C-Signal konfigürasyonunu döndür"""
        return {
            'alert_threshold': cls.C_SIGNAL_ALERT_THRESHOLD,
            'long_threshold': cls.get_c_signal_long_threshold(),
            'short_threshold': cls.get_c_signal_short_threshold(),
            'description': f'C-Signal >= +{cls.C_SIGNAL_ALERT_THRESHOLD} = LONG (L) | C-Signal <= {cls.get_c_signal_short_threshold()} = SHORT (S)'
        }

class ProductionConfig(Config):
    """Production ortamı konfigürasyonu"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Production'da mutlaka .env'den alınmalı
    
    # Production'da daha sıkı limitler
    PARALLEL_WORKERS = 3
    AUTO_UPDATE_INTERVAL = 60  # 1 dakika
    
    # Production'da daha az veri
    TIMEFRAME_LIMITS = {
        '1m': 200, '5m': 200, '15m': 200, '30m': 200,
        '1h': 200, '2h': 200, '4h': 200, '1d': 200
    }

class DevelopmentConfig(Config):
    """Development ortamı konfigürasyonu"""
    DEBUG = True
    
    # Development'ta daha gevşek limitler
    PARALLEL_WORKERS = 5
    AUTO_UPDATE_INTERVAL = 30  # 30 saniye

class TestingConfig(Config):
    """Test ortamı konfigürasyonu"""
    DEBUG = True
    TESTING = True
    
    # Test ortamında hızlı işlem
    PARALLEL_WORKERS = 2
    AUTO_UPDATE_INTERVAL = 10  # 10 saniye
    MIN_TELEGRAM_INTERVAL = 5  # 5 saniye
    
    # Test'te daha az veri
    TIMEFRAME_LIMITS = {
        '1m': 100, '5m': 100, '15m': 100, '30m': 100,
        '1h': 100, '2h': 100, '4h': 100, '1d': 100
    }

# Ortam bazlı konfigürasyon seçimi
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': Config
}

def get_config(env_name=None):
    """Ortam adına göre konfigürasyon döndür"""
    if env_name is None:
        env_name = os.getenv('FLASK_ENV', 'default')
    
    return config_map.get(env_name, Config)