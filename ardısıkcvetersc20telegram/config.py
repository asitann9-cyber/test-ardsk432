"""
Configuration settings for Consecutive Candles Analysis System
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
        '1m': 30, '5m': 30, '15m': 30, '30m': 30,
        '1h': 30, '2h': 30, '4h': 30, '1d': 30
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
    
    # Consecutive Candles Thresholds
    HIGH_CONSECUTIVE_THRESHOLD = 5  # Yüksek ardışık sayısı eşiği
    HIGH_PERCENTAGE_THRESHOLD = 10.0  # Yüksek yüzde değişim eşiği
    
    # Reverse Momentum Configuration
    STRONG_SIGNAL_THRESHOLD = 2.0  # Güçlü sinyal eşiği
    MEDIUM_SIGNAL_THRESHOLD = 1.0  # Orta sinyal eşiği
    
    # Telegram Spam Prevention
    MIN_TELEGRAM_INTERVAL = 300  # 5 dakika (saniye cinsinden)
    
    # Auto Update Configuration
    AUTO_UPDATE_INTERVAL = 30  # 30 saniye
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    LOG_FILE = os.getenv('LOG_FILE', 'consecutive_candles_system.log')
    
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
        
        # Threshold değerleri pozitif olmalı
        if cls.HIGH_CONSECUTIVE_THRESHOLD <= 0:
            errors.append("ERROR: HIGH_CONSECUTIVE_THRESHOLD pozitif olmalı")
        
        if cls.HIGH_PERCENTAGE_THRESHOLD <= 0:
            errors.append("ERROR: HIGH_PERCENTAGE_THRESHOLD pozitif olmalı")
        
        # Worker sayısı makul aralıkta olmalı
        if cls.PARALLEL_WORKERS < 1 or cls.PARALLEL_WORKERS > 20:
            errors.append("ERROR: PARALLEL_WORKERS 1-20 arasında olmalı")
        
        return errors
    
    @classmethod
    def get_summary(cls):
        """Konfigürasyon özetini döndür"""
        return {
            'telegram_configured': bool(cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID),
            'debug_mode': cls.DEBUG,
            'parallel_workers': cls.PARALLEL_WORKERS,
            'high_consecutive_threshold': cls.HIGH_CONSECUTIVE_THRESHOLD,
            'high_percentage_threshold': cls.HIGH_PERCENTAGE_THRESHOLD,
            'auto_update_interval': cls.AUTO_UPDATE_INTERVAL,
            'timeframe_limits': cls.TIMEFRAME_LIMITS,
            'log_level': cls.LOG_LEVEL
        }

class ProductionConfig(Config):
    """Production ortamı konfigürasyonu"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Production'da mutlaka .env'den alınmalı
    
    # Production'da daha sıkı limitler
    PARALLEL_WORKERS = 3
    AUTO_UPDATE_INTERVAL = 60  # 1 dakika

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