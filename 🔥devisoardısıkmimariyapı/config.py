"""
🔧 Kripto AI Sistemi - Konfigürasyon Dosyası (Deviso Entegrasyonu)
Tüm ayarlar ve sabitler burada tanımlanır + Deviso Bot ayarları
"""

import os
import pytz
import logging
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 🔥 .ENV DOSYASI YÜKLE
load_dotenv()

# =============================================================================
# 🌐 TEMEL AYARLAR
# =============================================================================

LOCAL_TZ = pytz.timezone("Europe/Istanbul")
DEFAULT_TIMEFRAME = "15m"
LIMIT = 500 
SYMBOL_LIMIT = None  

# =============================================================================
# 🔑 LIVE TRADING AYARLARI
# =============================================================================

# Binance API Keys (.env dosyasından)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'testnet')  # testnet veya mainnet

# Trading Parametreleri
LIVE_TRADING_ACTIVE = False
INITIAL_CAPITAL = 1000.0  # USDT
MAX_OPEN_POSITIONS = 3
STOP_LOSS_PCT = 0.02  # %2 stop loss
TAKE_PROFIT_PCT = 0.04  # %4 take profit
SCAN_INTERVAL = 5  # saniye

# Risk Yönetimi
MAX_POSITION_SIZE_PCT = 33  # Her pozisyon max %33 sermaye
MIN_ORDER_SIZE = 10  # Minimum order büyüklüğü (USDT)

# =============================================================================
# 🤖 DEVISO BOT AYARLARI (TIMEFRAME ADAPTIVE) - YENİ!
# =============================================================================

# Deviso Live Test Ayarları
DEVISO_LIVE_TEST = {
    'demo_balance': 10000.0,
    'position_size_adaptive': {  # Timeframe'e göre adaptive
        'scalping': 0.03,    # %3 (1m, 5m)
        'swing': 0.05,       # %5 (15m, 1h)  
        'position': 0.08     # %8 (4h, 1d)
    },
    'max_trades': 3,  # En iyi 3 kuralı
    'min_signal_scores': {  # Timeframe'e göre adaptive AI eşikleri
        'scalping': 85,     # Scalping için daha düşük
        'swing': 90,        # Swing için standart
        'position': 95      # Position için yüksek
    },
    'timeframe_strategies': {
        'scalping': ['1m', '5m'],
        'swing': ['15m', '1h'],
        'position': ['1h', '4h']
    },
    'scan_intervals': {  # Timeframe'e göre tarama sıklığı
        'scalping': 30,     # 30 saniye (hızlı)
        'swing': 60,        # 60 saniye (normal)
        'position': 120     # 120 saniye (yavaş)
    }
}

# Deviso Live Trading Ayarları  
DEVISO_LIVE_TRADING = {
    'futures_balance': 1000.0,
    'risk_per_trade_adaptive': {  # Timeframe'e göre risk
        'scalping': 0.015,  # %1.5 (daha sık işlem)
        'swing': 0.02,      # %2.0 (standart)
        'position': 0.025   # %2.5 (daha az işlem)
    },
    'leverage_adaptive': {  # Timeframe'e göre kaldıraç
        'scalping': 15,     # Yüksek kaldıraç
        'swing': 10,        # Orta kaldıraç  
        'position': 5       # Düşük kaldıraç
    },
    'top3_selection_weights': {  # En iyi 3 seçim ağırlıkları
        'ai_score': 0.4,        # %40 AI skoru
        'timeframe_consistency': 0.3,  # %30 TF tutarlılığı
        'momentum': 0.2,        # %20 Momentum
        'volume': 0.1          # %10 Hacim
    }
}

# Timeframe Adaptive Sinyal Ayarları
DEVISO_TIMEFRAME_SIGNALS = {
    'analysis_combinations': {
        '1m': ['1m', '5m'],       # Scalping
        '5m': ['5m', '15m'],      # Kısa vade
        '15m': ['15m', '1h'],     # Orta vade  
        '1h': ['1h', '4h'],       # Uzun vade
        '4h': ['4h', '1d']        # Çok uzun vade
    },
    'timeframe_weights': {
        'scalping': [0.7, 0.3],     # Kısa TF ağırlık
        'swing': [0.5, 0.5],        # Dengeli ağırlık
        'position': [0.3, 0.7]      # Uzun TF ağırlık
    }
}

# Deviso Bot Control Flags - YENİ!
DEVISO_AVAILABLE = True  # Deviso sisteminin aktif olup olmadığı
deviso_manager = None    # Global Deviso manager instance

# =============================================================================
# 📊 TEKNİK ANALİZ PARAMETRELERİ
# =============================================================================

VOL_SMA_LEN = 20
DEFAULT_MIN_STREAK = 3
DEFAULT_MIN_PCT = 0.5
DEFAULT_MIN_VOLR = 1.5
DEFAULT_MIN_AI_SCORE = 0.3

# Deviso Gösterge Parametreleri
DEVISO_PARAMS = {
    'zigzag_high_period': 10,
    'zigzag_low_period': 10,
    'min_movement_pct': 0.160,
    'ma_period': 20,
    'std_mult': 2.0,
    'ma_length': 10
}

# AI Model Parametreleri
AI_PARAMS = {
    'model_type': 'random_forest',
    'retrain_interval': 50,
    'min_data_for_training': 20,
    'target_profit_threshold': 1.0,
}

# =============================================================================
# 📁 DOSYA YOLLARI
# =============================================================================

TRADES_CSV = 'ai_crypto_trades.csv'
CAPITAL_CSV = 'ai_crypto_capital.csv'
AI_MODEL_FILE = 'crypto_improved_ai_model.pkl'

# Deviso dosyaları - YENİ!
DEVISO_LIVE_TEST_DB = 'deviso_live_test.db'
DEVISO_LIVE_TRADING_DB = 'deviso_live_trading.db'

# =============================================================================
# 🌐 API AYARLARI
# =============================================================================

MAX_WORKERS = 8           
REQ_SLEEP = 0.05          
TIMEOUT = 10
AUTO_REFRESH_INTERVAL = 1

# Binance API Endpoints
if ENVIRONMENT == 'testnet':
    BASE = "https://testnet.binancefuture.com"
    WS_BASE = "wss://fstream.binancefuture.com"  # 🔧 DÜZELTME: Doğru testnet WebSocket URL
else:
    BASE = "https://fapi.binance.com"
    WS_BASE = "wss://fstream.binance.com"

EXCHANGE_INFO = f"{BASE}/fapi/v1/exchangeInfo"
KLINES = f"{BASE}/fapi/v1/klines"

# =============================================================================
# 📝 LOGGING AYARLARI
# =============================================================================

def setup_logging():
    """Logging sistemini yapılandır"""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)s %(message)s'
    )
    
    # Dash logger'ı sustur
    dash_logger = logging.getLogger('werkzeug')
    dash_logger.setLevel(logging.WARNING)
    
    return logging.getLogger("crypto-analytics")

# =============================================================================
# 🌐 HTTP SESSION AYARLARI
# =============================================================================

def create_session():
    """Optimize edilmiş HTTP session oluştur"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "crypto-analytics/1.0"
    })

    retry = Retry(
        total=3,
        backoff_factor=0.3,  
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    
    adapter = HTTPAdapter(
        pool_connections=100, 
        pool_maxsize=100, 
        max_retries=retry
    )
    
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session

# =============================================================================
# 🎨 UI AYARLARI
# =============================================================================

# Dash uygulama ayarları
DASH_CONFIG = {
    'debug': False,
    'host': "127.0.0.1",
    'port': 8050,
    'title': "🤖 AI Crypto Analytics + Live Trading Bot + Deviso"
}

# Tablo güncelleme aralığı (ms)
TABLE_REFRESH_INTERVAL = 2000

# =============================================================================
# 🔧 GLOBAL DEĞİŞKENLER (Diğer modüller tarafından import edilecek)
# =============================================================================

# Bu değişkenler runtime'da güncellenir
current_capital = INITIAL_CAPITAL
open_positions = {}
trading_active = False
auto_scan_active = False

# Global veri depolama
current_data = None
current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_streak': DEFAULT_MIN_STREAK,
    'min_pct': DEFAULT_MIN_PCT,
    'min_volr': DEFAULT_MIN_VOLR,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}

saved_signals = {}

# =============================================================================
# 🆕 DEVISO GLOBAL DEĞİŞKENLER - YENİ!
# =============================================================================

# Deviso bot durumları
deviso_live_test_active = False
deviso_live_trading_active = False

# Deviso pozisyonları (ana open_positions'dan ayrı)
deviso_live_test_positions = {}
deviso_live_trading_positions = {}

# Deviso ayarları
deviso_current_timeframe = DEFAULT_TIMEFRAME
deviso_current_strategy = 'swing'  # scalping, swing, position

# =============================================================================
# 🆕 DEVISO YARDIMCI FONKSİYONLAR - YENİ!
# =============================================================================

def get_strategy_from_timeframe(timeframe: str) -> str:
    """Timeframe'den strateji belirle"""
    if timeframe in ['1m', '5m']:
        return 'scalping'
    elif timeframe in ['15m', '1h']:
        return 'swing'
    else:
        return 'position'

def get_adaptive_ai_threshold(strategy: str) -> float:
    """Strateji'ye göre adaptive AI eşiği"""
    return DEVISO_LIVE_TEST['min_signal_scores'].get(strategy, 90)

def get_scan_interval_for_strategy(strategy: str) -> int:
    """Strateji'ye göre tarama aralığı"""
    return DEVISO_LIVE_TEST['scan_intervals'].get(strategy, 60)

# =============================================================================
# 📦 BAŞLATMA FONKSİYONU - GÜNCELLENDİ!
# =============================================================================

def initialize():
    """Sistemin temel bileşenlerini başlat"""
    global deviso_manager
    
    logger = setup_logging()
    session = create_session()
    
    logger.info("🚀 Kripto AI sistemi yapılandırıldı")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"💰 Başlangıç sermayesi: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    
    # Deviso durumu
    if DEVISO_AVAILABLE:
        logger.info("🤖 Deviso bot sistemi AKTİF")
        logger.info(f"🎯 Varsayılan timeframe: {DEFAULT_TIMEFRAME}")
        logger.info(f"🔄 Varsayılan strateji: {get_strategy_from_timeframe(DEFAULT_TIMEFRAME)}")
        
        # Deviso manager'ı başlat (import burada yapılacak circular import'u önlemek için)
        try:
            from trading.deviso_manager import DevsoManager
            deviso_manager = DevsoManager()
            logger.info("✅ Deviso Manager başlatıldı")
        except ImportError as e:
            logger.warning(f"⚠️ Deviso Manager import edilemedi: {e}")
            deviso_manager = None
    else:
        logger.info("⚠️ Deviso bot sistemi KAPALI")
    
    # API key kontrolü
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        logger.info("✅ Binance API anahtarları yüklendi")
    else:
        logger.warning("⚠️ Binance API anahtarları bulunamadı (.env dosyasını kontrol edin)")
    
    return logger, session