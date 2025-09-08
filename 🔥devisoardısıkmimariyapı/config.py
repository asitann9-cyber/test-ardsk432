"""
🔧 Kripto AI Sistemi - Konfigürasyon Dosyası
Tüm ayarlar ve sabitler burada tanımlanır
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
    'title': "🤖 AI Crypto Analytics + Live Trading Bot"
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
# 📦 BAŞLATMA FONKSİYONU
# =============================================================================

def initialize():
    """Sistemin temel bileşenlerini başlat"""
    logger = setup_logging()
    session = create_session()
    
    logger.info("🚀 Kripto AI sistemi yapılandırıldı")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"💰 Başlangıç sermayesi: ${INITIAL_CAPITAL:.2f}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    
    # API key kontrolü
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        logger.info("✅ Binance API anahtarları yüklendi")
    else:
        logger.warning("⚠️ Binance API anahtarları bulunamadı (.env dosyasını kontrol edin)")
    
    return logger, session