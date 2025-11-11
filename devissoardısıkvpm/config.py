"""
🔧 Kripto AI Sistemi - Konfigürasyon Dosyası - VPMV Sistemi
 YENİ: VPMV (Volume-Price-Momentum-Volatility) parametreleri
 ESKİLER KALDIRILDI: Deviso, Z-Score, Gauss, Log Volume
 SADECE LIVE TRADING
"""

import os
import pytz
import logging
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#.ENV DOSYASI YÜKLE
load_dotenv()

# =============================================================================
# TEMEL AYARLAR
# =============================================================================

LOCAL_TZ = pytz.timezone("Europe/Istanbul")
DEFAULT_TIMEFRAME = "15m"
LIMIT = 500 
SYMBOL_LIMIT = None  

# =============================================================================
# LIVE TRADING AYARLARI
# =============================================================================

# Binance API Keys (.env dosyasından)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'testnet')  # testnet veya mainnet

# Trading Parametreleri
LIVE_TRADING_ACTIVE = False
INITIAL_CAPITAL = 1000.0  # USDT (sadece referans için)
MAX_OPEN_POSITIONS = 1
STOP_LOSS_PCT = 0.01  
TAKE_PROFIT_PCT = 0.02 
SCAN_INTERVAL = 1  # saniye

# Risk Yönetimi
MAX_POSITION_SIZE_PCT = 100  
MIN_ORDER_SIZE = 10  # Minimum order büyüklüğü (USDT)

# =============================================================================
# VPMV PARAMETRELERİ
# =============================================================================

# SuperTrend Parametreleri
SUPERTREND_PARAMS = {
    'atr_period': 10,      # ATR periyodu
    'multiplier': 3.0      # ATR çarpanı
}

# VPMV Bileşen Ağırlıkları
VPMV_WEIGHTS = {
    'price': 0.7,          # %70 - En yüksek ağırlık
    'volume': 0.1,         # %10
    'momentum': 0.1,       # %10
    'volatility': 0.1      # %10
}

# Tetikleyici Eşikleri
TRIGGER_THRESHOLDS = {
    'price': 20,           # Price component >= 20
    'momentum': 10,        # Momentum component >= 10
    'volume': 15,          # Volume component >= 15
    'volatility': 8        # Volatility component >= 8
}

# TIME Alignment Parametreleri
TIME_ALIGNMENT = {
    'timeframes': ['1h', '2h', '4h', '1d', '1w'],  # İzlenecek timeframe'ler
    'min_match': 3         # Minimum uyum sayısı (sinyal için)
}

# Filtreleme Parametreleri
DEFAULT_MIN_VPMV_SCORE = 10.0        # Minimum VPMV skoru
DEFAULT_MIN_AI_SCORE = 0.3           # Minimum AI skoru (0-1)
DEFAULT_MIN_TIME_MATCH = 2           # Minimum TIME uyumu

# AI Model Parametreleri
AI_PARAMS = {
    'model_type': 'random_forest_regressor',
    'retrain_interval': 50,
    'min_data_for_training': 20,
    'target_profit_threshold': 1.0,
    'ml_weight': 0.6,      # ML model ağırlığı
    'manual_weight': 0.4   # Manuel skor ağırlığı
}

# =============================================================================
# DOSYA YOLLARI
# =============================================================================

TRADES_CSV = 'ai_crypto_trades.csv'
CAPITAL_CSV = 'ai_crypto_capital.csv'
AI_MODEL_FILE = 'crypto_vpmv_ai_model.pkl'

# =============================================================================
# API AYARLARI
# =============================================================================

MAX_WORKERS = 8           
REQ_SLEEP = 0.05          
TIMEOUT = 10
AUTO_REFRESH_INTERVAL = 1

# Binance API Endpoints
if ENVIRONMENT == 'testnet':
    BASE = "https://testnet.binancefuture.com"
    WS_BASE = "wss://fstream.binancefuture.com"
else:
    BASE = "https://fapi.binance.com"
    WS_BASE = "wss://fstream.binance.com"

EXCHANGE_INFO = f"{BASE}/fapi/v1/exchangeInfo"
KLINES = f"{BASE}/fapi/v1/klines"

# =============================================================================
#  LOGGING AYARLARI
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
#  HTTP SESSION AYARLARI
# =============================================================================

def create_session():
    """Optimize edilmiş HTTP session oluştur"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "crypto-analytics/2.0-vpmv"
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
# UI AYARLARI
# =============================================================================

# Dash uygulama ayarları
DASH_CONFIG = {
    'debug': False,
    'host': "127.0.0.1",
    'port': 8050,
    'title': "🤖 AI Crypto Analytics - VPMV System + Live Trading"
}

# Tablo güncelleme aralığı (ms)
TABLE_REFRESH_INTERVAL = 1000

# =============================================================================
# TRADING DEĞİŞKENLERİ - SADECE LIVE TRADING
# =============================================================================

# Live Trading Değişkenleri  
live_capital = 0.0
live_positions = {}
live_trading_active = False

# ⚠️ GEÇICI: App.py uyumluluğu için - KULLANILMAYACAK
# Bu attributelar sadece app.py'nin hata vermemesi için
paper_capital = 0.0  # KULLANILMAZ - sadece compatibility
paper_positions = {}  # KULLANILMAZ - sadece compatibility

# Genel sistem değişkenleri
auto_scan_active = False
current_data = None
current_settings = {
    'timeframe': DEFAULT_TIMEFRAME,
    'min_vpmv_score': DEFAULT_MIN_VPMV_SCORE,
    'min_time_match': DEFAULT_MIN_TIME_MATCH,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}
saved_signals = {}

# =============================================================================
# LIVE TRADING KONTROL FONKSİYONLARI
# =============================================================================

def switch_to_live_mode():
    """Live trading moduna geç"""
    global live_trading_active
    live_trading_active = True
    logging.getLogger("crypto-analytics").info("🤖 Live Trading moduna geçildi")

def is_live_mode():
    """Live trading modunda mı? - ARTIK HER ZAMAN TRUE"""
    return True  # Sadece live trading olduğu için her zaman True

def update_live_capital(new_balance: float):
    """Live trading bakiyesini güncelle"""
    global live_capital
    live_capital = new_balance
    logging.getLogger("crypto-analytics").info(f"💰 Live capital güncellendi: ${new_balance:.2f}")

def update_live_positions(new_positions: dict):
    """Live trading pozisyonlarını güncelle"""
    global live_positions
    live_positions = new_positions
    logging.getLogger("crypto-analytics").debug(f"📊 Live positions güncellendi: {len(new_positions)} pozisyon")

def get_live_trading_summary():
    """Live Trading özetini döndür"""
    return {
        'capital': live_capital,
        'positions': len(live_positions),
        'active': live_trading_active,
        'symbols': list(live_positions.keys())
    }

def reset_live_trading():
    """Live trading verilerini sıfırla"""
    global live_trading_active
    live_trading_active = False
    logging.getLogger("crypto-analytics").info("🔄 Live trading durduruldu")

# ⚠️ GEÇICI COMPATIBILITY FONKSIYONLARI - App.py için
def switch_to_paper_mode():
    """Paper mode'a geç - KULLANILMAZ artık"""
    logging.getLogger("crypto-analytics").warning("⚠️ Paper mode çağrısı - artık sadece Live Trading var!")
    pass

def reset_paper_trading():
    """Paper trading sıfırla - KULLANILMAZ artık"""
    logging.getLogger("crypto-analytics").warning("⚠️ Paper reset çağrısı - artık sadece Live Trading var!")
    pass

# =============================================================================
#  VPMV YARDIMCI FONKSİYONLAR
# =============================================================================

def get_vpmv_config() -> dict:
    """VPMV konfigürasyonunu döndür"""
    return {
        'supertrend': SUPERTREND_PARAMS,
        'weights': VPMV_WEIGHTS,
        'triggers': TRIGGER_THRESHOLDS,
        'time_alignment': TIME_ALIGNMENT,
        'filters': {
            'min_vpmv_score': DEFAULT_MIN_VPMV_SCORE,
            'min_ai_score': DEFAULT_MIN_AI_SCORE,
            'min_time_match': DEFAULT_MIN_TIME_MATCH
        }
    }

def validate_vpmv_signal(vpmv_score: float, time_match: int, ai_score: float) -> bool:
    """
    VPMV sinyalinin geçerli olup olmadığını kontrol et
    
    Args:
        vpmv_score: VPMV skoru
        time_match: TIME uyum sayısı
        ai_score: AI skoru (0-100)
        
    Returns:
        bool: Sinyal geçerli mi?
    """
    return (
        abs(vpmv_score) >= DEFAULT_MIN_VPMV_SCORE and
        time_match >= DEFAULT_MIN_TIME_MATCH and
        ai_score >= (DEFAULT_MIN_AI_SCORE * 100)
    )

# =============================================================================
# BAŞLATMA FONKSİYONU
# =============================================================================

def initialize():
    """Sistemin temel bileşenlerini başlat"""
    logger = setup_logging()
    session = create_session()
    
    logger.info("🚀 Kripto AI Sistemi - VPMV (Volume-Price-Momentum-Volatility)")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info("🎯 ESKİ SİSTEM KALDIRILDI: Deviso, Gauss, Z-Score")
    logger.info("🔥 YENİ SİSTEM: VPMV + SuperTrend + TIME Alignment")
    
    # VPMV config özeti
    vpmv_cfg = get_vpmv_config()
    logger.info(f"📈 SuperTrend: ATR={vpmv_cfg['supertrend']['atr_period']}, Mult={vpmv_cfg['supertrend']['multiplier']}")
    logger.info(f"⚖️ VPMV Ağırlıklar: P={vpmv_cfg['weights']['price']*100}%, V={vpmv_cfg['weights']['volume']*100}%, M={vpmv_cfg['weights']['momentum']*100}%, V={vpmv_cfg['weights']['volatility']*100}%")
    logger.info(f"🎯 Tetikleyici Eşikler: P>={vpmv_cfg['triggers']['price']}, M>={vpmv_cfg['triggers']['momentum']}, V>={vpmv_cfg['triggers']['volume']}, Vol>={vpmv_cfg['triggers']['volatility']}")
    logger.info(f"⏰ TIME Alignment: {len(vpmv_cfg['time_alignment']['timeframes'])} timeframe, min_match={vpmv_cfg['time_alignment']['min_match']}")
    
    # API key kontrolü
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        logger.info("✅ Binance API anahtarları yüklendi")
    else:
        logger.warning("⚠️ Binance API anahtarları bulunamadı (.env dosyasını kontrol edin)")
    
    return logger, session