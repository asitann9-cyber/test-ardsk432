"""
🔧 Kripto AI Sistemi - Konfigürasyon Dosyası - Ultra Panel v5
🔥 Heikin Ashi Multi-Timeframe analizi
🔥 Ultra Signal (3/4 HTF crossover) + Candle Power
🔥 Whale Detection + Memory System
🔥 VPMV SİSTEMİ KALDIRILDI
🔥 BOT: Testnet (Sabit) | VERİ: Mainnet/Testnet (Seçilebilir)
"""

import os
import pytz
import logging
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# .ENV DOSYASI YÜKLE
load_dotenv()

# =============================================================================
# TEMEL AYARLAR
# =============================================================================

LOCAL_TZ = pytz.timezone("Europe/Istanbul")
DEFAULT_TIMEFRAME = "15m"
LIMIT = 500 
SYMBOL_LIMIT = None  

# =============================================================================
# ENVIRONMENT AYARLARI - İKİ AYRI SİSTEM
# =============================================================================

# 🤖 BOT için SABİT TESTNET (Gerçek para riski yok)
BOT_ENVIRONMENT = 'testnet'
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# 📊 VERİ ÇEKME için SEÇİLEBİLİR (Mainnet veya Testnet)
# .env'den okunuyor, varsayılan: mainnet
DATA_ENVIRONMENT = os.getenv('DATA_ENVIRONMENT', 'mainnet')  # mainnet veya testnet

# Geriye uyumluluk için (live_trader.py için)
ENVIRONMENT = BOT_ENVIRONMENT

# =============================================================================
# LIVE TRADING AYARLARI (TESTNET)
# =============================================================================

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
# ULTRA PANEL v5 PARAMETRELERİ
# =============================================================================

# HTF (Higher Timeframe) Çarpanları
# Base timeframe'in kaç katı olacak (örn: 15m base ise 8x = 2H)
HTF_MULTIPLIERS = {
    'htf8': 8,      # ≈ 2H  (15m × 8)
    'htf12': 12,    # ≈ 3H  (15m × 12)
    'htf16': 16,    # ≈ 4H  (15m × 16)
    'htf24': 24     # ≈ 6H  (15m × 24)
}

# Ultra Signal Parametreleri
ULTRA_SIGNAL_PARAMS = {
    'min_htf_count': 3,              # Minimum HTF crossover sayısı (3/4 veya 4/4)
    'min_candle_change': 3.0,        # Minimum candle değişim % (güçlü mum için)
    'use_volume_in_power': True,     # Candle power hesabında volume kullan
    'power_multiplier_4_4': 2.0,     # 4/4 Ultra için power çarpanı
    'power_multiplier_3_4': 1.5      # 3/4 Ultra için power çarpanı
}

# Whale Volume Detection
WHALE_PARAMS = {
    'volume_spike_multiplier': 2.5,  # Daily volume MA'nın kaç katı (2.5x)
    'min_volume_ma_period': 50       # Minimum volume MA periyodu
}

# Filtreleme Parametreleri
DEFAULT_MIN_POWER = 5.0              # Minimum total power
DEFAULT_MIN_HTF_COUNT = 3            # Minimum HTF count (3/4)
DEFAULT_MIN_AI_SCORE = 0.3           # Minimum AI skoru (0-1)

# Memory Sistemi Parametreleri
MEMORY_PARAMS = {
    'max_age_minutes': 15,           # Maksimum sinyal yaşı (dakika)
    'base_penalty_weak': 20,         # Zayıf sinyal için baz ceza
    'base_penalty_medium': 10,       # Orta sinyal için baz ceza
    'base_penalty_strong': 5,        # Güçlü sinyal için baz ceza
    'age_penalty_threshold_1': 3,    # İlk yaş eşiği (dakika)
    'age_penalty_threshold_2': 7,    # İkinci yaş eşiği (dakika)
    'strong_power_threshold': 20.0,  # Güçlü sinyal power eşiği
    'medium_power_threshold': 10.0   # Orta sinyal power eşiği
}

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
AI_MODEL_FILE = 'crypto_ultra_ai_model.pkl'

# =============================================================================
# API AYARLARI - İKİ AYRI URL SETİ
# =============================================================================

MAX_WORKERS = 8           
REQ_SLEEP = 0.05          
TIMEOUT = 10
AUTO_REFRESH_INTERVAL = 1

# 🤖 BOT için TESTNET URL'leri (Sabit)
BOT_BASE = "https://testnet.binancefuture.com"
BOT_WS_BASE = "wss://fstream.binancefuture.com"

# 📊 VERİ ÇEKME için URL'ler (Seçilebilir) - GLOBAL DEĞİŞKEN
DATA_BASE = ""
DATA_WS_BASE = ""

# İlk başlatma
if DATA_ENVIRONMENT == 'testnet':
    DATA_BASE = "https://testnet.binancefuture.com"
    DATA_WS_BASE = "wss://fstream.binancefuture.com"
else:  # mainnet
    DATA_BASE = "https://fapi.binance.com"
    DATA_WS_BASE = "wss://fstream.binance.com"

# Geriye uyumluluk için (live_trader.py için)
BASE = BOT_BASE
WS_BASE = BOT_WS_BASE

# Veri çekme endpoint'leri (DATA_BASE kullanır) - GLOBAL DEĞİŞKEN
EXCHANGE_INFO = ""
KLINES = ""
TICKER_PRICE = ""

# İlk başlatma
EXCHANGE_INFO = f"{DATA_BASE}/fapi/v1/exchangeInfo"
KLINES = f"{DATA_BASE}/fapi/v1/klines"
TICKER_PRICE = f"{DATA_BASE}/fapi/v1/ticker/price"

# =============================================================================
# LOGGING AYARLARI
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
# HTTP SESSION AYARLARI
# =============================================================================

def create_session():
    """Optimize edilmiş HTTP session oluştur"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "crypto-analytics/3.0-ultra-panel"
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
    'title': "🤖 AI Crypto Analytics - Ultra Panel v5 Multi-HTF + Live Trading"
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
    'min_power': DEFAULT_MIN_POWER,
    'min_htf_count': DEFAULT_MIN_HTF_COUNT,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}
saved_signals = {}

# =============================================================================
# VERİ KAYNAĞINI DEĞİŞTİRME FONKSİYONU
# =============================================================================

def switch_data_source(source: str):
    """
    🔥 DÜZELTME: Veri kaynağını değiştir (mainnet/testnet)
    Global değişkenleri günceller
    
    Args:
        source: 'mainnet' veya 'testnet'
        
    Returns:
        bool: Başarılı mı?
    """
    global DATA_ENVIRONMENT, DATA_BASE, DATA_WS_BASE
    global EXCHANGE_INFO, KLINES, TICKER_PRICE
    
    logger = logging.getLogger("crypto-analytics")
    
    if source not in ['mainnet', 'testnet']:
        logger.error(f"❌ Geçersiz veri kaynağı: {source}")
        return False
    
    # Değişiklik yoksa geç
    if DATA_ENVIRONMENT == source:
        logger.debug(f"ℹ️ Veri kaynağı zaten {source}")
        return True
    
    DATA_ENVIRONMENT = source
    
    if source == 'testnet':
        DATA_BASE = "https://testnet.binancefuture.com"
        DATA_WS_BASE = "wss://fstream.binancefuture.com"
        logger.info("=" * 70)
        logger.info("🧪 VERİ KAYNAĞI DEĞİŞTİRİLDİ: TESTNET")
        logger.info("=" * 70)
    else:  # mainnet
        DATA_BASE = "https://fapi.binance.com"
        DATA_WS_BASE = "wss://fstream.binance.com"
        logger.info("=" * 70)
        logger.info("🚀 VERİ KAYNAĞI DEĞİŞTİRİLDİ: MAINNET")
        logger.info("=" * 70)
    
    # Endpoint'leri güncelle
    EXCHANGE_INFO = f"{DATA_BASE}/fapi/v1/exchangeInfo"
    KLINES = f"{DATA_BASE}/fapi/v1/klines"
    TICKER_PRICE = f"{DATA_BASE}/fapi/v1/ticker/price"
    
    logger.info(f"📡 Veri URL güncellendi: {DATA_BASE}")
    logger.info(f"🔗 Exchange Info: {EXCHANGE_INFO}")
    logger.info("=" * 70)
    
    return True


def get_current_data_source() -> dict:
    """
    🔥 YENİ: Mevcut veri kaynağı bilgilerini al
    
    Returns:
        dict: Veri kaynağı bilgileri
    """
    return {
        'environment': DATA_ENVIRONMENT,
        'base_url': DATA_BASE,
        'is_mainnet': DATA_ENVIRONMENT == 'mainnet',
        'is_testnet': DATA_ENVIRONMENT == 'testnet',
        'display_name': '🚀 Binance Mainnet' if DATA_ENVIRONMENT == 'mainnet' else '🧪 Binance Testnet'
    }

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
# ULTRA PANEL YARDIMCI FONKSİYONLAR
# =============================================================================

def get_ultra_config() -> dict:
    """Ultra Panel v5 konfigürasyonunu döndür"""
    return {
        'htf_multipliers': HTF_MULTIPLIERS,
        'ultra_signal': ULTRA_SIGNAL_PARAMS,
        'whale': WHALE_PARAMS,
        'memory': MEMORY_PARAMS,
        'filters': {
            'min_power': DEFAULT_MIN_POWER,
            'min_htf_count': DEFAULT_MIN_HTF_COUNT,
            'min_ai_score': DEFAULT_MIN_AI_SCORE
        }
    }

def validate_ultra_signal(total_power: float, htf_count: int, ai_score: float) -> bool:
    """
    Ultra Panel sinyalinin geçerli olup olmadığını kontrol et
    
    Args:
        total_power: Total candle power
        htf_count: HTF crossover sayısı (3 veya 4)
        ai_score: AI skoru (0-100)
        
    Returns:
        bool: Sinyal geçerli mi?
    """
    return (
        total_power >= DEFAULT_MIN_POWER and
        htf_count >= DEFAULT_MIN_HTF_COUNT and
        ai_score >= (DEFAULT_MIN_AI_SCORE * 100)
    )

# =============================================================================
# GERİYE UYUMLULUK (ESKİ VPMV FONKSİYONLARI)
# =============================================================================

def get_vpmv_config() -> dict:
    """
    🔄 GERİYE UYUMLULUK: get_ultra_config()'e yönlendir
    DEPRECATED: Yeni kodda get_ultra_config() kullan
    """
    logger = logging.getLogger("crypto-analytics")
    logger.warning("⚠️ get_vpmv_config() deprecated - get_ultra_config() kullan")
    return get_ultra_config()

def validate_vpmv_signal(vpmv_score: float, ai_score: float) -> bool:
    """
    🔄 GERİYE UYUMLULUK: validate_ultra_signal()'e yönlendir
    DEPRECATED: Yeni kodda validate_ultra_signal() kullan
    """
    logger = logging.getLogger("crypto-analytics")
    logger.warning("⚠️ validate_vpmv_signal() deprecated - validate_ultra_signal() kullan")
    # VPMV score'u power'a dönüştür (yaklaşık)
    return validate_ultra_signal(
        total_power=abs(vpmv_score) / 2.0,  # Yaklaşık dönüşüm
        htf_count=3,  # Varsayılan
        ai_score=ai_score
    )

# =============================================================================
# BAŞLATMA FONKSİYONU
# =============================================================================

def initialize():
    """Sistemin temel bileşenlerini başlat"""
    logger = setup_logging()
    session = create_session()
    
    logger.info("🚀 Kripto AI Sistemi - Ultra Panel v5 Multi-HTF")
    logger.info("🔥 Heikin Ashi Multi-Timeframe Analizi")
    logger.info("🔥 Ultra Signal (3/4 HTF crossover) + Candle Power")
    logger.info("🔥 Whale Detection + Memory System")
    logger.info("=" * 70)
    logger.info(f"🤖 Bot Environment: {BOT_ENVIRONMENT.upper()} (TESTNET - Sabit)")
    logger.info(f"📊 Veri Environment: {DATA_ENVIRONMENT.upper()} (Seçilebilir)")
    logger.info("=" * 70)
    logger.info(f"🤖 Bot URL: {BOT_BASE}")
    logger.info(f"📡 Veri URL: {DATA_BASE}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info("❌ ESKİ SİSTEM KALDIRILDI: VPMV, Deviso, Gauss, Z-Score")
    logger.info("🔥 YENİ SİSTEM: Ultra Panel v5 Multi-HTF")
    
    # Ultra Panel config özeti
    ultra_cfg = get_ultra_config()
    logger.info(f"📈 HTF Çarpanları: {ultra_cfg['htf_multipliers']}")
    logger.info(f"🎯 Ultra Signal: Min HTF={ultra_cfg['ultra_signal']['min_htf_count']}/4, Min Change={ultra_cfg['ultra_signal']['min_candle_change']}%")
    logger.info(f"🐋 Whale Detection: Spike={ultra_cfg['whale']['volume_spike_multiplier']}x Volume MA")
    logger.info(f"🧠 Memory: Max Age={ultra_cfg['memory']['max_age_minutes']} dakika")
    logger.info(f"🔍 Filtreler: Power>={ultra_cfg['filters']['min_power']}, HTF>={ultra_cfg['filters']['min_htf_count']}/4, AI>={ultra_cfg['filters']['min_ai_score']*100}%")
    
    # API key kontrolü
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        logger.info("✅ Binance API anahtarları yüklendi (Testnet Bot için)")
    else:
        logger.warning("⚠️ Binance API anahtarları bulunamadı (.env dosyasını kontrol edin)")
    
    return logger, session
