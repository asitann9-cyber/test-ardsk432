"""
🔧 Kripto AI Sistemi - Konfigürasyon Dosyası - GÜNCELLEME
🔥 YENİ: Position validation, real-time sync, debug fonksiyonları eklendi
🔧 DÜZELTME: Test sorunları için kritik fonksiyonlar
"""

import os
import pytz
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
INITIAL_CAPITAL = 1000.0  # USDT (sadece referans için)
MAX_OPEN_POSITIONS = 1
STOP_LOSS_PCT = 0.01  
TAKE_PROFIT_PCT = 0.02 
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
    WS_BASE = "wss://fstream.binancefuture.com"
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
# 🔧 TRADING DEĞİŞKENLERİ - SADECE LIVE TRADING
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
    'min_streak': DEFAULT_MIN_STREAK,
    'min_pct': DEFAULT_MIN_PCT,
    'min_volr': DEFAULT_MIN_VOLR,
    'min_ai': DEFAULT_MIN_AI_SCORE * 100
}
saved_signals = {}

# 🔧 YENİ: Position tracking değişkenleri
_last_position_sync = None
_position_validation_errors = []

# =============================================================================
# 🔧 LIVE TRADING KONTROL FONKSİYONLARI - GÜNCELLENMİŞ
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
    """🔧 GÜNCELLEME: Live trading bakiyesini güncelle - validation eklendi"""
    global live_capital
    
    # Validation
    if not isinstance(new_balance, (int, float)) or new_balance < 0:
        logging.getLogger("crypto-analytics").warning(f"⚠️ Geçersiz capital değeri: {new_balance}")
        return False
    
    old_capital = live_capital
    live_capital = float(new_balance)
    
    # Değişim logu
    if old_capital != live_capital:
        change = live_capital - old_capital
        logging.getLogger("crypto-analytics").info(f"💰 Live capital güncellendi: ${old_capital:.2f} → ${live_capital:.2f} ({change:+.2f})")
    
    return True

def update_live_positions(new_positions: dict):
    """🔧 GÜNCELLEME: Live trading pozisyonlarını güncelle - validation eklendi"""
    global live_positions, _last_position_sync
    
    try:
        # Input validation
        if not isinstance(new_positions, dict):
            logging.getLogger("crypto-analytics").error(f"❌ Geçersiz positions tipi: {type(new_positions)}")
            return False
        
        # Position validation
        validation_errors = validate_positions_data(new_positions)
        if validation_errors:
            logging.getLogger("crypto-analytics").warning(f"⚠️ Position validation hataları: {len(validation_errors)}")
            for error in validation_errors[:3]:  # İlk 3 hatayı göster
                logging.getLogger("crypto-analytics").debug(f"   • {error}")
        
        # Değişiklik analizi
        old_symbols = set(live_positions.keys())
        new_symbols = set(new_positions.keys())
        
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        
        # Güncelleme
        live_positions = new_positions.copy()
        _last_position_sync = datetime.now(LOCAL_TZ)
        
        # Change log
        if added or removed:
            logging.getLogger("crypto-analytics").info(f"📊 Live positions güncellendi: {len(new_positions)} pozisyon")
            if added:
                logging.getLogger("crypto-analytics").info(f"   ➕ Eklenen: {', '.join(added)}")
            if removed:
                logging.getLogger("crypto-analytics").info(f"   ➖ Kaldırılan: {', '.join(removed)}")
        else:
            logging.getLogger("crypto-analytics").debug(f"📊 Live positions senkronize: {len(new_positions)} pozisyon")
        
        return True
        
    except Exception as e:
        logging.getLogger("crypto-analytics").error(f"❌ Position güncelleme hatası: {e}")
        return False

def validate_positions_data(positions: dict) -> List[str]:
    """
    🔧 YENİ: Position verilerini validate et
    
    Args:
        positions (dict): Position verileri
        
    Returns:
        List[str]: Validation hataları
    """
    errors = []
    
    for symbol, position in positions.items():
        try:
            # Temel field kontrolü
            required_fields = ['symbol', 'side', 'quantity', 'entry_price']
            for field in required_fields:
                if field not in position:
                    errors.append(f"{symbol}: Eksik field '{field}'")
                elif position[field] is None:
                    errors.append(f"{symbol}: Field '{field}' None değeri")
            
            # Veri tipi kontrolü
            if 'quantity' in position:
                try:
                    qty = float(position['quantity'])
                    if qty <= 0:
                        errors.append(f"{symbol}: Geçersiz quantity {qty}")
                except (ValueError, TypeError):
                    errors.append(f"{symbol}: Quantity parse hatası")
            
            if 'entry_price' in position:
                try:
                    price = float(position['entry_price'])
                    if price <= 0:
                        errors.append(f"{symbol}: Geçersiz entry_price {price}")
                except (ValueError, TypeError):
                    errors.append(f"{symbol}: Entry_price parse hatası")
            
            # Side kontrolü
            if 'side' in position:
                if position['side'] not in ['LONG', 'SHORT']:
                    errors.append(f"{symbol}: Geçersiz side '{position['side']}'")
                    
        except Exception as e:
            errors.append(f"{symbol}: Validation hatası - {e}")
    
    return errors

def get_live_trading_summary():
    """🔧 GÜNCELLEME: Live Trading özetini döndür - detaylı bilgi eklendi"""
    global _last_position_sync
    
    summary = {
        'capital': live_capital,
        'positions': len(live_positions),
        'active': live_trading_active,
        'symbols': list(live_positions.keys()),
        'last_sync': _last_position_sync.strftime('%Y-%m-%d %H:%M:%S') if _last_position_sync else 'Never',
        'validation_errors': len(_position_validation_errors)
    }
    
    # Position details
    if live_positions:
        total_invested = sum(pos.get('invested_amount', 0) for pos in live_positions.values())
        summary['total_invested'] = total_invested
        
        # Side analizi
        long_count = sum(1 for pos in live_positions.values() if pos.get('side') == 'LONG')
        short_count = len(live_positions) - long_count
        summary['long_positions'] = long_count
        summary['short_positions'] = short_count
    
    return summary

def reset_live_trading():
    """🔧 GÜNCELLEME: Live trading verilerini sıfırla - tam temizlik"""
    global live_trading_active, live_positions, _last_position_sync, _position_validation_errors
    
    live_trading_active = False
    live_positions.clear()
    _last_position_sync = None
    _position_validation_errors.clear()
    
    logging.getLogger("crypto-analytics").info("🔄 Live trading tamamen sıfırlandı")

def force_position_sync() -> bool:
    """
    🔧 YENİ: Pozisyon senkronizasyonunu zorla - test sorunları için kritik
    
    Returns:
        bool: Senkronizasyon başarılı mı
    """
    try:
        global _last_position_sync
        
        logging.getLogger("crypto-analytics").info("🔄 Pozisyon senkronizasyonu zorlanıyor...")
        
        # Binance'den gerçek pozisyonları al (eğer varsa)
        try:
            # Bu fonksiyon live_trader.py'den çağrılacak
            from trading.live_trader import get_current_live_positions
            
            real_positions = get_current_live_positions()
            
            if real_positions != live_positions:
                logging.getLogger("crypto-analytics").info(f"🔄 Binance pozisyonları ile senkronize ediliyor: {len(real_positions)} pozisyon")
                update_live_positions(real_positions)
                return True
            else:
                logging.getLogger("crypto-analytics").debug("✅ Pozisyonlar zaten senkron")
                return True
                
        except ImportError:
            logging.getLogger("crypto-analytics").debug("⚠️ Live trader modülü yok - manuel sync")
            _last_position_sync = datetime.now(LOCAL_TZ)
            return True
            
    except Exception as e:
        logging.getLogger("crypto-analytics").error(f"❌ Force sync hatası: {e}")
        return False

def debug_position_state() -> Dict:
    """
    🔧 YENİ: Position durumunu debug et - test sorunları için
    
    Returns:
        Dict: Debug bilgileri
    """
    try:
        debug_info = {
            'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'live_capital': live_capital,
            'live_trading_active': live_trading_active,
            'positions_count': len(live_positions),
            'position_symbols': list(live_positions.keys()),
            'last_sync': _last_position_sync.strftime('%Y-%m-%d %H:%M:%S') if _last_position_sync else 'Never',
            'validation_errors_count': len(_position_validation_errors),
            'position_details': {}
        }
        
        # Her pozisyon için detaylı bilgi
        for symbol, position in live_positions.items():
            debug_info['position_details'][symbol] = {
                'side': position.get('side', 'UNKNOWN'),
                'quantity': position.get('quantity', 0),
                'entry_price': position.get('entry_price', 0),
                'auto_sltp': position.get('auto_sltp', False),
                'has_signal_data': 'signal_data' in position,
                'fields_count': len(position)
            }
        
        # Validation errors
        if _position_validation_errors:
            debug_info['recent_validation_errors'] = _position_validation_errors[-5:]  # Son 5 hata
        
        return debug_info
        
    except Exception as e:
        return {
            'timestamp': datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'live_capital': live_capital,
            'positions_count': len(live_positions) if live_positions else 0
        }

def validate_live_positions() -> Tuple[bool, List[str]]:
    """
    🔧 YENİ: Live pozisyonları validate et - test sisteminde sorun tespiti için
    
    Returns:
        Tuple[bool, List[str]]: (başarılı_mı, hata_listesi)
    """
    try:
        errors = []
        
        # Temel kontroller
        if not isinstance(live_positions, dict):
            errors.append(f"live_positions tipi hatalı: {type(live_positions)}")
            return False, errors
        
        # Her pozisyon için detaylı kontrol
        validation_errors = validate_positions_data(live_positions)
        errors.extend(validation_errors)
        
        # Capital kontrolü
        if not isinstance(live_capital, (int, float)):
            errors.append(f"live_capital tipi hatalı: {type(live_capital)}")
        elif live_capital < 0:
            errors.append(f"live_capital negatif: {live_capital}")
        
        # Position limit kontrolü
        if len(live_positions) > MAX_OPEN_POSITIONS:
            errors.append(f"Pozisyon limiti aşıldı: {len(live_positions)} > {MAX_OPEN_POSITIONS}")
        
        # Timestamp kontrolü
        if _last_position_sync:
            age_minutes = (datetime.now(LOCAL_TZ) - _last_position_sync).total_seconds() / 60
            if age_minutes > 10:  # 10 dakikadan eski
                errors.append(f"Position sync yaşı fazla: {age_minutes:.1f} dakika")
        
        success = len(errors) == 0
        
        if not success:
            logging.getLogger("crypto-analytics").warning(f"⚠️ Position validation: {len(errors)} hata")
            global _position_validation_errors
            _position_validation_errors.extend(errors)
            # En fazla 20 hata sakla
            _position_validation_errors = _position_validation_errors[-20:]
        
        return success, errors
        
    except Exception as e:
        error_msg = f"Position validation exception: {e}"
        logging.getLogger("crypto-analytics").error(f"❌ {error_msg}")
        return False, [error_msg]

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
# 📦 BAŞLATMA FONKSİYONU - GÜNCELLENMİŞ
# =============================================================================

def initialize():
    """🔧 GÜNCELLEME: Sistemin temel bileşenlerini başlat - validation eklendi"""
    logger = setup_logging()
    session = create_session()
    
    logger.info("🚀 Kripto AI sistemi - SADECE LIVE TRADING")
    logger.info(f"🔑 Environment: {ENVIRONMENT}")
    logger.info(f"📊 Maksimum pozisyon: {MAX_OPEN_POSITIONS}")
    logger.info(f"⏰ Tarama aralığı: {SCAN_INTERVAL} saniye")
    logger.info("🎯 Paper Trading KALDIRILDI - Sadece Live Trading")
    
    # API key kontrolü
    if BINANCE_API_KEY and BINANCE_SECRET_KEY:
        logger.info("✅ Binance API anahtarları yüklendi")
        logger.debug(f"🔑 API Key: {BINANCE_API_KEY[:8]}...")
    else:
        logger.warning("⚠️ Binance API anahtarları bulunamadı (.env dosyasını kontrol edin)")
    
    # Position validation
    is_valid, validation_errors = validate_live_positions()
    if not is_valid:
        logger.warning(f"⚠️ Başlangıç position validation: {len(validation_errors)} sorun")
    else:
        logger.debug("✅ Position validation başarılı")
    
    return logger, session