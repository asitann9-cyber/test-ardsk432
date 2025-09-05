# Deviso System Modules Package
# Bu dosya modules klasörünü Python paketi yapar

__version__ = "1.0.0"
__author__ = "Deviso System"

# Mevcut modüller
AVAILABLE_MODULES = {
    "ema_scanner": "Binance Futures EMA200 Scanner",
    # Gelecekte eklenecek modüller buraya eklenecek
}

def get_module_info():
    """Mevcut modüller hakkında bilgi döndür"""
    return AVAILABLE_MODULES
