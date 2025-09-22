"""
💰 Trading Modülü - LIVE TRADING + ANALYZER
Sinyal analizi ve live trading sistemi
"""

from .analyzer import (
    analyze_symbol_with_ai, batch_analyze_with_ai, filter_signals,
    get_top_signals, analyze_signal_quality, update_signal_scores
)

# Live Trading import (conditional - sadece varsa)
try:
    from .live_trader import (
        start_live_trading, stop_live_trading, is_live_trading_active,
        get_live_trading_status, get_live_bot_status_for_symbol, get_auto_sltp_count
    )
    LIVE_TRADER_AVAILABLE = True
except ImportError:
    LIVE_TRADER_AVAILABLE = False

__all__ = [
    # Analyzer
    'analyze_symbol_with_ai', 'batch_analyze_with_ai', 'filter_signals',
    'get_top_signals', 'analyze_signal_quality', 'update_signal_scores'
]

# Live Trading fonksiyonlarını da ekle (eğer mevcut ise)
if LIVE_TRADER_AVAILABLE:
    __all__.extend([
        'start_live_trading', 'stop_live_trading', 'is_live_trading_active',
        'get_live_trading_status', 'get_live_bot_status_for_symbol', 'get_auto_sltp_count'
    ])