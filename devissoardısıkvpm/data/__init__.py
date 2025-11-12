"""
📁 Data Modülü
Veri çekme, API bağlantıları ve veritabanı işlemleri
"""

from .fetch_data import (
    get_usdt_perp_symbols, fetch_klines, get_current_price,
    fetch_multiple_symbols, validate_ohlcv_data, get_market_summary
)

from .database import (
    setup_csv_files, log_trade_to_csv, log_capital_to_csv,
    load_trades_from_csv, load_capital_history_from_csv,
    get_trade_statistics, backup_csv_files, cleanup_old_backups,
    export_to_excel
)

__all__ = [
    # Fetch Data
    'get_usdt_perp_symbols', 'fetch_klines', 'get_current_price',
    'fetch_multiple_symbols', 'validate_ohlcv_data', 'get_market_summary',
    
    # Database
    'setup_csv_files', 'log_trade_to_csv', 'log_capital_to_csv',
    'load_trades_from_csv', 'load_capital_history_from_csv',
    'get_trade_statistics', 'backup_csv_files', 'cleanup_old_backups',
    'export_to_excel'
]