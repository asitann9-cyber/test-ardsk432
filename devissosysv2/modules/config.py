# Modules Configuration
# Modüller için yapılandırma ayarları

import os
from typing import Dict, Any

class ModuleConfig:
    """Modül yapılandırma sınıfı"""
    
    def __init__(self):
        self.modules = {
            "ema_scanner": {
                "name": "EMA200 Scanner",
                "description": "Binance Futures EMA200 Crossover Scanner",
                "version": "1.0.0",
                "enabled": True,
                "settings": {
                    "max_coins": 30,
                    "timeframe": "5m",
                    "ema_length": 200,
                    "min_ratio": 0
                }
            },
            "rsi_macd_scanner": {
                "name": "RSI/MACD Scanner",
                "description": "Binance Futures RSI & MACD Signal Scanner",
                "version": "1.0.0",
                "enabled": True,
                "settings": {
                    "timeframe": "1h",
                    "limit": 50,
                    "max_bars_ago": 50,
                    "priority_signals": ["C20L", "C20S", "M5S", "M5L"]
                }
            },
            "super_signal_scanner": {
                "name": "Deviso Super Signal",
                "description": "EMA200 + RSI/MACD Kombinasyon Scanner",
                "version": "1.0.0",
                "enabled": True,
                "settings": {
                    "max_coins": 50,
                    "timeframe": "5m",
                    "scan_interval": 60,
                    "priority_signals": ["C20L", "C20S", "M5S", "M5L"]
                }
            },
            "pro_signal_scanner": {
                "name": "Deviso Pro Signal",
                "description": "Ultra Quant Scanner - Çoklu Strateji Kombinasyonu",
                "version": "1.0.0",
                "enabled": True,
                "settings": {
                    "max_coins": 50,
                    "scan_interval": 300,
                    "min_signal_score": 5,
                    "timeframes": ["5m", "15m", "1h"],
                    "signal_categories": ["MEGA SIGNAL", "PRO SIGNAL", "STANDARD SIGNAL", "WEAK SIGNAL"]
                }
            },
            "live_test_scanner": {
                "name": "Deviso Live Test",
                "description": "Pro Signal Demo Trading & Performance Analytics",
                "version": "1.0.0",
                "enabled": True,
                "settings": {
                    "max_trades": 10,
                    "demo_balance": 10000,
                    "position_size": 0.1,
                    "auto_trade": True,
                    "min_signal_score": 7
                }
            },
            "live_trading_scanner": {
                "name": "Deviso Live Trading",
                "description": "Binance Futures Live Trading with Real API",
                "version": "1.0.0",
                "enabled": True,
                "settings": {
                    "max_trades": 10,
                    "position_size": 0.1,
                    "auto_trade": True,
                    "min_signal_score": 9,
                    "use_testnet": True,
                    "api_key": "",
                    "api_secret": "",
                    "futures_balance": 1000,
                    "risk_per_trade": 0.02,
                    "max_risk_per_day": 0.1,
                    "leverage": 10,
                    "margin_type": "ISOLATED",
                    "min_position_value": 10,
                    "max_position_value": 100,
                    "auto_adjust_leverage": True,
                    "use_trailing_stop": False,
                    "trailing_stop_distance": 0.02,
                    "auto_update_balance": True,
                    "balance_update_interval": 30
                }
            }
            # Gelecekte eklenecek modüller buraya eklenecek
        }
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Belirli bir modülün yapılandırmasını getir"""
        return self.modules.get(module_name, {})
    
    def get_all_modules(self) -> Dict[str, Any]:
        """Tüm modülleri getir"""
        return self.modules
    
    def is_module_enabled(self, module_name: str) -> bool:
        """Modülün aktif olup olmadığını kontrol et"""
        module = self.modules.get(module_name, {})
        return module.get("enabled", False)
    
    def update_module_setting(self, module_name: str, setting: str, value: Any):
        """Modül ayarını güncelle"""
        if module_name in self.modules:
            if "settings" not in self.modules[module_name]:
                self.modules[module_name]["settings"] = {}
            self.modules[module_name]["settings"][setting] = value

# Global yapılandırma örneği
config = ModuleConfig()
