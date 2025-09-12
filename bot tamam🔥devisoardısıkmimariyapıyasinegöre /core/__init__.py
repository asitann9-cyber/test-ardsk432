"""
🔧 Core Modülü
AI modeli, teknik göstergeler ve yardımcı fonksiyonlar
🔥 CVD + ROC Momentum sistemi güncellemesi
"""

from .utils import (
    gauss_sum, safe_log, pine_sma, crossunder, crossover, cross,
    validate_dataframe, safe_division, format_number, 
    calculate_percentage_change, clamp, is_valid_symbol, clean_numeric_value
)

from .ai_model import CryptoMLModel, ai_model

from .indicators import (
    calculate_zigzag_high, calculate_zigzag_low, 
    calculate_deviso_ratio, 
    compute_cvd_momentum_metrics,  # 🔥 YENİ: CVD ana fonksiyonu
    compute_consecutive_metrics    # 🔧 ESKİ: Backward compatibility için korundu
)

__all__ = [
    # Utils
    'gauss_sum', 'safe_log', 'pine_sma', 'crossunder', 'crossover', 'cross',
    'validate_dataframe', 'safe_division', 'format_number', 
    'calculate_percentage_change', 'clamp', 'is_valid_symbol', 'clean_numeric_value',
    
    # AI Model
    'CryptoMLModel', 'ai_model',
    
    # Indicators
    'calculate_zigzag_high', 'calculate_zigzag_low', 
    'calculate_deviso_ratio', 
    'compute_cvd_momentum_metrics',  # 🔥 YENİ: CVD ana fonksiyonu
    'compute_consecutive_metrics'    # 🔧 ESKİ: Geçici uyumluluk
]