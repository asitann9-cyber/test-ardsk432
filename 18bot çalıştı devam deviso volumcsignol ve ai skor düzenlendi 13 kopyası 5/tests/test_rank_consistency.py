# tests/test_rank_consistency.py
"""
🔍 Entegre Test: LiveTradingBot + Config + Database + App
Amaç: SL/TP tetiklendiğinde CSV ve config senkronizasyonunun doğru çalıştığını test etmek.
"""

import sys, os
import pandas as pd
from datetime import datetime

# ✅ Proje root yolunu ekle (config, trading, data erişilebilsin)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from trading.live_trader import LiveTradingBot
from data.database import log_trade_to_csv
from app import get_live_trading_status

TEST_CSV = "ai_crypto_trades.csv"  # normal CSV’yi kontrol ediyoruz


def run_integration_test():
    print("✅ Entegre test başlatıldı...")

    # 1️⃣ Dummy pozisyon oluştur
    config.live_positions = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.1,
            "entry_price": 50000.0,
            "invested_amount": 5000.0,
            "stop_loss": 49000.0,
            "take_profit": 51000.0,
            "entry_time": datetime.now(config.LOCAL_TZ),
            "signal_data": {
                "ai_score": 85,
                "run_type": "LONG",
                "run_count": 3,
                "run_perc": 75.0,
                "gauss_run": 2,
                "vol_ratio": 1.2,
                "deviso_ratio": 0.9,
            },
            "auto_sltp": True,
        }
    }

    bot = LiveTradingBot()

    # 2️⃣ TP tetiklenmiş gibi simüle et
    fake_filled_order = {"avgPrice": 51000.0}
    bot._handle_auto_close("BTCUSDT", "Take Profit - Auto", fake_filled_order)

    # 3️⃣ SL tetiklenmiş gibi simüle et
    config.live_positions = {
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "side": "LONG",
            "quantity": 0.5,
            "entry_price": 2000.0,
            "invested_amount": 1000.0,
            "stop_loss": 1900.0,
            "take_profit": 2100.0,
            "entry_time": datetime.now(config.LOCAL_TZ),
            "signal_data": {
                "ai_score": 72,
                "run_type": "LONG",
                "run_count": 2,
                "run_perc": 65.0,
                "gauss_run": 1,
                "vol_ratio": 0.8,
                "deviso_ratio": 0.7,
            },
            "auto_sltp": True,
        }
    }

    fake_filled_order2 = {"avgPrice": 1900.0}
    bot._handle_auto_close("ETHUSDT", "Stop Loss - Auto", fake_filled_order2)

    # 4️⃣ CSV çıktısını kontrol et
    if os.path.exists(TEST_CSV):
        df = pd.read_csv(TEST_CSV)
        print("📊 CSV son satırlar:")
        print(df.tail(3))
    else:
        print("❌ CSV bulunamadı!")

    # 5️⃣ Config & App senkronizasyonu kontrol et
    status = get_live_trading_status()
    print("📊 Live Trading Status:", status)

    print("✅ Entegre test tamamlandı.")


if __name__ == "__main__":
    run_integration_test()
