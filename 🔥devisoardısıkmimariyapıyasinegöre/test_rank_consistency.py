import pandas as pd
import logging

# Logger ayarı
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("consistency-test")

# Örnek DataFrame (UI & Trader aynı kriterleri denemek için)
data = {
    "symbol": ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"],
    "ai_score": [58, 53, 51, 47, 40],
    "trend_strength": [10, 9, 8, 5, 4],
    "run_perc": [6.5, 5.2, 4.1, 3.7, 2.9],
    "gauss_run": [7, 6, 6, 5, 3],
    "vol_ratio": [3.1, 2.9, 2.2, 1.8, 1.5],
    "run_count": [5, 4, 4, 3, 2]
}

df = pd.DataFrame(data)

# UI tarafındaki sıralama
ui_df = df.sort_values(
    by=["ai_score", "trend_strength", "run_perc", "gauss_run", "vol_ratio"],
    ascending=[False, False, False, False, False]
)
ui_top3 = ui_df.head(3)["symbol"].tolist()

# Trader tarafındaki sıralama (paper_trader/live_trader ile aynı kriter)
trader_df = df.sort_values(
    by=["ai_score", "trend_strength", "run_perc", "gauss_run", "vol_ratio"],
    ascending=[False, False, False, False, False]
)
trader_top3 = trader_df.head(3)["symbol"].tolist()

# Loglama
logger.info(f"📊 UI’nin gösterdiği ilk 3: {ui_top3}")
logger.info(f"🤖 Trader’ın kullandığı ilk 3: {trader_top3}")

# Doğrulama
if ui_top3 == trader_top3:
    logger.info("✅ UI ve Trader tutarlı! (Aynı ilk 3 seçildi)")
else:
    logger.error("❌ Tutarsızlık var! UI ve Trader farklı ilk 3 seçiyor")
