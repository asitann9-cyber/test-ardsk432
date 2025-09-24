# === tests/test_tp_sl_deep_probe.py ==========================================
# Canlı Testnet mini simülasyon: SL/TP → CLOSED → CSV → Perf zincirini doğrular.
# WebSocket yok; REST polling ile izler.

# sys.path: proje kökünü ekle (config, trading, data erişilsin)
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import time
import pandas as pd

from config import ENVIRONMENT, TRADES_CSV
# LiveTrader yoksa LiveTradingBot'u alias'la
try:
    from trading.live_trader import LiveTrader
except Exception:
    from trading.live_trader import LiveTradingBot as LiveTrader

from data.fetch_data import get_usdt_perp_symbols, get_current_price
from data.database import setup_csv_files, load_trades_from_csv, calculate_performance_metrics

# opsiyonel
try:
    from trading.analyzer import get_top_symbols
except Exception:
    get_top_symbols = None

try:
    from app import get_live_trading_status
except Exception:
    get_live_trading_status = None


def _pos_qty_zero(pos: dict) -> bool:
    """Pozisyon miktarını anahtar ismine bakmaksızın sıfır say (qty/quantity/positionAmt/size)."""
    if not pos:
        return True
    for k in ("qty", "quantity", "positionAmt", "size"):
        v = pos.get(k)
        try:
            if abs(float(v)) > 0:
                return False
        except Exception:
            continue
    return True


def run_probe_for_symbol(bot: LiveTrader, symbol: str, usd: float, tp_pct: float, sl_pct: float, timeout: int) -> int:
    # 1) Fiyat ve miktar
    price = get_current_price(symbol)
    if not price or price <= 0:
        print(f"SKIP: {symbol} no price available.")
        return 0

    qty = max(0.001, usd / float(price))  # basit miktar; lot/precision bot içinde yuvarlanır

    # 2) Pozisyon aç (LiveTradingBot için uyum: force_qty)
    try:
        opened = False
        try:
            # Tercih edilen imza
            opened = bot.open_position(symbol=symbol, side="LONG", qty=qty,
                                       tp_pct=tp_pct, sl_pct=sl_pct, reason="TP_SL_PROBE")
        except TypeError:
            # Bazı sürümlerde open_position(signal_dict) beklenir
            signal = {
                "symbol": symbol, "run_type": "LONG", "ai_score": 50,
                "run_count": 1, "run_perc": 0.0, "gauss_run": 0,
                "vol_ratio": 0.0, "deviso_ratio": 0.0,
                "force_qty": qty, "reason": "TP_SL_PROBE",
            }
            opened = bot.open_position(signal)
        if not opened:
            print(f"ERR: {symbol} open_position returned False")
            return 0
    except Exception as e:
        print(f"ERR: {symbol} open failed: {e}")
        return 0

    # 3) TP/SL kimliklerini pozisyon sözlüğünden oku (varsa)
    pos = getattr(bot, "live_positions", {}).get(symbol, {})
    tp_oid  = pos.get("tp_order_id", "N/A")
    tp_coid = pos.get("tp_client_oid", pos.get("tp_conditional_id", "N/A"))
    tp_stop = pos.get("tp_price",  pos.get("tp_stop_price", "N/A"))
    sl_oid  = pos.get("sl_order_id", "N/A")
    sl_coid = pos.get("sl_client_oid", pos.get("sl_conditional_id", "N/A"))
    sl_stop = pos.get("sl_price",  pos.get("sl_stop_price", "N/A"))

    print(f"=== {symbol}: open usd={usd} tp={tp_pct}% sl={sl_pct}% ===")
    print(f"SET: TP oid={tp_oid} coid={tp_coid} stop={tp_stop} | SL oid={sl_oid} coid={sl_coid} stop={sl_stop}")

    # 4) İzleme döngüsü
    start = time.time()
    closed_via = None
    while time.time() - start < timeout:
        try:
            bot.check_filled_orders()
        except Exception as e:
            print(f"ERR: check_filled_orders: {e}")

        print(f"CHK: {symbol} TP(oid={tp_oid},coid={tp_coid}) SL(oid={sl_oid},coid={sl_coid})")

        # Pozisyon kapandı mı?
        lp = getattr(bot, "live_positions", {})
        if symbol not in lp or _pos_qty_zero(lp.get(symbol)):
            if closed_via is None:
                closed_via = "RECONCILE"
                print(f"HIT: ? UNKNOWN @ {symbol}")
            break

        time.sleep(1)

    # 5) Timeout ise manuel kapat
    if closed_via is None:
        try:
            bot.close_position(symbol, reason="TIMEOUT")
            closed_via = "TIMEOUT"
        except Exception as e:
            print(f"ERR: manual close {symbol}: {e}")
            return 0

    print(f"CLOSE: {symbol} via {closed_via} -> CSV")
    time.sleep(2)  # CSV yazımını beklet

    # 6) CSV doğrulaması
    if not os.path.exists(TRADES_CSV):
        setup_csv_files()

    df = load_trades_from_csv()
    sym_rows = df[df["symbol"] == symbol]
    if sym_rows.empty:
        print("ASSERT-CSV-CLOSED-ROW-ABSENT")
        raise AssertionError("No row in CSV for symbol")

    last = sym_rows.iloc[-1]
    if str(last.get("status", "")).upper() != "CLOSED":
        print("ASSERT-CSV-CLOSED-ROW-ABSENT")
        raise AssertionError("No CLOSED row in CSV for symbol")

    # entry/exit kolon isimleri farklı olabilir; ikisini de kabul et
    entry_field = "entry" if "entry" in df.columns else ("entry_price" if "entry_price" in df.columns else None)
    exit_field  = "exit"  if "exit"  in df.columns else ("exit_price"  if "exit_price"  in df.columns else None)
    assert entry_field and exit_field, "ASSERT-CSV-SCHEMA-INVALID"
    assert pd.notna(last[entry_field]) and pd.notna(last[exit_field]) and pd.notna(last["pnl"]), "ASSERT-CSV-FIELDS-NAN"

    print(f"CSV: write status=CLOSED symbol={symbol} pnl={last['pnl']} -> {TRADES_CSV}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="TP/SL Deep Probe Script")
    parser.add_argument("--usd", type=float, default=5.0, help="USD amount per position")
    parser.add_argument("--tp", type=float, default=0.15, help="Take profit percentage")
    parser.add_argument("--sl", type=float, default=0.15, help="Stop loss percentage")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    args = parser.parse_args()

    print(f"TP/SL DEEP PROBE | ENV={ENVIRONMENT} | TRADES_CSV={TRADES_CSV}")

    if not os.path.exists(TRADES_CSV):
        setup_csv_files()

    # Baseline (calculate_performance_metrics sizde argümansız)
    _ = load_trades_from_csv()  # dosya varlığını garanti etmiş olalım
    base_perf = calculate_performance_metrics()
    base_total = base_perf.get("total_trades", 0)

    # Semboller
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        all_syms = get_usdt_perp_symbols()
        if get_top_symbols:
            try:
                symbols = get_top_symbols(all_syms, top_n=3)
            except Exception:
                symbols = all_syms[:3]
        else:
            symbols = all_syms[:3]

    bot = LiveTrader()
    closed_count = 0

    for sym in symbols[:3]:  # en fazla 3
        try:
            closed_count += run_probe_for_symbol(bot, sym, args.usd, args.tp, args.sl, args.timeout)
        except AssertionError as e:
            print(f"ERR: {sym} assert: {e}")
        except Exception as e:
            print(f"ERR: {sym} fatal: {e}")

    # Final doğrulamalar
    if not os.path.exists(TRADES_CSV):
        print("ERR: CSV path issue.")
        return

    df = load_trades_from_csv()
    need_cols = {"symbol", "status", "pnl"}
    assert need_cols.issubset(df.columns), "ASSERT-CSV-SCHEMA-INVALID"
    assert (df["status"].str.upper() == "CLOSED").any(), "ASSERT-CSV-CLOSED-ROW-ABSENT"

    # Metrikler (argümansız)
    perf = calculate_performance_metrics()
    assert perf.get("total_trades", 0) >= base_total + closed_count, "ASSERT-PERF-TOTAL-NOT-INCREASED"

    # Tail
    print("TAIL:")
    try:
        print(df.tail(5).to_string(index=False))
    except Exception:
        print(df.tail(5))

    # Perf
    print(f"PERF: total_trades={perf.get('total_trades')} win_rate={perf.get('win_rate')} pnl_total={perf.get('pnl_total')}")

    # App status (opsiyonel)
    if get_live_trading_status:
        try:
            print(f"STATUS: {get_live_trading_status()}")
        except Exception:
            pass

    if closed_count == 0:
        raise AssertionError("ASSERT-NO-CLOSED-TRADES")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERR: {e}")
# ============================================================================
