"""
run_honest_backtest.py
======================
Runs the CORRECTED pnl_backtester across all real NSE symbols with an honest,
leakage-free configuration and prints a consolidated true-metrics table.

Honest configuration
---------------------
  - Trailing stop OFF  (trail_atr_mult=0) -> fixed 2:1 target
  - Conservative intrabar resolution (stop-first on same-candle) [bug fixed]
  - All zone structures (DBR, RBD, RBR, DBD)
  - No strength cherry-picking (min=0.0, max=1.0)
  - RR 2:1, 1% risk, Rs.500,000 capital, 5-day max hold, no confirm gate
"""
import logging, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import yaml
from src.backtesting.pnl_backtester import run_pnl_backtest

cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))

logger = logging.getLogger("honest_bt")
logging.basicConfig(level=logging.ERROR)  # quiet — we only want the table

SYMBOLS = ["^NSEI", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "IDFCFIRSTB.NS"]

def run_all(use_ml):
    rows = []
    for sym in SYMBOLS:
        try:
            trades, m = run_pnl_backtest(
                symbol=sym, cfg=cfg, logger=logger,
                start_capital=500_000.0, risk_pct=0.01, rr_ratio=2.0,
                max_hold_candles=5, confirm_entry=False,
                min_strength=0.0, max_strength=1.0,
                trade_structures=["DBR", "RBD", "RBR", "DBD"],
                trail_atr_mult=0.0, use_ml_zones=use_ml,
            )
        except Exception as e:
            print(f"  {sym}: ERROR {e}")
            continue
        if not m:
            print(f"  {sym}: no trades")
            continue
        n = m.get("wins", 0) + m.get("losses", 0)
        rows.append({
            "symbol": sym,
            "trades": n,
            "win_rate_%": round(m.get("win_rate", 0) * 100, 1),
            "profit_factor": round(m.get("profit_factor", float("nan")), 3),
            "total_return_%": round(m.get("total_return_pct", 0), 2),
            "max_dd_%": round(m.get("max_drawdown_pct", 0), 2),
            "sharpe": round(m.get("sharpe_r") or float("nan"), 3),
            "avg_R": round(m.get("avg_r", float("nan")), 3),
        })
    return pd.DataFrame(rows)

def summarize(df, label):
    print(f"\n===== {label} =====")
    if df.empty:
        print("  (no results)")
        return
    print(df.to_string(index=False))
    # portfolio-level pooled win rate weighted by trade count
    tot = df["trades"].sum()
    wr = (df["win_rate_%"] * df["trades"]).sum() / tot if tot else 0
    print(f"  Pooled: {tot} trades | trade-weighted WR = {wr:.1f}% | "
          f"median PF = {df['profit_factor'].median():.3f} | "
          f"median return = {df['total_return_%'].median():.2f}%")

if __name__ == "__main__":
    ml = run_all(use_ml=True)
    algo = run_all(use_ml=False)
    summarize(ml, "ML-detected zones (XGBoost)")
    summarize(algo, "Rule-based zones")
    ml.assign(zone_source="ML_XGBoost").to_csv(ROOT / "true_metrics_ml_zones.csv", index=False)
    algo.assign(zone_source="rule_based").to_csv(ROOT / "true_metrics_rule_zones.csv", index=False)
    print("\nSaved: true_metrics_ml_zones.csv, true_metrics_rule_zones.csv")
