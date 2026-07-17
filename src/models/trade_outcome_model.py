"""
trade_outcome_model.py
======================
XGBoost classifier that predicts TRADE OUTCOME (win vs loss) for a swing
trade taken from a rule-based supply/demand zone.

Why this model exists
---------------------
The earlier `zone_detection_model.py` trained XGBoost to reproduce the
rule-based *zone labels* — i.e. it answered "is this window a zone?".
That target has no reason to correlate with profitability, and the backtest
confirmed it: average P(zone) was 0.862 for winning trades vs 0.872 for
losers — no discriminative power. Filtering trades by P(zone) cannot help.

This module changes the target to the thing we actually care about:

    label = 1  if the trade taken from this zone WON  (r_multiple > 0)
    label = 0  if it lost / timed out at a loss       (r_multiple <= 0)

The trained model is then used as a *trade filter* on top of the rule-based
zone detector: only take zones whose predicted P(win) exceeds a threshold.
Whether this actually improves trading metrics is an open empirical question
that we answer out-of-sample below — it is NOT assumed.

Leakage discipline (the whole point of a valid result)
------------------------------------------------------
1. FEATURES are restricted to a whitelist of quantities knowable AT ZONE
   FORMATION. All post-hoc columns in the zones file are dropped:
     strength, quality_score, adjusted_strength_posthoc, merged_count,
     test_count, status, last_test_date, invalidation_date, and every
     absolute price level (top/bottom/proximal/distal/midpoint) which would
     merely encode the symbol and era. We keep ATR-normalised and `_pit`
     (point-in-time) features only.
2. The LABEL comes from a corrected, conservative backtest (stop-first
   intrabar resolution, fixed 2:1 target, no trailing-stop look-ahead).
3. EVALUATION is temporal walk-forward (TimeSeriesSplit): train on past
   trades, test on strictly future trades. The probability threshold is
   tuned on the TRAIN fold only and then applied blindly to the TEST fold.
   Success is judged on out-of-sample TRADING metrics, not accuracy.

Sample-size caveat
------------------
Pooling all six symbols yields only a few hundred trades. This is small for
machine learning; we therefore keep the model shallow, report a
logistic-regression baseline for comparison, and treat any improvement with
appropriate scepticism. A negative result here is a legitimate finding.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.backtesting.pnl_backtester import run_pnl_backtest  # noqa: E402

# ── Point-in-time feature whitelist ───────────────────────────────────────────
# Only quantities knowable at or before formation_date. Categorical `structure`
# and `type` are encoded separately. NaNs are left in place — XGBoost handles
# them natively; the logistic baseline imputes with the train-fold median.
PIT_NUMERIC_FEATURES = [
    "width_atr", "base_length", "base_wick_frac", "avg_atr",
    "departure_body_atr", "departure_close_ratio",
    "leg_out_clear_atr", "leg_out_disp_atr", "leg_out_candles",
    "leg_out_velocity", "leg_out_vol_exp",
    "leg_in_disp_atr", "leg_in_candles", "leg_in_velocity",
    "disp_base_ratio", "arrival_cleanliness",
    "departure_volume_ratio", "base_volume_ratio",
    "trend_score", "strength_pit", "freshness_score",
    "weekly_dist_atr", "weekly_zone_strength", "weekly_confluence_score",
]
# Boolean PIT features (cast to int; NaN -> 0)
PIT_BOOL_FEATURES = [
    "trend_aligned", "weekly_trend_align", "weekly_in_zone",
    "weekly_zone_fresh", "weekly_confirmed",
]
# Explicitly banned (post-hoc / future / leakage / identifiers)
BANNED = {
    "strength", "quality_score", "adjusted_strength_posthoc",
    "merged_count", "test_count", "status", "last_test_date",
    "invalidation_date", "top", "bottom", "midpoint", "proximal",
    "distal", "width", "zone_id", "formation_idx",
}

SYMBOLS = ["^NSEI", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "IDFCFIRSTB.NS"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _stem(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "IDX_")


def build_dataset(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Run the corrected backtest per symbol, join each trade to its
    point-in-time zone features, pool, and sort by entry_date.
    """
    frames = []
    for sym in SYMBOLS:
        trades, _ = run_pnl_backtest(
            symbol=sym, cfg=cfg, logger=logger,
            start_capital=500_000.0, risk_pct=0.01, rr_ratio=2.0,
            max_hold_candles=5, confirm_entry=False,
            min_strength=0.0, max_strength=1.0,
            trade_structures=["DBR", "RBD", "RBR", "DBD"],
            trail_atr_mult=0.0, use_ml_zones=False,
        )
        if trades is None or trades.empty:
            continue
        zones = pd.read_csv(PROJECT_ROOT / "data" / "zones" / f"{_stem(sym)}_zones.csv")
        keep = [c for c in zones.columns
                if c in PIT_NUMERIC_FEATURES + PIT_BOOL_FEATURES + ["zone_id", "structure", "type"]]
        zfeat = zones[keep]
        merged = trades[["zone_id", "entry_date", "r_multiple", "outcome"]].merge(
            zfeat, on="zone_id", how="left")
        merged["symbol"] = sym
        frames.append(merged)
        logger.info(f"{sym}: {len(merged)} trades joined to zone features")

    df = pd.concat(frames, ignore_index=True)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    # Label: 1 = win (positive R), 0 = loss/timeout-at-loss
    df["label"] = (df["r_multiple"] > 0).astype(int)

    # Encode categoricals
    df["is_demand"] = (df["type"] == "demand").astype(int)
    struct_dummies = pd.get_dummies(df["structure"], prefix="struct").astype(int)
    df = pd.concat([df, struct_dummies], axis=1)

    for c in PIT_BOOL_FEATURES:
        if c in df.columns:
            df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in PIT_NUMERIC_FEATURES if c in df.columns]
    cols += [c for c in PIT_BOOL_FEATURES if c in df.columns]
    cols += ["is_demand"] + [c for c in df.columns if c.startswith("struct_")]
    return [c for c in cols if c not in BANNED]


# ── Trading-impact helper ─────────────────────────────────────────────────────
def trade_stats(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return {"n": 0, "wr": np.nan, "pf": np.nan, "avg_r": np.nan, "total_r": 0.0}
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return {
        "n": len(r),
        "wr": float((r > 0).mean()),
        "pf": float(wins / losses) if losses > 0 else float("inf"),
        "avg_r": float(r.mean()),
        "total_r": float(r.sum()),
    }
