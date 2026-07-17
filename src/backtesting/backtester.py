"""
backtester.py
=============
Simulates swing trades on detected supply/demand zones using EOD price data.

Trade logic
-----------
Every confirmed zone from zone_detector.py is a candidate trade. The
backtester walks through price history chronologically and executes a
trade the first time price enters each zone after its formation date.

Entry
  - Demand zone: price enters from above (close drops into [bottom, top])
  - Supply zone: price enters from below (close rises into [bottom, top])
  - Entry price: proximal edge of the zone (the body edge closest to price)
    Demand → proximal = top body level (close to where price entered)
    Supply → proximal = bottom body level

Stop Loss
  - Demand: below zone distal (bottom of zone)
  - Supply: above zone distal (top of zone)
  - Stop is the point where the zone is invalidated

Take Profit
  - Fixed 2R target: profit distance = 2 × risk distance
  - Demand: target = entry + 2 × (entry - stop)
  - Supply: target = entry − 2 × (stop − entry)
  - 2R is conservative for swing trading — feel free to change via config

Intraday simulation (EOD data)
  - We use daily OHLCV — we cannot know the intraday order of High/Low.
  - Pessimistic assumption: if both stop and target can be hit on the same
    candle (High >= target AND Low <= stop), the stop is hit (loss).
    This is the most conservative and realistic assumption for EOD data.
  - Trade closes when: High >= target (win) or Low <= stop (loss)

One trade per zone
  - Only the FIRST touch of a zone triggers a trade.
  - After a win or loss, the zone is marked 'traded' and ignored.
  - This reflects the institutional view: first touch = highest probability.

ML filtering (optional)
  - If a trained RF model is provided, zones are ranked by model probability.
  - Trades below the ml_prob_threshold are skipped.
  - This lets you compare baseline (all zones) vs ML-filtered performance.

Metrics computed
----------------
  Trading metrics:
    total_trades, wins, losses, open_trades
    win_rate          wins / (wins + losses)
    avg_r             mean R-multiple across all closed trades
    profit_factor     gross_profit_R / gross_loss_R
    max_drawdown_r    largest peak-to-trough decline in cumulative R
    sharpe_r          mean(R per trade) / std(R per trade) × √(trades/year)
    total_r           sum of all R-multiples (proxy for total return)

  ML metrics (when model is used):
    avg_ml_prob_wins    mean model probability for winning trades
    avg_ml_prob_losses  mean model probability for losing trades

Output files
------------
  data/backtest/<SYMBOL>_trades.csv       one row per trade
  data/backtest/<SYMBOL>_bt_report.txt    text summary
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = PROJECT_ROOT / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg["logging"]["log_level"].upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if cfg["logging"].get("log_to_file", True):
        handlers.append(logging.FileHandler(log_dir / "backtester.log"))
    if cfg["logging"].get("log_to_console", True):
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=level, handlers=handlers,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("backtester")


def _symbol_to_stem(symbol: str) -> str:
    return "IDX_" + symbol[1:] if symbol.startswith("^") else symbol.replace(".", "_")


# ── Data loading ────────────────────────────────────────────────────────────

def load_zones(symbol: str, cfg: dict) -> pd.DataFrame:
    zones_dir = PROJECT_ROOT / cfg["data"]["zones_dir"]
    path = zones_dir / (_symbol_to_stem(symbol) + "_zones.csv")
    if not path.exists():
        raise FileNotFoundError(f"Zones not found: {path}")
    df = pd.read_csv(path)
    df["formation_date"] = pd.to_datetime(df["formation_date"])
    return df


def load_processed(symbol: str, cfg: dict) -> pd.DataFrame:
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    path = processed_dir / (_symbol_to_stem(symbol) + ".csv")
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ── Single-zone trade simulation ────────────────────────────────────────────

def simulate_zone_trade(
    zone: pd.Series,
    price_df: pd.DataFrame,
    rr_ratio: float = 2.0,
    max_candles: int = 60,
) -> Optional[dict]:
    """
    Simulate one trade on a single zone.

    Parameters
    ----------
    zone       : one row from zones_df
    price_df   : full processed DataFrame with Date, Open, High, Low, Close
    rr_ratio   : reward-to-risk ratio for take-profit (default 2.0)
    max_candles: maximum candles to hold before closing as open trade (default 60)

    Returns
    -------
    dict with trade details, or None if price never entered the zone.
    """
    formation_date = zone["formation_date"]
    zone_top       = float(zone["top"])
    zone_bottom    = float(zone["bottom"])
    zone_proximal  = float(zone["proximal"])
    zone_distal    = float(zone["distal"])
    zone_type      = zone["type"]        # 'demand' or 'supply'
    structure      = zone["structure"]   # DBR / RBD / RBR / DBD

    # Price data after zone formation
    future = price_df[price_df["Date"] > formation_date].reset_index(drop=True)
    if future.empty:
        return None

    # ── Find first entry candle ──────────────────────────────────────────
    # Entry triggered when Close enters the zone range
    entry_idx = None
    for i, row in future.iterrows():
        close = row["Close"]
        if zone_bottom <= close <= zone_top:
            entry_idx = i
            break

    if entry_idx is None:
        return None  # Price never entered zone

    entry_candle   = future.iloc[entry_idx]
    entry_date     = entry_candle["Date"]
    entry_price    = zone_proximal  # Limit-order assumption at proximal edge

    # ── Set stop and target ──────────────────────────────────────────────
    stop_price = zone_distal

    if zone_type == "demand":
        # Demand: long trade, stop below zone, target above entry
        risk   = entry_price - stop_price
        target = entry_price + rr_ratio * risk
    else:
        # Supply: short trade, stop above zone, target below entry
        risk   = stop_price - entry_price
        target = entry_price - rr_ratio * risk

    if risk <= 0:
        return None  # Degenerate zone — skip

    # ── Walk forward and check stop/target ──────────────────────────────
    outcome     = "open"
    exit_price  = None
    exit_date   = None
    r_multiple  = None
    hold_candles = 0

    post_entry = future.iloc[entry_idx + 1 :].reset_index(drop=True)

    for i, row in post_entry.iterrows():
        if hold_candles >= max_candles:
            outcome    = "timeout"
            exit_date  = row["Date"]
            exit_price = row["Close"]
            r_multiple = (exit_price - entry_price) / risk if zone_type == "demand" \
                         else (entry_price - exit_price) / risk
            break

        high  = row["High"]
        low   = row["Low"]

        if zone_type == "demand":
            stop_hit   = low  <= stop_price
            target_hit = high >= target
        else:
            stop_hit   = high >= stop_price
            target_hit = low  <= target

        # Pessimistic: if both can be hit, take the stop (loss)
        if stop_hit:
            outcome    = "loss"
            exit_price = stop_price
            exit_date  = row["Date"]
            r_multiple = -1.0
            break
        elif target_hit:
            outcome    = "win"
            exit_price = target
            exit_date  = row["Date"]
            r_multiple = float(rr_ratio)
            break

        hold_candles += 1

    return {
        "zone_id":       zone["zone_id"],
        "zone_type":     zone_type,
        "structure":     structure,
        "formation_date": str(formation_date.date()),
        "entry_date":    str(entry_date.date()),
        "exit_date":     str(exit_date.date()) if exit_date else None,
        "entry_price":   round(entry_price, 2),
        "stop_price":    round(stop_price, 2),
        "target_price":  round(target, 2),
        "exit_price":    round(exit_price, 2) if exit_price else None,
        "risk_points":   round(risk, 2),
        "rr_ratio":      rr_ratio,
        "r_multiple":    round(r_multiple, 4) if r_multiple is not None else None,
        "outcome":       outcome,
        "hold_candles":  hold_candles,
        "ml_zone_prob":  float(zone.get("ml_zone_prob", np.nan))
                         if "ml_zone_prob" in zone.index else np.nan,
        "zone_strength": float(zone.get("strength_pit", zone.get("strength", np.nan))),
        "zone_top":      round(zone_top, 2),
        "zone_bottom":   round(zone_bottom, 2),
    }


# ── Full backtest ───────────────────────────────────────────────────────────

class ZoneBacktester:
    """
    Runs backtests on all detected zones (or a filtered subset).

    Parameters
    ----------
    cfg               : loaded config.yaml
    rr_ratio          : take-profit in R multiples (default 2.0)
    max_hold_candles  : max days in a trade before forced exit.
                        Default 5 (short-term swing: 3-6 days).
                        Set to 3 for very tight, 6 for a bit more room.
    ml_prob_threshold : minimum ML probability to take a trade.
                        0.0 = trade all zones (no ML filter).
                        0.5 = only zones where model says P(zone)>=0.5.
    logger            : optional logger
    """

    def __init__(
        self,
        cfg: dict,
        rr_ratio: float = 2.0,
        max_hold_candles: int = 5,
        ml_prob_threshold: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg               = cfg
        self.rr_ratio          = rr_ratio
        self.max_hold_candles  = max_hold_candles
        self.ml_prob_threshold = ml_prob_threshold
        self.logger            = logger or logging.getLogger("backtester")

    def run(self, symbol: str, zones_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate all zone trades and return trades DataFrame.

        Parameters
        ----------
        symbol   : ticker (for logging)
        zones_df : output of zone_detector (may have ml_zone_prob column)
        price_df : preprocessed OHLCV

        Returns
        -------
        pd.DataFrame  — one row per trade attempted (including no-touch zones
                        are excluded — only zones where price entered are kept)
        """
        total_zones    = len(zones_df)
        filtered_zones = zones_df

        # Apply ML probability filter if threshold is set
        if self.ml_prob_threshold > 0 and "ml_zone_prob" in zones_df.columns:
            filtered_zones = zones_df[
                zones_df["ml_zone_prob"].isna() |
                (zones_df["ml_zone_prob"] >= self.ml_prob_threshold)
            ]
            self.logger.info(
                f"ML filter (threshold={self.ml_prob_threshold}): "
                f"{len(filtered_zones)} / {total_zones} zones pass"
            )
        else:
            self.logger.info(f"No ML filter — trading all {total_zones} zones")

        trades = []
        no_touch = 0

        for _, zone in filtered_zones.iterrows():
            result = simulate_zone_trade(
                zone, price_df,
                rr_ratio=self.rr_ratio,
                max_candles=self.max_hold_candles,
            )
            if result is None:
                no_touch += 1
            else:
                trades.append(result)

        self.logger.info(
            f"Backtest: {len(filtered_zones)} zones → "
            f"{len(trades)} trades triggered, {no_touch} never touched"
        )

        if not trades:
            self.logger.warning("No trades generated.")
            return pd.DataFrame()

        return pd.DataFrame(trades)

    def calculate_metrics(self, trades_df: pd.DataFrame) -> dict:
        """
        Compute all trading performance metrics from the trades DataFrame.

        Excludes 'open' and 'timeout' trades from win/loss metrics since
        their final outcome is unknown. They are counted separately.
        """
        if trades_df.empty:
            return {}

        closed = trades_df[trades_df["outcome"].isin(["win", "loss"])].copy()
        wins   = closed[closed["outcome"] == "win"]
        losses = closed[closed["outcome"] == "loss"]
        open_t = trades_df[trades_df["outcome"] == "open"]
        timeout= trades_df[trades_df["outcome"] == "timeout"]

        n_closed  = len(closed)
        n_wins    = len(wins)
        n_losses  = len(losses)
        win_rate  = n_wins / n_closed if n_closed > 0 else 0.0

        r_values   = closed["r_multiple"].dropna().values
        total_r    = float(r_values.sum())
        avg_r      = float(r_values.mean()) if len(r_values) > 0 else 0.0

        gross_win  = float(wins["r_multiple"].sum())  if n_wins   > 0 else 0.0
        gross_loss = float(losses["r_multiple"].sum()) if n_losses > 0 else 0.0
        profit_factor = (
            abs(gross_win) / abs(gross_loss)
            if gross_loss != 0 else float("inf")
        )

        # Max drawdown in R (peak-to-trough of cumulative R curve)
        cum_r     = np.cumsum(r_values)
        peak      = np.maximum.accumulate(cum_r)
        drawdowns = peak - cum_r
        max_dd_r  = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        # Sharpe ratio (R-based, not return-based)
        # Annualise using average holding period
        avg_hold = closed["hold_candles"].mean() if n_closed > 0 else 20
        trades_per_year = 252 / max(avg_hold, 1)
        sharpe_r = (
            float(np.mean(r_values) / np.std(r_values) * np.sqrt(trades_per_year))
            if len(r_values) > 1 and np.std(r_values) > 0
            else float("nan")
        )

        # Per-structure breakdown
        structure_stats = {}
        for struct in closed["structure"].unique():
            sub = closed[closed["structure"] == struct]
            sub_wins = (sub["outcome"] == "win").sum()
            structure_stats[struct] = {
                "trades":   len(sub),
                "wins":     int(sub_wins),
                "win_rate": round(sub_wins / len(sub), 3),
                "avg_r":    round(float(sub["r_multiple"].mean()), 3),
            }

        # ML correlation (if ml_zone_prob available)
        ml_stats = {}
        if "ml_zone_prob" in closed.columns and closed["ml_zone_prob"].notna().any():
            ml_stats["avg_ml_prob_wins"]   = round(
                float(wins["ml_zone_prob"].mean()), 3) if n_wins else None
            ml_stats["avg_ml_prob_losses"] = round(
                float(losses["ml_zone_prob"].mean()), 3) if n_losses else None

        return {
            "total_zones_traded": len(trades_df),
            "total_closed":       n_closed,
            "wins":               n_wins,
            "losses":             n_losses,
            "open_trades":        len(open_t),
            "timeout_trades":     len(timeout),
            "win_rate":           round(win_rate, 4),
            "avg_r":              round(avg_r, 4),
            "total_r":            round(total_r, 4),
            "gross_win_r":        round(gross_win, 4),
            "gross_loss_r":       round(gross_loss, 4),
            "profit_factor":      round(profit_factor, 4),
            "max_drawdown_r":     round(max_dd_r, 4),
            "sharpe_r":           round(sharpe_r, 4) if not np.isnan(sharpe_r) else None,
            "avg_hold_candles":   round(float(closed["hold_candles"].mean()), 1) if n_closed else None,
            "by_structure":       structure_stats,
            "ml_correlation":     ml_stats,
        }

    def print_report(self, metrics: dict, symbol: str, mode: str = "baseline") -> str:
        """Format metrics into a readable report string."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Backtest Report  |  {symbol}  |  mode={mode}")
        lines.append(f"  RR ratio : {self.rr_ratio}  |  "
                     f"Max hold : {self.max_hold_candles} candles")
        if self.ml_prob_threshold > 0:
            lines.append(f"  ML threshold : {self.ml_prob_threshold}")
        lines.append("=" * 60)

        lines.append(f"  Total zones traded  : {metrics.get('total_zones_traded', 0)}")
        lines.append(f"  Closed trades       : {metrics.get('total_closed', 0)}")
        lines.append(f"  Wins                : {metrics.get('wins', 0)}")
        lines.append(f"  Losses              : {metrics.get('losses', 0)}")
        lines.append(f"  Open (never closed) : {metrics.get('open_trades', 0)}")
        lines.append(f"  Timeout (>{self.max_hold_candles}d) : {metrics.get('timeout_trades', 0)}")
        lines.append("")
        lines.append(f"  Win Rate            : {metrics.get('win_rate', 0):.1%}")
        lines.append(f"  Avg R per trade     : {metrics.get('avg_r', 0):+.3f}R")
        lines.append(f"  Total R             : {metrics.get('total_r', 0):+.2f}R")
        lines.append(f"  Profit Factor       : {metrics.get('profit_factor', 0):.3f}")
        lines.append(f"  Max Drawdown        : -{metrics.get('max_drawdown_r', 0):.2f}R")
        lines.append(f"  Sharpe (R-based)    : {metrics.get('sharpe_r', 'N/A')}")
        lines.append(f"  Avg Hold (candles)  : {metrics.get('avg_hold_candles', 'N/A')}")

        by_struct = metrics.get("by_structure", {})
        if by_struct:
            lines.append("")
            lines.append("  By structure:")
            for struct, s in sorted(by_struct.items()):
                lines.append(
                    f"    {struct:<6}: {s['trades']} trades  "
                    f"WR={s['win_rate']:.0%}  "
                    f"avg_R={s['avg_r']:+.2f}R"
                )

        ml_corr = metrics.get("ml_correlation", {})
        if ml_corr:
            lines.append("")
            lines.append("  ML probability correlation:")
            lines.append(f"    avg P(zone) winners : {ml_corr.get('avg_ml_prob_wins', 'N/A')}")
            lines.append(f"    avg P(zone) losers  : {ml_corr.get('avg_ml_prob_losses', 'N/A')}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_results(
        self,
        trades_df: pd.DataFrame,
        metrics: dict,
        report_str: str,
        symbol: str,
        mode: str = "baseline",
    ) -> tuple[Path, Path]:
        bt_dir = PROJECT_ROOT / self.cfg.get("backtest", {}).get(
            "output_dir", "data/backtest"
        )
        bt_dir.mkdir(parents=True, exist_ok=True)
        stem = _symbol_to_stem(symbol)

        trades_path = bt_dir / f"{stem}_{mode}_trades.csv"
        report_path = bt_dir / f"{stem}_{mode}_bt_report.txt"

        trades_df.to_csv(trades_path, index=False)
        report_path.write_text(report_str)

        self.logger.info(f"Trades saved  → {trades_path.relative_to(PROJECT_ROOT)}")
        self.logger.info(f"Report saved  → {report_path.relative_to(PROJECT_ROOT)}")
        return trades_path, report_path


# ── Pipeline entry point ────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
    rr_ratio: float = 2.0,
    max_hold_candles: int = 5,
    ml_prob_threshold: float = 0.0,
    zones_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Called from zone_pipeline.py.

    Runs TWO backtests:
      1. Baseline   — all zones, no ML filter
      2. ML-filtered — zones with ml_zone_prob >= ml_prob_threshold
                       (only if ml_zone_prob column is present)

    Returns trades_df and metrics for the baseline run.
    """
    if zones_df is None:
        zones_df = load_zones(symbol, cfg)
    price_df = load_processed(symbol, cfg)

    results = {}

    for mode, threshold in [("baseline", 0.0), ("ml_filtered", ml_prob_threshold)]:
        # Skip ML-filtered run if no model scores or threshold is 0
        if mode == "ml_filtered" and (
            "ml_zone_prob" not in zones_df.columns or
            zones_df["ml_zone_prob"].isna().all() or
            threshold == 0.0
        ):
            logger.info(f"Skipping ML-filtered backtest (no ml_zone_prob scores or threshold=0)")
            continue

        bt = ZoneBacktester(
            cfg=cfg,
            rr_ratio=rr_ratio,
            max_hold_candles=max_hold_candles,
            ml_prob_threshold=threshold,
            logger=logger,
        )

        logger.info(f"Running {mode} backtest...")
        trades_df = bt.run(symbol, zones_df, price_df)

        if trades_df.empty:
            logger.warning(f"{mode}: no trades generated.")
            continue

        metrics    = bt.calculate_metrics(trades_df)
        report_str = bt.print_report(metrics, symbol, mode=mode)

        print(report_str)
        bt.save_results(trades_df, metrics, report_str, symbol, mode=mode)

        results[mode] = {"trades": trades_df, "metrics": metrics}

    baseline_trades  = results.get("baseline", {}).get("trades",  pd.DataFrame())
    baseline_metrics = results.get("baseline", {}).get("metrics", {})
    return baseline_trades, baseline_metrics


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run zone backtest.")
    parser.add_argument("--symbol",    type=str,   default=None)
    parser.add_argument("--rr",        type=float, default=2.0,
                        help="Reward-to-risk ratio (default 2.0)")
    parser.add_argument("--max-hold",  type=int,   default=5,
                        help="Max hold period in candles (default 5 — short swing)")
    parser.add_argument("--ml-thresh", type=float, default=0.5,
                        help="ML probability threshold for filtered run (default 0.5)")
    args   = parser.parse_args()
    cfg    = load_config()
    logger = setup_logging(cfg)
    symbol = args.symbol or cfg["data"]["symbol"]

    logger.info(f"=== Zone Backtester — {symbol} ===")
    run_backtest(
        symbol, cfg, logger,
        rr_ratio=args.rr,
        max_hold_candles=args.max_hold,
        ml_prob_threshold=args.ml_thresh,
    )


if __name__ == "__main__":
    main()
