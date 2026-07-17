"""
pnl_backtester.py
=================
Simulates swing trades on detected supply/demand zones and reports
actual Profit & Loss in Indian Rupees (₹), not just R-multiples.

How this differs from backtester.py
-------------------------------------
The existing backtester.py measures performance in abstract R-multiples
(e.g. +2R, -1R). That is excellent for strategy-level analysis because
it normalises across different price levels and volatility regimes.

This file answers a different and more concrete question:
  "If I started with ₹5,00,000, how much money would I have made or lost?"

It adds:
  1. Starting capital  — your initial account size in ₹
  2. Risk per trade    — what % of current capital you risk on each trade
                         (e.g. 1% of ₹5,00,000 = ₹5,000 at risk)
  3. Position sizing   — shares = risk_amount / (entry_price - stop_price)
                         This gives you exact ₹ exposure per trade.
  4. P&L per trade     — (exit_price - entry_price) × shares for longs
                         (entry_price - exit_price) × shares for shorts
  5. Running equity    — capital grows/shrinks after every trade
  6. Drawdown in ₹     — peak-to-trough drop in your account value
  7. Return %          — (final_capital - start_capital) / start_capital

Trade logic (identical to backtester.py)
------------------------------------------
  Entry   : first time Close enters zone range after formation date
  Stop    : distal edge of zone (where zone is invalidated)
  Target  : 2R (default) — adjustable via rr_ratio argument
  Exit    : stop hit → loss of -1R × risk_amount
             target hit → profit of +rr_ratio × risk_amount
             timeout → exit at Close after max_hold_candles days

Pessimistic assumption on EOD data
-------------------------------------
If both High >= target AND Low <= stop on the same candle, we assume
the stop was hit first (worst-case). This is the most conservative and
realistic assumption when using daily bars.

Position sizing model
----------------------
Risk-based sizing:
  risk_amount = current_capital × risk_pct   (e.g. 1% of ₹5,00,000 = ₹5,000)
  shares      = risk_amount / risk_per_share
  risk_per_share = |entry_price - stop_price|

  Win  → profit = shares × risk_per_share × rr_ratio = risk_amount × rr_ratio
  Loss → loss   = shares × risk_per_share             = risk_amount

This means no matter what the stock price is, your maximum loss per
trade is always exactly risk_pct of your capital at that moment.

Slippage & commission
----------------------
Optional slippage (in %) is applied to the entry price to simulate
realistic fill. Commission (brokerage) is a flat amount per trade.
These default to 0 so you can see the "clean" numbers first.

Output files
------------
  data/backtest/<SYMBOL>_pnl_trades.csv   — one row per trade with ₹ P&L
  data/backtest/<SYMBOL>_pnl_report.txt   — full text summary
  data/backtest/<SYMBOL>_equity_curve.csv — date, equity after each trade

Usage
-----
    # Run with defaults (₹5L capital, 1% risk per trade, 2:1 RR, 5-day hold)
    python src/backtesting/pnl_backtester.py

    # Custom parameters
    python src/backtesting/pnl_backtester.py \\
        --symbol "^NSEI" \\
        --capital 1000000 \\
        --risk-pct 0.01 \\
        --rr 2.0 \\
        --max-hold 10 \\
        --slippage 0.001 \\
        --commission 20
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"


# ── Config & logging ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = PROJECT_ROOT / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    level    = getattr(logging, cfg["logging"].get("log_level", "INFO").upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if cfg["logging"].get("log_to_file", True):
        handlers.append(logging.FileHandler(log_dir / "pnl_backtester.log"))
    if cfg["logging"].get("log_to_console", True):
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("pnl_backtester")


def _symbol_to_stem(symbol: str) -> str:
    return "IDX_" + symbol[1:] if symbol.startswith("^") else symbol.replace(".", "_")


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_zones(symbol: str, cfg: dict, use_ml_zones: bool = False) -> pd.DataFrame:
    """
    Load zone file.

    Parameters
    ----------
    use_ml_zones : if True, load ML-detected zones (_ml_zones.csv).
                   if False (default), load rule-based zones (_zones.csv).

    Rule-based zones  — detected by hand-crafted ATR/displacement rules.
                        Have full strength_pit, quality_score, trend_aligned columns.
    ML zones          — detected by XGBoost trained on rule-based labels.
                        Have ml_prob (model confidence) but strength_pit is NaN.
                        Use --min-strength to filter on ml_prob instead.
    """
    stem   = _symbol_to_stem(symbol)
    suffix = "_ml_zones.csv" if use_ml_zones else "_zones.csv"
    path   = PROJECT_ROOT / cfg["data"]["zones_dir"] / (stem + suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"Zones file not found: {path}\n"
            f"{'Run zone_pipeline.py first.' if not use_ml_zones else 'Run zone_pipeline.py to generate ML zones.'}"
        )
    df = pd.read_csv(path)
    df["formation_date"] = pd.to_datetime(df["formation_date"])

    # For ML zones: strength_pit is NaN — fill from ml_prob so the
    # strength filter works transparently on both zone types
    if use_ml_zones and "ml_prob" in df.columns:
        if df["strength_pit"].isna().all():
            df["strength_pit"] = df["ml_prob"]

    return df


def load_processed(symbol: str, cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["data"]["processed_dir"] / (_symbol_to_stem(symbol) + ".csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            "Run data_pipeline.py first."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ── Zone quality filter ─────────────────────────────────────────────────────────

def filter_zones(
    zones_df: pd.DataFrame,
    min_strength: float = 0.0,
    max_strength: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Filter zones by strength_pit score before running backtest.

    Why strength filtering works
    ----------------------------
    Analysis of ^NSEI zones shows that only the 0.65–0.80 strength bucket
    produces a positive expected value (+₹10,183 total P&L, 33% WR).
    Weak zones (<0.5) and very-strong zones (>0.8) both lose money because:
      - Weak zones: price blows straight through, no institutional interest
      - Very-strong zones: already widely known → counter-traded, stop hunted

    The sweet spot is zones that are strong enough to attract buyers/sellers
    but not so obvious that they become crowded trades.

    Parameters
    ----------
    min_strength : minimum strength_pit score (default 0.0 = no filter)
    max_strength : maximum strength_pit score (default 1.0 = no filter)
    """
    if logger is None:
        logger = logging.getLogger("pnl_backtester")

    original = len(zones_df)

    if "strength_pit" not in zones_df.columns:
        logger.warning("strength_pit column not found in zones — skipping strength filter")
        return zones_df

    filtered = zones_df[
        zones_df["strength_pit"].between(min_strength, max_strength, inclusive="both")
    ].copy()

    removed = original - len(filtered)
    if removed > 0:
        logger.info(
            f"Strength filter [{min_strength:.2f}, {max_strength:.2f}]: "
            f"{len(filtered)} zones kept, {removed} removed"
        )

    return filtered


# ── Single-zone trade simulation ────────────────────────────────────────────────

def simulate_zone_trade_pnl(
    zone: pd.Series,
    price_df: pd.DataFrame,
    current_capital: float,
    risk_pct: float          = 0.01,
    rr_ratio: float          = 2.0,
    max_candles: int         = 5,
    slippage_pct: float      = 0.0,
    commission: float        = 0.0,
    confirm_entry: bool      = False,
    trail_atr_mult: float    = 0.0,
    stop_atr_mult: float     = 0.0,
    trail_trigger_r: float   = 2.0,
    floor_atr_mult: float    = 0.5,
) -> Optional[dict]:
    """
    Simulate one trade on a single zone and compute ₹ P&L.

    Parameters
    ----------
    zone            : one row from zones_df (must have proximal, distal, type)
    price_df        : full processed OHLCV DataFrame
    current_capital : current account size in ₹ (used for position sizing)
    risk_pct        : fraction of capital to risk per trade (default 0.01 = 1%)
    rr_ratio        : reward-to-risk multiple for take-profit (default 2.0).
                      This is the MINIMUM exit target. With trail enabled, the
                      trade continues beyond this level — the trail determines
                      the actual exit. Without trail, closes at exactly rr_ratio×R.
    max_candles     : max days to hold trade before forced exit
    slippage_pct    : entry price is worsened by this fraction (default 0 = no slippage)
    commission      : flat ₹ commission per trade (both entry and exit, default 0)
    confirm_entry   : if True, wait for a confirmation candle before entering.
                      Demand: a candle that closes ABOVE the proximal edge
                              (price rejected back up from demand zone).
                      Supply: a candle that closes BELOW the proximal edge
                              (price rejected back down from supply zone).
                      This eliminates "touched and immediately ran through"
                      entries which are 48% of all trades in baseline.
    trail_atr_mult  : ATR multiplier for trailing stop (default 0.0 = disabled).
                      Controls how far the stop trails behind the best price in Phase 2.
                      1.0 = tight (locks profit fast), 1.5 = balanced, 2.0 = loose.
                      Requires "ATR" column in price_df (computed by preprocessor.py).

    trail_trigger_r : R-multiple at which the trail activates (default 2.0).
                      When price reaches (entry + trail_trigger_r × risk), Phase 2 begins.

                      HOW IT WORKS:
                        Phase 1 — price hasn't hit trail_trigger_r yet:
                          Stop   : fixed at zone distal (or ATR-based if stop_atr_mult > 0)
                          Target : fixed at rr_ratio × R  (only used when trail OFF)
                          → Identical to no-trail behaviour

                        Phase 2 — price HITS trail_trigger_r × R (no candle skipped):
                          Trail ACTIVATES on the same candle that hit the trigger.
                          Stop trails (trail_atr_mult × ATR) behind best price.
                          Floor: active_stop >= trigger_price − floor_atr_mult × ATR
                          Trade runs freely. Exit when trail stop is hit.

                      trail_trigger_r examples (with rr_ratio=2.0):
                        2.0 → trail activates at 2R  (safe  — min win ≈ 2R − buffer)
                        1.5 → trail activates at 1.5R (moderate)
                        1.0 → trail activates at 1R   (aggressive)

    floor_atr_mult  : Controls how far below trigger_price the trail stop floor sits.
                      floor = trigger_price − floor_atr_mult × ATR
                      0.0 = no floor (pure ATR trail from activation, can dip far below trigger)
                      0.5 = floor is half-ATR below trigger (default — avoids instant exit)
                      1.0 = floor is one full ATR below trigger (more room on activation candle)
                      Higher = more tolerance; lower = tighter minimum exit guarantee.

    stop_atr_mult   : ATR multiplier for initial stop in Phase 1 (default 0.0).
                      0.0 = use zone distal as stop (original zone-based behaviour).
                      >0  = stop = entry ± stop_atr_mult × ATR at entry candle.

    Returns
    -------
    dict with full trade details including ₹ P&L, or None if price never entered zone.
    """
    formation_date = zone["formation_date"]
    zone_top       = float(zone["top"])
    zone_bottom    = float(zone["bottom"])
    zone_type      = zone["type"]           # "demand" or "supply"
    structure      = zone["structure"]      # DBR / RBD / RBR / DBD

    # Proximal = entry edge (closest to current price)
    # Distal   = stop edge  (invalidation level)
    entry_price_raw = float(zone["proximal"])
    stop_price      = float(zone["distal"])

    # Apply slippage: for long (demand), we get a slightly worse fill
    # For short (supply), the fill is slightly higher than the bid
    if zone_type == "demand":
        entry_price = entry_price_raw * (1 + slippage_pct)
    else:
        entry_price = entry_price_raw * (1 - slippage_pct)

    # ── Risk calculation ────────────────────────────────────────────────────
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return None  # Degenerate zone — distal == proximal

    risk_amount = current_capital * risk_pct          # ₹ risked on this trade

    # Position size: number of shares (fractional shares allowed for indices)
    shares = risk_amount / risk_per_share

    # ── Take-profit target ──────────────────────────────────────────────────
    if zone_type == "demand":
        target_price = entry_price + rr_ratio * risk_per_share
    else:
        target_price = entry_price - rr_ratio * risk_per_share

    # ── Entry logic ───────────────────────────────────────────────────────────
    #
    # Supply/Demand zone entry — two phase approach:
    #
    # Phase 1 — Zone touch:
    #   Demand (DBR/RBR): price drops INTO zone → Low < zone_top
    #   Supply (RBD/DBD): price rallies INTO zone → High > zone_bottom
    #   Once touched, we place a limit order at the proximal edge and wait.
    #
    # Phase 2 — Limit order fills at proximal:
    #   Demand: price reverses UP and High >= proximal → BUY at proximal
    #   Supply: price reverses DOWN and Low <= proximal → SELL at proximal
    #
    # If confirm_entry=True: require price to fully ENTER the zone first
    #   (close inside zone), then place limit at proximal.
    # If confirm_entry=False: place limit at proximal as soon as any part
    #   of the candle enters the zone (Low inside for demand).
    #
    # Zone invalidation: if price breaks through the DISTAL edge at any
    #   point before the limit fills → zone is dead, skip the trade.
    #
    # Example — Demand zone (DBR), proximal=19,600, distal=19,450:
    #   Day 3: Low=19,550 → zone touched, place limit BUY at 19,600
    #   Day 4: Low=19,520, High=19,650 → High >= 19,600 → FILLED at 19,600
    #   Day 4 Low (19,520) > distal (19,450) → zone still valid ✅
    #   Entry = 19,600, Stop = 19,450, Target = 19,600 + 3×150 = 20,050

    future = price_df[price_df["Date"] > formation_date].reset_index(drop=True)
    if future.empty:
        return None

    entry_idx    = None
    zone_touched = False  # True once price has entered the zone

    for i, row in future.iterrows():
        low   = float(row["Low"])
        high  = float(row["High"])
        close = float(row["Close"])

        # ── Check zone invalidation first ──────────────────────────────────
        # If price breaks through the distal (stop level) before entry, zone dead
        if zone_type == "demand" and low < stop_price:
            return None  # Broke below demand zone distal
        if zone_type == "supply" and high > stop_price:
            return None  # Broke above supply zone distal

        if not zone_touched:
            # ── Phase 1: detect zone touch ─────────────────────────────────
            if zone_type == "demand":
                # Demand: price enters zone when Low drops below zone top
                # If confirm_entry=True: require Close to be inside zone (stronger signal)
                if confirm_entry:
                    touched = (zone_bottom <= close <= zone_top)
                else:
                    touched = (low <= zone_top)
            else:
                # Supply: price enters zone when High rises above zone bottom
                if confirm_entry:
                    touched = (zone_bottom <= close <= zone_top)
                else:
                    touched = (high >= zone_bottom)

            if touched:
                zone_touched = True
                # On the same candle as touch, check if limit already fills
                # (price entered zone AND hit proximal in the same bar)
                if zone_type == "demand" and high >= entry_price_raw:
                    entry_idx = i
                    break
                if zone_type == "supply" and low <= entry_price_raw:
                    entry_idx = i
                    break

        else:
            # ── Phase 2: limit order waiting at proximal ───────────────────
            # Price is now inside zone. Wait for it to hit the proximal edge.
            if zone_type == "demand" and high >= entry_price_raw:
                # Price reversed up and touched zone TOP → BUY at proximal
                entry_idx = i
                break
            if zone_type == "supply" and low <= entry_price_raw:
                # Price reversed down and touched zone BOTTOM → SELL at proximal
                entry_idx = i
                break

    if entry_idx is None:
        return None  # Price never triggered the limit order

    entry_candle = future.iloc[entry_idx]
    entry_date   = entry_candle["Date"]

    # ── ATR-based initial stop override (stop_atr_mult > 0) ──────────────────
    # When stop_atr_mult > 0, replace zone distal with an ATR-based stop.
    # This makes the initial stop proportional to current volatility at entry
    # rather than tied to a historical zone boundary.
    # Note: zone distal is still used for invalidation checks BEFORE entry (above).
    if stop_atr_mult > 0 and "ATR" in price_df.columns:
        atr_at_entry = float(entry_candle.get("ATR", 0) or 0)
        if atr_at_entry > 0:
            if zone_type == "demand":
                stop_price = entry_price - stop_atr_mult * atr_at_entry
            else:
                stop_price = entry_price + stop_atr_mult * atr_at_entry
            # Recompute all risk-dependent values with the new stop
            risk_per_share = abs(entry_price - stop_price)
            if risk_per_share <= 0:
                return None
            shares = risk_amount / risk_per_share
            if zone_type == "demand":
                target_price = entry_price + rr_ratio * risk_per_share
            else:
                target_price = entry_price - rr_ratio * risk_per_share

    # ── Trail activation price (trigger_price) ────────────────────────────────
    # trigger_price = the price level at which the trail turns on.
    # Default: trail_trigger_r = rr_ratio (e.g. 2.0) → trail activates at 2R target.
    # Custom : trail_trigger_r = 1.0 → trail activates at 1R (before full target).
    # The floor on the trail stop = trigger_price (min exit when trail active).
    if zone_type == "demand":
        trigger_price = entry_price + trail_trigger_r * risk_per_share
    else:
        trigger_price = entry_price - trail_trigger_r * risk_per_share

    # ── Walk forward: check stop and target ───────────────────────────────────
    outcome      = "open"
    exit_price   = None
    exit_date    = None
    hold_candles = 0
    post_entry   = future.iloc[entry_idx + 1:].reset_index(drop=True)

    # ── Trailing stop state ───────────────────────────────────────────────────
    #
    # PHASE 1 — before trail activates (price hasn't hit trigger_price yet):
    #   Stop   = fixed at zone distal (or ATR-based if stop_atr_mult > 0)
    #   Target = fixed at rr_ratio × R  (only used when trail is OFF)
    #   Trail OFF: check target_price → close as win
    #   Trail ON:  check trigger_price → activate trail instead of closing
    #
    # PHASE 2 — trail is active:
    #   Stop   = trails (trail_atr_mult × ATR) behind best price (only tightens)
    #   Floor  = trigger_price (trail stop never goes below activation level)
    #   Target = removed — trade runs freely beyond trigger level
    #   Exit   = trail stop hit → outcome = "trail_win"

    use_trail    = trail_atr_mult > 0 and "ATR" in price_df.columns
    trail_active = False
    active_stop  = stop_price   # starts at zone distal (or ATR override); tightens in trail
    best_price   = entry_price  # best price seen since trail activated

    for i, row in post_entry.iterrows():
        if hold_candles >= max_candles:
            outcome    = "timeout"
            exit_price = float(row["Close"])
            exit_date  = row["Date"]
            break

        high = float(row["High"])
        low  = float(row["Low"])

        # ── Phase 1: check for exit or trail activation ───────────────────
        if not trail_active:
            if zone_type == "demand":
                # Trail ON  → check trigger_price for activation
                # Trail OFF → check target_price for normal close
                check_price   = trigger_price if use_trail else target_price
                price_reached = high >= check_price
                stop_hit      = low  <= active_stop
            else:
                check_price   = trigger_price if use_trail else target_price
                price_reached = low  <= check_price
                stop_hit      = high >= active_stop

            if stop_hit:
                # Conservative intrabar resolution: if the stop and the target
                # both fall inside this candle's range, we cannot know which was
                # touched first, so we assume the stop was hit first (worst-case).
                # This matches the methodology stated in the module docstring and
                # removes the look-ahead optimism of booking such candles as wins.
                outcome    = "loss"
                exit_price = active_stop
                exit_date  = row["Date"]
                break

            if price_reached:
                if not use_trail:
                    # Trail disabled: close as normal win at fixed target
                    outcome    = "win"
                    exit_price = target_price
                    exit_date  = row["Date"]
                    break
                else:
                    # Trail enabled: activate trail from trigger level, keep riding.
                    # No candle skip — trail stop is checked on this same candle.
                    trail_active = True
                    best_price   = high if zone_type == "demand" else low
                    atr = float(row.get("ATR", 0) or 0)
                    if atr > 0:
                        trail_dist = trail_atr_mult * atr
                        if zone_type == "demand":
                            candidate = best_price - trail_dist
                        else:
                            candidate = best_price + trail_dist
                        if zone_type == "demand" and candidate > active_stop:
                            active_stop = candidate
                        elif zone_type == "supply" and candidate < active_stop:
                            active_stop = candidate

                    # ── FLOOR: trigger_price − floor_atr_mult × ATR ───────────
                    # Prevents active_stop from sitting too far below trigger on
                    # a large-ATR activation candle. Using trigger_price as the
                    # floor (old approach) caused immediate exits because the
                    # activation candle's Low is almost always < trigger_price.
                    # floor_atr_mult gives a buffer: e.g. 0.5 × ATR below trigger
                    # means the candle must retrace half an ATR from the trigger
                    # level before the trail fires — realistic, not artificial.
                    if atr > 0 and floor_atr_mult >= 0:
                        floor_dist = floor_atr_mult * atr
                        if zone_type == "demand":
                            floor_price = trigger_price - floor_dist
                            active_stop = max(active_stop, floor_price)
                        else:
                            floor_price = trigger_price + floor_dist
                            active_stop = min(active_stop, floor_price)

                    # Check trail stop on the activation candle itself (real trading
                    # behaviour — no candle skip). If price ran up to trigger then
                    # immediately reversed by more than floor_dist, we exit here.
                    if zone_type == "demand" and low <= active_stop:
                        outcome    = "trail_win"
                        exit_price = active_stop
                        exit_date  = row["Date"]
                        hold_candles += 1
                        break
                    if zone_type == "supply" and high >= active_stop:
                        outcome    = "trail_win"
                        exit_price = active_stop
                        exit_date  = row["Date"]
                        hold_candles += 1
                        break

        # ── Phase 2: trail is active — update stop and check exit ─────────
        if trail_active:
            atr = float(row.get("ATR", 0) or 0)
            if atr > 0:
                trail_dist = trail_atr_mult * atr
                if zone_type == "demand":
                    best_price  = max(best_price, high)
                    candidate   = best_price - trail_dist
                    if candidate > active_stop:     # only tighten, never widen
                        active_stop = candidate
                else:
                    best_price  = min(best_price, low)
                    candidate   = best_price + trail_dist
                    if candidate < active_stop:     # only tighten, never widen
                        active_stop = candidate

            # Check if trail stop was hit
            if zone_type == "demand" and low <= active_stop:
                outcome    = "trail_win"
                exit_price = active_stop
                exit_date  = row["Date"]
                break
            if zone_type == "supply" and high >= active_stop:
                outcome    = "trail_win"
                exit_price = active_stop
                exit_date  = row["Date"]
                break

        hold_candles += 1

    # If still open (no future candles at all)
    if exit_price is None:
        return None

    # ── Compute ₹ P&L ────────────────────────────────────────────────────────
    if zone_type == "demand":
        gross_pnl = (exit_price - entry_price) * shares
        r_multiple = (exit_price - entry_price) / risk_per_share
    else:
        gross_pnl  = (entry_price - exit_price) * shares
        r_multiple = (entry_price - exit_price) / risk_per_share

    # Subtract commission (entry + exit)
    net_pnl = gross_pnl - (2 * commission)

    # Position value (how much capital was deployed)
    position_value = shares * entry_price

    return {
        # Zone identifiers
        "zone_id":         zone.get("zone_id", ""),
        "zone_type":       zone_type,
        "structure":       structure,
        "zone_top":        round(zone_top, 2),
        "zone_bottom":     round(zone_bottom, 2),
        # Trade dates
        "formation_date":  str(formation_date.date()),
        "entry_date":      str(entry_date.date()),
        "exit_date":       str(exit_date.date()) if exit_date is not None else None,
        # Prices
        "entry_price":     round(entry_price, 2),
        "stop_price":      round(stop_price, 2),
        "target_price":    round(target_price, 2),
        "exit_price":      round(exit_price, 2),
        # Position sizing
        "capital_at_entry": round(current_capital, 2),
        "risk_amount_inr":  round(risk_amount, 2),
        "shares":           round(shares, 4),
        "position_value":   round(position_value, 2),
        "risk_per_share":   round(risk_per_share, 2),
        # P&L
        "gross_pnl_inr":   round(gross_pnl, 2),
        "commission_inr":  round(2 * commission, 2),
        "net_pnl_inr":     round(net_pnl, 2),
        "r_multiple":      round(r_multiple, 4),
        "outcome":         outcome,
        "hold_candles":    hold_candles,
        # Trailing stop context
        "trail_used":      trail_active,
        "trail_exit_stop": round(active_stop, 2) if trail_active else None,
        "trail_best_price": round(best_price, 2) if trail_active else None,
        # Zone quality context
        "zone_strength":   float(zone.get("strength_pit", zone.get("strength", float("nan")))),
    }


# ── Full backtest engine ────────────────────────────────────────────────────────

class PnLBacktester:
    """
    Runs a full P&L backtest over all detected zones.

    Parameters
    ----------
    cfg               : loaded config.yaml dict
    start_capital     : starting account size in ₹ (default 500,000)
    risk_pct          : fraction of capital to risk per trade (default 0.01 = 1%)
    rr_ratio          : take-profit in R multiples (default 2.0)
    max_hold_candles  : max days to stay in a trade (default 5)
    slippage_pct      : entry slippage as fraction (default 0.0)
    commission        : flat ₹ brokerage per trade entry+exit (default 0)
    confirm_entry     : wait for rejection candle before entering (default False)
    min_strength      : only trade zones with strength_pit >= this (default 0.0)
    max_strength      : only trade zones with strength_pit <= this (default 1.0)
    trade_structures  : list of zone structures to trade.
                        Default ["DBR", "RBD", "RBR", "DBD"] (all types).
                        Set to ["DBR", "RBD"] to trade only reversal zones.
                        DBR (Drop-Base-Rally) = demand reversal zone
                        RBD (Rally-Base-Drop) = supply reversal zone
                        RBR (Rally-Base-Rally) = demand continuation zone  ← skipped if excluded
                        DBD (Drop-Base-Drop)   = supply continuation zone  ← skipped if excluded
    trail_atr_mult    : ATR multiplier for trailing stop (default 0.0 = disabled).
                        When > 0, trail activates when price hits trail_trigger_r × R.
                        Stop then trails (trail_atr_mult × ATR) behind best price.
    trail_trigger_r   : R-multiple at which trail activates (default 2.0).
    floor_atr_mult    : Floor buffer below trigger_price (default 0.5).
                        floor = trigger_price − floor_atr_mult × ATR.
                        0.5 = half-ATR buffer (avoids instant exit on activation candle).
    stop_atr_mult     : ATR multiplier for initial stop in Phase 1 (default 0.0).
                        0.0 = zone distal. >0 = entry ± stop_atr_mult × ATR.
    logger            : optional logger
    """

    # All valid zone structures
    ALL_STRUCTURES = ["DBR", "RBD", "RBR", "DBD"]

    def __init__(
        self,
        cfg: dict,
        start_capital: float    = 500_000.0,
        risk_pct: float         = 0.01,
        rr_ratio: float         = 2.0,
        max_hold_candles: int   = 5,
        slippage_pct: float     = 0.0,
        commission: float       = 0.0,
        confirm_entry: bool     = False,
        min_strength: float     = 0.0,
        max_strength: float     = 1.0,
        trade_structures: Optional[list] = None,
        trail_atr_mult: float   = 0.0,
        trail_trigger_r: float  = 2.0,
        floor_atr_mult: float   = 0.5,
        stop_atr_mult: float    = 0.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg              = cfg
        self.start_capital    = start_capital
        self.risk_pct         = risk_pct
        self.rr_ratio         = rr_ratio
        self.max_hold_candles = max_hold_candles
        self.slippage_pct     = slippage_pct
        self.commission       = commission
        self.confirm_entry    = confirm_entry
        self.min_strength     = min_strength
        self.max_strength     = max_strength
        self.trade_structures = [s.upper() for s in trade_structures] if trade_structures else self.ALL_STRUCTURES
        self.trail_atr_mult   = trail_atr_mult
        self.trail_trigger_r  = trail_trigger_r
        self.floor_atr_mult   = floor_atr_mult
        self.stop_atr_mult    = stop_atr_mult
        self.logger           = logger or logging.getLogger("pnl_backtester")

    def run(
        self,
        symbol: str,
        zones_df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate all zone trades chronologically.
        Capital is updated after every closed trade.

        Parameters
        ----------
        symbol   : ticker string (for logging only)
        zones_df : zones from zone_detector
        price_df : preprocessed OHLCV data

        Returns
        -------
        pd.DataFrame — one row per trade that was entered
        """
        confirm_str = "ON (wait for rejection candle)" if self.confirm_entry else "OFF"
        trail_str   = (
            f"{self.trail_atr_mult}×ATR | trigger={self.trail_trigger_r}R"
            f" | floor={self.floor_atr_mult}×ATR below trigger"
            + (f" | init_stop={self.stop_atr_mult}×ATR" if self.stop_atr_mult > 0 else "")
            if self.trail_atr_mult > 0 else "OFF (fixed zone stop)"
        )
        skipped_structures = [s for s in self.ALL_STRUCTURES if s not in self.trade_structures]
        self.logger.info(
            f"PnL Backtest — {symbol} | "
            f"capital=₹{self.start_capital:,.0f} | "
            f"risk={self.risk_pct:.1%}/trade | "
            f"RR={self.rr_ratio} | "
            f"max_hold={self.max_hold_candles}d | "
            f"confirm={confirm_str} | "
            f"trail={trail_str} | "
            f"strength=[{self.min_strength},{self.max_strength}]"
        )
        self.logger.info(
            f"Trade structures : {', '.join(self.trade_structures)}"
            + (f" | Skipping : {', '.join(skipped_structures)}" if skipped_structures else "")
        )

        # Apply zone quality filter before running trades
        zones_filtered = filter_zones(
            zones_df,
            min_strength = self.min_strength,
            max_strength = self.max_strength,
            logger       = self.logger,
        )

        # Sort by formation date so trades happen in time order
        zones_sorted = zones_filtered.sort_values("formation_date").reset_index(drop=True)

        current_capital  = self.start_capital
        trades           = []
        no_touch         = 0
        skipped_struct   = 0

        for _, zone in zones_sorted.iterrows():
            # ── Structure filter (no look-ahead — structure is known at zone formation) ──
            zone_structure = str(zone.get("structure", "")).upper()
            if zone_structure not in self.trade_structures:
                skipped_struct += 1
                continue

            result = simulate_zone_trade_pnl(
                zone             = zone,
                price_df         = price_df,
                current_capital  = current_capital,
                risk_pct         = self.risk_pct,
                rr_ratio         = self.rr_ratio,
                max_candles      = self.max_hold_candles,
                slippage_pct     = self.slippage_pct,
                commission       = self.commission,
                confirm_entry    = self.confirm_entry,
                trail_atr_mult   = self.trail_atr_mult,
                trail_trigger_r  = self.trail_trigger_r,
                floor_atr_mult   = self.floor_atr_mult,
                stop_atr_mult    = self.stop_atr_mult,
            )

            if result is None:
                no_touch += 1
                continue

            # Update running capital after every closed trade
            current_capital += result["net_pnl_inr"]
            current_capital  = max(current_capital, 0.0)   # floor at zero (can't go negative)

            result["equity_after_trade"] = round(current_capital, 2)
            trades.append(result)

        self.logger.info(
            f"Zones: {len(zones_sorted)} total | "
            f"{skipped_struct} skipped (structure filter) | "
            f"{len(trades)} traded | {no_touch} never touched"
        )

        if not trades:
            self.logger.warning("No trades were generated.")
            return pd.DataFrame()

        return pd.DataFrame(trades)

    def calculate_metrics(
        self,
        trades_df: pd.DataFrame,
    ) -> dict:
        """
        Compute full set of P&L and trading performance metrics.

        Separates closed trades (win/loss) from timeouts and open trades.
        All ₹ metrics are computed on net_pnl_inr (after commission).
        """
        if trades_df.empty:
            return {}

        # Separate outcomes
        # win        = price hit fixed target (rr_ratio×R), trail NOT active
        # trail_win  = trail activated at target, extended further, then trail stop hit
        # loss       = zone distal stop hit before target (thesis invalidation)
        # timeout    = max_hold_candles reached
        closed  = trades_df[trades_df["outcome"].isin(["win", "loss", "trail_win"])].copy()
        timeout = trades_df[trades_df["outcome"] == "timeout"].copy()
        open_t  = trades_df[trades_df["outcome"] == "open"].copy()
        all_settled = trades_df[trades_df["outcome"] != "open"].copy()  # closed + timeout

        wins       = closed[closed["outcome"].isin(["win", "trail_win"])]
        clean_wins = closed[closed["outcome"] == "win"]
        trail_wins = closed[closed["outcome"] == "trail_win"]
        losses     = closed[closed["outcome"] == "loss"]

        n_total      = len(trades_df)
        n_closed     = len(closed)
        n_wins       = len(wins)
        n_clean_wins = len(clean_wins)
        n_trail_wins = len(trail_wins)
        n_losses     = len(losses)
        n_timeout    = len(timeout)
        n_open       = len(open_t)

        win_rate = n_wins / n_closed if n_closed > 0 else 0.0

        # ── ₹ P&L metrics ────────────────────────────────────────────────────
        total_pnl_inr     = float(all_settled["net_pnl_inr"].sum())
        total_gross_win   = float(wins["net_pnl_inr"].sum())        if n_wins        > 0 else 0.0
        total_gross_loss  = float(losses["net_pnl_inr"].sum())      if n_losses      > 0 else 0.0
        trail_win_pnl     = float(trail_wins["net_pnl_inr"].sum())  if n_trail_wins  > 0 else 0.0
        avg_win_inr       = float(wins["net_pnl_inr"].mean())       if n_wins        > 0 else 0.0
        avg_loss_inr      = float(losses["net_pnl_inr"].mean())     if n_losses      > 0 else 0.0
        avg_trail_win_inr = float(trail_wins["net_pnl_inr"].mean()) if n_trail_wins  > 0 else 0.0

        profit_factor = (
            abs(total_gross_win) / abs(total_gross_loss)
            if total_gross_loss != 0 else float("inf")
        )

        # ── Equity curve metrics ──────────────────────────────────────────────
        final_capital  = float(trades_df["equity_after_trade"].iloc[-1]) if n_total > 0 else self.start_capital
        total_return   = final_capital - self.start_capital
        total_return_pct = (total_return / self.start_capital) * 100.0

        # Max drawdown in ₹ (peak-to-trough on equity curve)
        equity_curve = trades_df["equity_after_trade"].values
        peak         = np.maximum.accumulate(equity_curve)
        drawdowns_inr = peak - equity_curve
        max_drawdown_inr = float(drawdowns_inr.max()) if len(drawdowns_inr) > 0 else 0.0
        max_drawdown_pct = (max_drawdown_inr / float(peak.max())) * 100.0 if peak.max() > 0 else 0.0

        # ── R-multiple metrics (for comparison) ──────────────────────────────
        r_values    = closed["r_multiple"].dropna().values
        total_r     = float(r_values.sum())       if len(r_values) > 0 else 0.0
        avg_r       = float(r_values.mean())      if len(r_values) > 0 else 0.0
        gross_win_r = float(wins["r_multiple"].sum())   if n_wins   > 0 else 0.0
        gross_los_r = float(losses["r_multiple"].sum()) if n_losses > 0 else 0.0

        # ── Sharpe (R-based, annualised) ─────────────────────────────────────
        avg_hold   = closed["hold_candles"].mean() if n_closed > 0 else self.max_hold_candles
        trades_per_year = 252 / max(float(avg_hold), 1.0)
        if len(r_values) > 1 and float(np.std(r_values)) > 0:
            sharpe_r = float(np.mean(r_values) / np.std(r_values) * math.sqrt(trades_per_year))
        else:
            sharpe_r = float("nan")

        # ── Per-structure breakdown ───────────────────────────────────────────
        structure_stats: dict = {}
        for struct in closed["structure"].dropna().unique():
            sub      = closed[closed["structure"] == struct]
            sub_wins = int((sub["outcome"] == "win").sum())
            structure_stats[struct] = {
                "trades":       len(sub),
                "wins":         sub_wins,
                "win_rate":     round(sub_wins / len(sub), 3),
                "total_pnl":    round(float(sub["net_pnl_inr"].sum()), 2),
                "avg_pnl":      round(float(sub["net_pnl_inr"].mean()), 2),
                "avg_r":        round(float(sub["r_multiple"].mean()), 3),
            }

        # ── Timeout summary ───────────────────────────────────────────────────
        timeout_pnl = round(float(timeout["net_pnl_inr"].sum()), 2) if n_timeout > 0 else 0.0

        return {
            # Capital
            "start_capital":       self.start_capital,
            "final_capital":       round(final_capital, 2),
            "total_return_inr":    round(total_return, 2),
            "total_return_pct":    round(total_return_pct, 4),
            # Trade counts
            "total_trades":        n_total,
            "closed_trades":       n_closed,
            "wins":                n_wins,
            "clean_wins":          n_clean_wins,
            "trail_wins":          n_trail_wins,
            "losses":              n_losses,
            "timeout_trades":      n_timeout,
            "open_trades":         n_open,
            # Win/loss stats
            "win_rate":            round(win_rate, 4),
            "profit_factor":       round(profit_factor, 4),
            "avg_win_inr":         round(avg_win_inr, 2),
            "avg_loss_inr":        round(avg_loss_inr, 2),
            "avg_trail_win_inr":   round(avg_trail_win_inr, 2),
            "trail_win_pnl_inr":   round(trail_win_pnl, 2),
            # P&L ₹
            "total_pnl_inr":       round(total_pnl_inr, 2),
            "gross_profit_inr":    round(total_gross_win, 2),
            "gross_loss_inr":      round(total_gross_loss, 2),
            "timeout_pnl_inr":     timeout_pnl,
            # Drawdown
            "max_drawdown_inr":    round(max_drawdown_inr, 2),
            "max_drawdown_pct":    round(max_drawdown_pct, 4),
            # R-multiple equivalents
            "total_r":             round(total_r, 4),
            "avg_r":               round(avg_r, 4),
            "sharpe_r":            round(sharpe_r, 4) if not math.isnan(sharpe_r) else None,
            "avg_hold_candles":    round(float(closed["hold_candles"].mean()), 1) if n_closed > 0 else None,
            # Per-structure
            "by_structure":        structure_stats,
            # Parameters used
            "params": {
                "risk_pct":          self.risk_pct,
                "rr_ratio":          self.rr_ratio,
                "max_hold_candles":  self.max_hold_candles,
                "slippage_pct":      self.slippage_pct,
                "commission":        self.commission,
                "confirm_entry":     self.confirm_entry,
                "min_strength":      self.min_strength,
                "max_strength":      self.max_strength,
                "trade_structures":  self.trade_structures,
                "trail_atr_mult":    self.trail_atr_mult,
            },
        }

    def print_report(self, metrics: dict, symbol: str) -> str:
        """Format all metrics into a readable ₹ P&L report."""
        sep  = "=" * 65
        sep2 = "-" * 65
        p = metrics["params"]
        confirm_str  = "YES" if p.get("confirm_entry") else "NO"
        strength_str = f"[{p.get('min_strength', 0):.2f} – {p.get('max_strength', 1):.2f}]"
        trail_mult = p.get("trail_atr_mult", 0.0)
        trail_str  = (
            f"{trail_mult}×ATR (activates when {p.get('rr_ratio', 2.0)}R target is hit)"
            if trail_mult > 0 else "OFF"
        )

        struct_str = ", ".join(p.get("trade_structures", ["DBR", "RBD", "RBR", "DBD"]))
        lines = [
            sep,
            f"  P&L Backtest Report  |  {symbol}",
            f"  Strategy : Zone Supply & Demand  |  RR {p['rr_ratio']}:1",
            f"  Max hold : {p['max_hold_candles']} candles  "
            f"|  Risk per trade : {p['risk_pct']:.1%}",
            f"  Confirm entry : {confirm_str}  "
            f"|  Strength filter : {strength_str}",
            f"  Trailing stop : {trail_str}  "
            f"|  Structures : {struct_str}",
            f"  Slippage : {p['slippage_pct']:.3%}  "
            f"|  Commission : ₹{p['commission']:.0f}/side",
            sep,
            "",
            "── Capital ─────────────────────────────────────────────────",
            f"  Starting Capital     :  ₹{metrics['start_capital']:>12,.2f}",
            f"  Ending Capital       :  ₹{metrics['final_capital']:>12,.2f}",
            f"  Total Return (₹)     :  ₹{metrics['total_return_inr']:>+12,.2f}",
            f"  Total Return (%)     :  {metrics['total_return_pct']:>+11.2f}%",
            "",
            "── P&L Breakdown ───────────────────────────────────────────",
            f"  Gross Profit         :  ₹{metrics['gross_profit_inr']:>+12,.2f}",
            f"  Gross Loss           :  ₹{metrics['gross_loss_inr']:>+12,.2f}",
            f"  Net P&L (closed)     :  ₹{metrics['total_pnl_inr']:>+12,.2f}",
            f"  Timeout P&L          :  ₹{metrics['timeout_pnl_inr']:>+12,.2f}",
            "",
            "── Trade Statistics ─────────────────────────────────────────",
            f"  Total trades entered :  {metrics['total_trades']}",
            f"  Closed               :  {metrics['closed_trades']}",
            f"    Clean wins (2R)    :  {metrics['clean_wins']}",
            f"    Trail wins (>2R)   :  {metrics['trail_wins']}",
            f"    Losses (zone stop) :  {metrics['losses']}",
            f"  Timeout exits        :  {metrics['timeout_trades']}",
            f"  Open (never closed)  :  {metrics['open_trades']}",
            "",
            f"  Win Rate             :  {metrics['win_rate']:.1%}",
            f"  Profit Factor        :  {metrics['profit_factor']:.3f}",
            f"  Avg Win  (₹)         :  ₹{metrics['avg_win_inr']:>+10,.2f}",
            f"  Avg Trail Win (₹)    :  ₹{metrics['avg_trail_win_inr']:>+10,.2f}",
            f"  Avg Loss (₹)         :  ₹{metrics['avg_loss_inr']:>+10,.2f}",
            f"  Avg Hold (days)      :  {metrics['avg_hold_candles']}",
            "",
            "── Risk Metrics ─────────────────────────────────────────────",
            f"  Max Drawdown (₹)     :  ₹{metrics['max_drawdown_inr']:>12,.2f}",
            f"  Max Drawdown (%)     :  {metrics['max_drawdown_pct']:>11.2f}%",
            f"  Total R              :  {metrics['total_r']:>+10.2f}R",
            f"  Avg R / trade        :  {metrics['avg_r']:>+10.3f}R",
            f"  Sharpe (R-based)     :  {metrics['sharpe_r'] if metrics['sharpe_r'] is not None else 'N/A'}",
        ]

        # Per-structure breakdown
        by_struct = metrics.get("by_structure", {})
        if by_struct:
            lines.append("")
            lines.append("── By Zone Structure ────────────────────────────────────────")
            lines.append(
                f"  {'Structure':<10} {'Trades':>7} {'WR':>7} {'Total P&L':>14} "
                f"{'Avg P&L':>12} {'Avg R':>8}"
            )
            lines.append("  " + sep2)
            for struct, s in sorted(by_struct.items()):
                lines.append(
                    f"  {struct:<10} {s['trades']:>7}  "
                    f"{s['win_rate']:>6.0%}  "
                    f"₹{s['total_pnl']:>+12,.0f}  "
                    f"₹{s['avg_pnl']:>+9,.0f}  "
                    f"{s['avg_r']:>+7.3f}R"
                )

        lines += ["", sep]
        return "\n".join(lines)

    def save_results(
        self,
        trades_df: pd.DataFrame,
        metrics: dict,
        report_str: str,
        symbol: str,
    ) -> tuple[Path, Path, Path]:
        """Save trades CSV, equity curve CSV, and text report."""
        bt_dir = PROJECT_ROOT / self.cfg.get("backtest", {}).get("output_dir", "data/backtest")
        bt_dir.mkdir(parents=True, exist_ok=True)
        stem = _symbol_to_stem(symbol)

        # Trades
        trades_path = bt_dir / f"{stem}_pnl_trades.csv"
        trades_df.to_csv(trades_path, index=False)

        # Equity curve
        equity_cols  = ["entry_date", "exit_date", "outcome", "net_pnl_inr", "equity_after_trade"]
        equity_df    = trades_df[[c for c in equity_cols if c in trades_df.columns]].copy()
        equity_path  = bt_dir / f"{stem}_equity_curve.csv"
        equity_df.to_csv(equity_path, index=False)

        # Report
        report_path = bt_dir / f"{stem}_pnl_report.txt"
        report_path.write_text(report_str, encoding="utf-8")

        self.logger.info(f"Trades saved     → {trades_path.relative_to(PROJECT_ROOT)}")
        self.logger.info(f"Equity curve     → {equity_path.relative_to(PROJECT_ROOT)}")
        self.logger.info(f"Report saved     → {report_path.relative_to(PROJECT_ROOT)}")
        return trades_path, equity_path, report_path


# ── Pipeline entry point ────────────────────────────────────────────────────────

def run_pnl_backtest(
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
    start_capital: float    = 500_000.0,
    risk_pct: float         = 0.01,
    rr_ratio: float         = 2.0,
    max_hold_candles: int   = 5,
    slippage_pct: float     = 0.0,
    commission: float       = 0.0,
    confirm_entry: bool     = False,
    min_strength: float     = 0.0,
    max_strength: float     = 1.0,
    trade_structures: Optional[list] = None,
    trail_atr_mult: float   = 0.0,
    trail_trigger_r: float  = 2.0,
    floor_atr_mult: float   = 0.5,
    stop_atr_mult: float    = 0.0,
    use_ml_zones: bool      = True,
    zones_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Main entry point. Can be called from zone_pipeline.py or standalone.

    Returns
    -------
    (trades_df, metrics)  — DataFrame of all trades + metrics dict
    """
    if zones_df is None:
        zones_df = load_zones(symbol, cfg, use_ml_zones=use_ml_zones)
    price_df = load_processed(symbol, cfg)

    zone_source = "ML-detected" if use_ml_zones else "Rule-based (algo)"
    logger.info(f"Zone source: {zone_source} | {len(zones_df)} zones loaded")

    bt = PnLBacktester(
        cfg               = cfg,
        start_capital     = start_capital,
        risk_pct          = risk_pct,
        rr_ratio          = rr_ratio,
        max_hold_candles  = max_hold_candles,
        slippage_pct      = slippage_pct,
        commission        = commission,
        confirm_entry     = confirm_entry,
        min_strength      = min_strength,
        max_strength      = max_strength,
        trade_structures  = trade_structures,
        trail_atr_mult    = trail_atr_mult,
        trail_trigger_r   = trail_trigger_r,
        floor_atr_mult    = floor_atr_mult,
        stop_atr_mult     = stop_atr_mult,
        logger            = logger,
    )

    trades_df  = bt.run(symbol, zones_df, price_df)

    if trades_df.empty:
        logger.warning("No trades generated — nothing to report.")
        return trades_df, {}

    metrics    = bt.calculate_metrics(trades_df)
    report_str = bt.print_report(metrics, symbol)

    # Print to console
    print("\n" + report_str)

    bt.save_results(trades_df, metrics, report_str, symbol)

    return trades_df, metrics


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run P&L backtest on supply/demand zones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Ticker symbol. Defaults to symbol in config.yaml.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=500_000.0,
        help="Starting capital in ₹ (e.g. 500000 = ₹5 Lakh)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=0.01,
        help="Fraction of capital to risk per trade (e.g. 0.01 = 1%%)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=2.0,
        help="Reward-to-risk ratio for take-profit",
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=5,
        help="Maximum number of candles (days) to hold a trade",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Entry slippage as fraction (e.g. 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Flat ₹ commission per trade side (e.g. 20 = ₹20 per entry+exit)",
    )
    parser.add_argument(
        "--trail-atr-mult",
        type=float,
        default=0.0,
        help=(
            "ATR multiplier for trailing stop (default 0.0 = disabled). "
            "When > 0, stop trails (N × ATR) behind best price since entry. "
            "Recommended range: 1.0–2.0. Example: --trail-atr-mult 1.5"
        ),
    )

    args   = parser.parse_args()
    cfg    = load_config()
    logger = setup_logging(cfg)
    symbol = args.symbol or cfg["data"]["symbol"]

    logger.info(f"=== PnL Backtester — {symbol} ===")

    run_pnl_backtest(
        symbol           = symbol,
        cfg              = cfg,
        logger           = logger,
        start_capital    = args.capital,
        risk_pct         = args.risk_pct,
        rr_ratio         = args.rr,
        max_hold_candles = args.max_hold,
        slippage_pct     = args.slippage,
        commission       = args.commission,
        trail_atr_mult   = args.trail_atr_mult,
    )


if __name__ == "__main__":
    main()
