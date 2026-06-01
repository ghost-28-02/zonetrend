"""
swing_zones.py
==============
Detects Support and Resistance zones using the Swing High / Swing Low method.

All parameters are read from config/zone_config.yaml (swing section).
Do not hardcode any threshold here — change zone_config.yaml instead.

Theory
------
A swing high is a candle whose HIGH is greater than the HIGH of every candle
within N candles before it AND N candles after it. It is a local peak.

A swing low is a candle whose LOW is less than the LOW of every candle within
N candles before it AND N candles after it. It is a local trough.

Zone Boundaries
---------------
We use the BODY edge for one boundary and the WICK extreme for the other,
creating a zone band rather than a single price line.

    Resistance zone (from swing high):
        upper_boundary = high[i]           ← wick top  (extreme rejection)
        lower_boundary = max(open, close)  ← body top  (where price settled)

    Support zone (from swing low):
        upper_boundary = min(open, close)  ← body bottom (where price settled)
        lower_boundary = low[i]            ← wick bottom (extreme rejection)

Config parameters used (from zone_config.yaml → swing section):
    lookback           : candles to check on each side of the swing candle
    min_body_atr_ratio : minimum body size to accept a candle as zone origin
    save_zones         : whether to persist zones to disk after detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Lazy config loader (avoids circular import at module load time) ──
def _get_cfg():
    from src.zones import ZoneConfig
    return ZoneConfig()


# ─────────────────────────────────────────────────────────────
# Core detection helpers
# ─────────────────────────────────────────────────────────────

def detect_swing_highs(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Returns a boolean Series — True at every swing high candle.

    A swing high at index i:
        high[i] > max(high[i-lookback : i])      (left side)
        high[i] > max(high[i+1 : i+lookback+1])  (right side)

    Uses strictly-greater-than to handle flat tops:
    if two adjacent candles share the same high, neither qualifies.
    """
    highs = df["High"].values
    n = len(highs)
    is_swing_high = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        if highs[i] > highs[i - lookback : i].max() and \
           highs[i] > highs[i + 1 : i + lookback + 1].max():
            is_swing_high[i] = True

    return pd.Series(is_swing_high, index=df.index, name="is_swing_high")


def detect_swing_lows(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Returns a boolean Series — True at every swing low candle.

    A swing low at index i:
        low[i] < min(low[i-lookback : i])      (left side)
        low[i] < min(low[i+1 : i+lookback+1])  (right side)
    """
    lows = df["Low"].values
    n = len(lows)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        if lows[i] < lows[i - lookback : i].min() and \
           lows[i] < lows[i + 1 : i + lookback + 1].min():
            is_swing_low[i] = True

    return pd.Series(is_swing_low, index=df.index, name="is_swing_low")


# ─────────────────────────────────────────────────────────────
# Zone builder
# ─────────────────────────────────────────────────────────────

def build_zones_from_swings(
    df: pd.DataFrame,
    symbol: str,
    lookback: int,
    min_body_atr_ratio: float,
) -> pd.DataFrame:
    """
    Detect swing highs and lows → build zone records → return DataFrame.

    Parameters
    ----------
    df                 : Processed OHLCV DataFrame (must have ATR column)
    symbol             : Ticker string (stored in output for identification)
    lookback           : Swing detection lookback N (from config)
    min_body_atr_ratio : Minimum candle body / ATR ratio (from config)

    Returns
    -------
    DataFrame of zones sorted by formation_date.
    """
    if "ATR" not in df.columns:
        raise ValueError(
            "DataFrame must have an 'ATR' column. Run preprocessor.py first."
        )

    swing_highs = detect_swing_highs(df, lookback)
    swing_lows  = detect_swing_lows(df, lookback)
    records     = []

    # ── Resistance zones from swing highs ──────────────────────
    for idx in df.index[swing_highs]:
        row = df.loc[idx]
        atr = row["ATR"]
        if pd.isna(atr) or atr == 0:
            continue

        body = abs(row["Close"] - row["Open"])
        if body < min_body_atr_ratio * atr:
            continue

        upper = row["High"]
        lower = max(row["Open"], row["Close"])
        if upper <= lower:
            lower = upper - 0.01 * atr

        records.append(_zone_record(
            symbol, "resistance", upper, lower, atr,
            idx, df.index.get_loc(idx), row["High"], row["Low"],
            lookback, body
        ))

    # ── Support zones from swing lows ──────────────────────────
    for idx in df.index[swing_lows]:
        row = df.loc[idx]
        atr = row["ATR"]
        if pd.isna(atr) or atr == 0:
            continue

        body = abs(row["Close"] - row["Open"])
        if body < min_body_atr_ratio * atr:
            continue

        upper = min(row["Open"], row["Close"])
        lower = row["Low"]
        if upper <= lower:
            upper = lower + 0.01 * atr

        records.append(_zone_record(
            symbol, "support", upper, lower, atr,
            idx, df.index.get_loc(idx), row["High"], row["Low"],
            lookback, body
        ))

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out.sort_values("formation_date", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _zone_record(
    symbol, zone_type, upper, lower, atr,
    formation_date, formation_index, swing_high, swing_low,
    lookback, candle_body
) -> dict:
    """Helper to build a consistent zone record dict."""
    return {
        "symbol"           : symbol,
        "zone_type"        : zone_type,
        "upper_boundary"   : upper,
        "lower_boundary"   : lower,
        "midpoint"         : (upper + lower) / 2,
        "width"            : upper - lower,
        "width_atr"        : (upper - lower) / atr,
        "formation_date"   : formation_date,
        "formation_index"  : formation_index,
        "swing_high"       : swing_high,
        "swing_low"        : swing_low,
        "lookback"         : lookback,
        "candle_body"      : candle_body,
        "atr_at_formation" : atr,
        "is_valid"         : True,
    }


# ─────────────────────────────────────────────────────────────
# Public API — main entry point
# ─────────────────────────────────────────────────────────────

def detect_swing_zones(
    df: pd.DataFrame,
    symbol: str,
    lookback: Optional[int] = None,
    min_body_atr_ratio: Optional[float] = None,
    zone_type_filter: Optional[str] = None,
    cfg=None,
) -> pd.DataFrame:
    """
    Detect swing-based support and resistance zones.

    Parameters are loaded from config/zone_config.yaml by default.
    You can override any parameter by passing it explicitly.

    Parameters
    ----------
    df                 : Processed OHLCV DataFrame from preprocessor.py
    symbol             : Ticker string
    lookback           : Override config swing.lookback (default: from config)
    min_body_atr_ratio : Override config swing.min_body_atr_ratio
    zone_type_filter   : None = all zones | 'support' | 'resistance'
    cfg                : ZoneConfig instance (created from config if None)

    Returns
    -------
    DataFrame of detected zones sorted by formation_date.

    Example
    -------
    >>> from src.zones import detect_swing_zones
    >>> zones = detect_swing_zones(df, 'RELIANCE.NS')
    >>> # Override lookback for a specific call
    >>> zones = detect_swing_zones(df, 'RELIANCE.NS', lookback=10)
    """
    if cfg is None:
        cfg = _get_cfg()

    _lookback  = lookback           if lookback           is not None else cfg.swing.lookback
    _min_ratio = min_body_atr_ratio if min_body_atr_ratio is not None else cfg.swing.min_body_atr_ratio

    zones = build_zones_from_swings(df, symbol, _lookback, _min_ratio)

    if zones.empty:
        return zones

    if zone_type_filter is not None:
        zones = zones[zones["zone_type"] == zone_type_filter].reset_index(drop=True)

    return zones


# ─────────────────────────────────────────────────────────────
# Nearby zone filter
# ─────────────────────────────────────────────────────────────

def get_nearby_zones(
    zones: pd.DataFrame,
    current_price: float,
    atr: float,
    price_window_atr: Optional[float] = None,
    cfg=None,
) -> pd.DataFrame:
    """
    Filter zones to those within price_window_atr * ATR of current_price.

    price_window_atr defaults to config nearby_filter.price_window_atr.
    """
    if zones.empty:
        return zones

    if cfg is None:
        cfg = _get_cfg()

    window_atr = price_window_atr if price_window_atr is not None \
                 else cfg.nearby_filter.price_window_atr
    window = window_atr * atr

    mask = (
        (zones["upper_boundary"] >= current_price - window) &
        (zones["lower_boundary"] <= current_price + window)
    )
    return zones[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────

def save_zones(zones: pd.DataFrame, symbol: str, zones_dir: Path) -> Path:
    """Save detected zones for a symbol to CSV."""
    zones_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace(".", "_").replace("^", "IDX_")
    path = zones_dir / f"{safe}_swing_zones.csv"
    zones.to_csv(path, index=False)
    return path


def load_zones(symbol: str, zones_dir: Path) -> Optional[pd.DataFrame]:
    """Load previously saved zones for a symbol."""
    safe = symbol.replace(".", "_").replace("^", "IDX_")
    path = zones_dir / f"{safe}_swing_zones.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["formation_date"])


# ─────────────────────────────────────────────────────────────
# Summary utility
# ─────────────────────────────────────────────────────────────

def zone_summary(zones: pd.DataFrame, symbol: str) -> None:
    """Print a readable summary of detected zones."""
    if zones.empty:
        print(f"[{symbol}] No zones detected.")
        return

    n_sup = (zones["zone_type"] == "support").sum()
    n_res = (zones["zone_type"] == "resistance").sum()

    print(f"\n{'='*55}")
    print(f"Zone Summary — {symbol}")
    print(f"{'='*55}")
    print(f"Total zones     : {len(zones)}")
    print(f"Support zones   : {n_sup}")
    print(f"Resistance zones: {n_res}")
    print(f"Date range      : {zones['formation_date'].min().date()} → "
          f"{zones['formation_date'].max().date()}")
    print(f"Avg zone width  : {zones['width'].mean():.2f} price units "
          f"({zones['width_atr'].mean():.2f}x ATR)")

    strength_col = "zone_strength" if "zone_strength" in zones.columns else None
    if strength_col:
        print(f"Avg strength    : {zones[strength_col].mean():.3f}")

    print(f"\nMost recent zones:")
    display_cols = ["zone_type", "upper_boundary", "lower_boundary", "width_atr", "formation_date"]
    if strength_col:
        display_cols.append("zone_strength")
    print(zones[display_cols].tail(8).to_string(index=False))
    print(f"{'='*55}\n")
