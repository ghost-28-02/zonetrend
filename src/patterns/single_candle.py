"""
single_candle.py
================
Detects single-candle patterns from OHLCV data.

Patterns implemented
--------------------
Bullish signals:
    1. Hammer
    2. Inverted Hammer
    3. Dragonfly Doji

Bearish signals:
    4. Shooting Star
    5. Gravestone Doji

Neutral / Indecision:
    6. Doji
    7. Spinning Top

Design principles
-----------------
- All thresholds are ATR-relative, not absolute price values.
  This ensures the same rules work for a ₹50 stock and a ₹5000 stock.
- Each pattern function is independent and returns a boolean Series.
- The master function detect_single_candle_patterns() runs all detectors
  and returns one column per pattern added to the input DataFrame.
- No look-ahead: each candle is evaluated using only its own OHLCV values
  and the ATR computed from prior candles (already in the processed DataFrame).

Threshold parameters
--------------------
All defaults are stored here and can be overridden by passing a params dict.
When pattern_config.yaml is added, this file will read from it.

    min_body_atr      : Minimum body size relative to ATR.
                        Ensures the candle is not a doji when we need a real body.

    min_wick_body_ratio : Minimum (long wick / body) ratio.
                          For hammer/shooting star, the long wick must be at least
                          this many times larger than the body.

    max_opposite_wick_ratio : Maximum allowed opposite wick as a fraction of total range.
                              Keeps the "opposite side" wick small.

    doji_body_atr     : Maximum body size (as ATR fraction) to qualify as a doji.
                        A doji has virtually no body.

    spinning_top_body_atr : Maximum body size for a spinning top.
                            Larger than doji but smaller than a normal candle.

    min_wick_atr      : Minimum long wick size relative to ATR.
                        Ensures the wick is a meaningful rejection, not noise.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Default thresholds
# ─────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # Hammer / Shooting Star / Inverted Hammer
    "min_wick_body_ratio"       : 2.0,   # long wick >= 2x body
    "max_opposite_wick_ratio"   : 0.15,  # opposite wick <= 15% of total range
    "min_body_atr"              : 0.05,  # body >= 5% of ATR (not a doji)
    "min_wick_atr"              : 0.3,   # long wick >= 30% of ATR

    # Doji
    "doji_body_atr"             : 0.05,  # body <= 5% of ATR → doji
    "doji_min_wick_atr"         : 0.1,   # at least one wick > 10% of ATR

    # Dragonfly / Gravestone Doji
    "doji_max_upper_ratio"      : 0.05,  # upper wick <= 5% of range (dragonfly)
    "doji_max_lower_ratio"      : 0.05,  # lower wick <= 5% of range (gravestone)

    # Spinning Top
    "spinning_body_atr_max"     : 0.3,   # body <= 30% of ATR
    "spinning_body_atr_min"     : 0.05,  # body > 5% of ATR (distinguishes from doji)
    "spinning_min_wick_atr"     : 0.1,   # each wick >= 10% of ATR
}


# ─────────────────────────────────────────────────────────────
# Low-level geometry helpers
# ─────────────────────────────────────────────────────────────

def _components(o, h, l, c):
    """
    Return the five geometric components of a single candle.
    All values are scalar floats.

    body        = |close - open|
    upper_wick  = high - max(open, close)
    lower_wick  = min(open, close) - low
    candle_range = high - low
    is_bullish  = close >= open
    """
    body        = abs(c - o)
    upper_wick  = h - max(o, c)
    lower_wick  = min(o, c) - l
    candle_range = h - l
    is_bullish  = c >= o
    return body, upper_wick, lower_wick, candle_range, is_bullish


# ─────────────────────────────────────────────────────────────
# Pattern 1 — Hammer (bullish reversal at support)
# ─────────────────────────────────────────────────────────────

def is_hammer(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Hammer — bullish single-candle reversal pattern.

    Structure:
        - Small body near the TOP of the candle range
        - Long lower wick (at least 2x the body)
        - Small or no upper wick (<=15% of total range)
        - Body must be real (not a doji)

    Market psychology:
        Sellers drove price far below the open during the session.
        But buyers overwhelmed them and pushed price back up to close
        near the open. Shows strong buying pressure at the lows.
        Most significant when it forms at a support zone.

    Both bullish and bearish hammers are valid (colour does not matter).
    A bullish (green) hammer is slightly stronger.

    Returns
    -------
    Boolean Series: True where the hammer pattern is detected.
    """
    result = pd.Series(False, index=open_.index)

    for i in range(len(open_)):
        o, h, l, c, a = open_.iat[i], high.iat[i], low.iat[i], close.iat[i], atr.iat[i]
        if pd.isna(a) or a == 0:
            continue

        body, upper_wick, lower_wick, rng, _ = _components(o, h, l, c)

        if rng == 0:
            continue

        # Body must exist (filters doji-like candles)
        if body < params["min_body_atr"] * a:
            continue

        # Long lower wick must be at least min_wick_atr * ATR
        if lower_wick < params["min_wick_atr"] * a:
            continue

        # Lower wick must be at least min_wick_body_ratio times the body
        if body > 0 and lower_wick / body < params["min_wick_body_ratio"]:
            continue

        # Upper wick must be small
        if upper_wick / rng > params["max_opposite_wick_ratio"]:
            continue

        result.iat[i] = True

    return result


# ─────────────────────────────────────────────────────────────
# Pattern 2 — Inverted Hammer (bullish reversal at support)
# ─────────────────────────────────────────────────────────────

def is_inverted_hammer(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Inverted Hammer — potential bullish reversal.

    Structure:
        - Small body near the BOTTOM of the candle range
        - Long upper wick (at least 2x the body)
        - Small or no lower wick (<=15% of total range)
        - Body must be real

    Market psychology:
        Buyers tried to push price up (long upper wick) but were partially
        resisted. However, the fact that price recovered from its lows and
        did NOT close near the high is less convincing than a hammer.
        Requires confirmation from the next candle (next candle closes higher).

    Note: Structurally identical to a shooting star. Context distinguishes them:
        - After a downtrend at support → Inverted Hammer (bullish)
        - After an uptrend at resistance → Shooting Star (bearish)

    Returns
    -------
    Boolean Series: True where the inverted hammer pattern is detected.
    """
    result = pd.Series(False, index=open_.index)

    for i in range(len(open_)):
        o, h, l, c, a = open_.iat[i], high.iat[i], low.iat[i], close.iat[i], atr.iat[i]
        if pd.isna(a) or a == 0:
            continue

        body, upper_wick, lower_wick, rng, _ = _components(o, h, l, c)

        if rng == 0:
            continue

        if body < params["min_body_atr"] * a:
            continue

        if upper_wick < params["min_wick_atr"] * a:
            continue

        if body > 0 and upper_wick / body < params["min_wick_body_ratio"]:
            continue

        # Lower wick must be small
        if lower_wick / rng > params["max_opposite_wick_ratio"]:
            continue

        result.iat[i] = True

    return result


# ─────────────────────────────────────────────────────────────
# Pattern 3 — Shooting Star (bearish reversal at resistance)
# ─────────────────────────────────────────────────────────────

def is_shooting_star(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Shooting Star — bearish single-candle reversal pattern.

    Structure:
        - Small body near the BOTTOM of the candle range
        - Long upper wick (at least 2x the body)
        - Small or no lower wick (<=15% of total range)
        - Body must be real

    Market psychology:
        Buyers pushed price significantly higher during the session
        (long upper wick) but sellers overwhelmed them and drove price
        back down to close near the opening price. Shows strong selling
        pressure at the highs. Most significant at a resistance zone.

    Structurally identical to an inverted hammer — context is the key:
        - After an uptrend at resistance → Shooting Star (bearish)

    Returns
    -------
    Boolean Series: True where the shooting star pattern is detected.
    """
    # Structurally identical to inverted hammer — same geometric check
    return is_inverted_hammer(open_, high, low, close, atr, params)


# ─────────────────────────────────────────────────────────────
# Pattern 4 — Doji (indecision / potential reversal)
# ─────────────────────────────────────────────────────────────

def is_doji(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Doji — indecision candle with open ≈ close.

    Structure:
        - Very small body (open ≈ close, body <= 5% of ATR)
        - Wicks can extend in either or both directions
        - At least one wick must exist (not a completely flat candle)

    Market psychology:
        Neither buyers nor sellers won the session. Price opened,
        moved in both directions, and returned to the opening price.
        Complete balance between supply and demand.

    Context determines meaning:
        - After uptrend at resistance → potential bearish reversal
        - After downtrend at support  → potential bullish reversal
        - In the middle of a trend    → usually ignored

    Note: Dragonfly Doji and Gravestone Doji are specialised dojis
    handled by their own functions below.

    Returns
    -------
    Boolean Series: True where the generic doji is detected.
    (This excludes dragonfly and gravestone dojis to avoid overlap.)
    """
    result = pd.Series(False, index=open_.index)

    for i in range(len(open_)):
        o, h, l, c, a = open_.iat[i], high.iat[i], low.iat[i], close.iat[i], atr.iat[i]
        if pd.isna(a) or a == 0:
            continue

        body, upper_wick, lower_wick, rng, _ = _components(o, h, l, c)

        if rng == 0:
            continue

        # Body must be tiny
        if body > params["doji_body_atr"] * a:
            continue

        # At least one wick must be meaningful (not a flat line)
        if max(upper_wick, lower_wick) < params["doji_min_wick_atr"] * a:
            continue

        # Exclude dragonfly (almost no upper wick)
        if upper_wick / rng <= params["doji_max_upper_ratio"]:
            continue

        # Exclude gravestone (almost no lower wick)
        if lower_wick / rng <= params["doji_max_lower_ratio"]:
            continue

        result.iat[i] = True

    return result


# ─────────────────────────────────────────────────────────────
# Pattern 5 — Dragonfly Doji (bullish reversal at support)
# ─────────────────────────────────────────────────────────────

def is_dragonfly_doji(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Dragonfly Doji — strong bullish reversal signal.

    Structure:
        - Open, High, and Close are all at approximately the same level
          (top of the candle range)
        - Long lower wick
        - Little or no upper wick (upper wick <= 5% of total range)

    Market psychology:
        Sellers drove price far down from the open (long lower wick)
        but by the close, buyers had completely reclaimed all the losses.
        Price closed exactly at the high. This is a very strong signal
        of buyer dominance. More powerful than a standard hammer because
        the close is at the absolute high of the session.

    Returns
    -------
    Boolean Series: True where the dragonfly doji is detected.
    """
    result = pd.Series(False, index=open_.index)

    for i in range(len(open_)):
        o, h, l, c, a = open_.iat[i], high.iat[i], low.iat[i], close.iat[i], atr.iat[i]
        if pd.isna(a) or a == 0:
            continue

        body, upper_wick, lower_wick, rng, _ = _components(o, h, l, c)

        if rng == 0:
            continue

        # Body must be very small (doji condition)
        if body > params["doji_body_atr"] * a:
            continue

        # Upper wick must be very small (price closed at or near the high)
        if upper_wick / rng > params["doji_max_upper_ratio"]:
            continue

        # Long lower wick must exist
        if lower_wick < params["min_wick_atr"] * a:
            continue

        result.iat[i] = True

    return result


# ─────────────────────────────────────────────────────────────
# Pattern 6 — Gravestone Doji (bearish reversal at resistance)
# ─────────────────────────────────────────────────────────────

def is_gravestone_doji(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Gravestone Doji — strong bearish reversal signal.

    Structure:
        - Open, Low, and Close are all at approximately the same level
          (bottom of the candle range)
        - Long upper wick
        - Little or no lower wick (lower wick <= 5% of total range)

    Market psychology:
        The mirror of the dragonfly doji. Buyers pushed price
        significantly above the open (long upper wick) but sellers
        completely overwhelmed them and drove price back to the open
        by close. Price closed at the absolute low. Very strong signal
        of seller dominance at resistance zones.

    Returns
    -------
    Boolean Series: True where the gravestone doji is detected.
    """
    result = pd.Series(False, index=open_.index)

    for i in range(len(open_)):
        o, h, l, c, a = open_.iat[i], high.iat[i], low.iat[i], close.iat[i], atr.iat[i]
        if pd.isna(a) or a == 0:
            continue

        body, upper_wick, lower_wick, rng, _ = _components(o, h, l, c)

        if rng == 0:
            continue

        # Body must be very small (doji condition)
        if body > params["doji_body_atr"] * a:
            continue

        # Lower wick must be very small (price closed at or near the low)
        if lower_wick / rng > params["doji_max_lower_ratio"]:
            continue

        # Long upper wick must exist
        if upper_wick < params["min_wick_atr"] * a:
            continue

        result.iat[i] = True

    return result


# ─────────────────────────────────────────────────────────────
# Pattern 7 — Spinning Top (indecision, both wicks present)
# ─────────────────────────────────────────────────────────────

def is_spinning_top(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, atr: pd.Series,
    params: dict = DEFAULT_PARAMS,
) -> pd.Series:
    """
    Spinning Top — indecision candle with wicks on both sides.

    Structure:
        - Small body (larger than doji, but smaller than a normal candle)
        - Meaningful upper AND lower wicks on both sides
        - Body is roughly centred in the candle range

    Market psychology:
        Similar to a doji but with a slightly larger body. Both buyers
        and sellers were active during the session. Neither side achieved
        a decisive win. Like the doji, its significance is context-dependent.
        At a zone boundary, it signals that the market is at a decision point.

    Difference from Doji:
        A doji has open ≈ close (body ≈ 0).
        A spinning top has a small but visible body.

    Returns
    -------
    Boolean Series: True where the spinning top is detected.
    """
    result = pd.Series(False, index=open_.index)

    for i in range(len(open_)):
        o, h, l, c, a = open_.iat[i], high.iat[i], low.iat[i], close.iat[i], atr.iat[i]
        if pd.isna(a) or a == 0:
            continue

        body, upper_wick, lower_wick, rng, _ = _components(o, h, l, c)

        if rng == 0:
            continue

        # Body: larger than a doji but smaller than a normal candle
        if body < params["spinning_body_atr_min"] * a:
            continue  # too small → this is a doji
        if body > params["spinning_body_atr_max"] * a:
            continue  # too large → normal directional candle

        # Both wicks must be present and meaningful
        if upper_wick < params["spinning_min_wick_atr"] * a:
            continue
        if lower_wick < params["spinning_min_wick_atr"] * a:
            continue

        result.iat[i] = True

    return result


# ─────────────────────────────────────────────────────────────
# Master detection function
# ─────────────────────────────────────────────────────────────

def detect_single_candle_patterns(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Run all single-candle pattern detectors on a processed OHLCV DataFrame.

    Parameters
    ----------
    df     : Processed DataFrame with columns [Open, High, Low, Close, ATR].
             Must come from preprocessor.py (ATR column required).
    params : Optional dict of threshold overrides.
             If None, DEFAULT_PARAMS is used.

    Returns
    -------
    The original DataFrame with the following boolean columns added:

        pat_hammer          : Hammer (bullish)
        pat_inv_hammer      : Inverted Hammer (bullish, needs confirmation)
        pat_shooting_star   : Shooting Star (bearish)
        pat_doji            : Generic Doji (indecision)
        pat_dragonfly_doji  : Dragonfly Doji (bullish)
        pat_gravestone_doji : Gravestone Doji (bearish)
        pat_spinning_top    : Spinning Top (indecision)
        pat_any_single      : True if ANY single-candle pattern is detected

    None of the added columns use future data — each row is evaluated
    only from its own OHLCV values and the ATR from prior candles.

    Example
    -------
    >>> from src.patterns.single_candle import detect_single_candle_patterns
    >>> df = detect_single_candle_patterns(df)
    >>> df[df['pat_hammer']].tail(5)
    """
    if "ATR" not in df.columns:
        raise ValueError("DataFrame must have an 'ATR' column. Run preprocessor.py first.")

    p = {**DEFAULT_PARAMS, **(params or {})}

    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    a = df["ATR"]

    out = df.copy()

    out["pat_hammer"]          = is_hammer(o, h, l, c, a, p)
    out["pat_inv_hammer"]      = is_inverted_hammer(o, h, l, c, a, p)
    out["pat_shooting_star"]   = is_shooting_star(o, h, l, c, a, p)
    out["pat_doji"]            = is_doji(o, h, l, c, a, p)
    out["pat_dragonfly_doji"]  = is_dragonfly_doji(o, h, l, c, a, p)
    out["pat_gravestone_doji"] = is_gravestone_doji(o, h, l, c, a, p)
    out["pat_spinning_top"]    = is_spinning_top(o, h, l, c, a, p)

    pattern_cols = [
        "pat_hammer", "pat_inv_hammer", "pat_shooting_star",
        "pat_doji", "pat_dragonfly_doji", "pat_gravestone_doji",
        "pat_spinning_top",
    ]
    out["pat_any_single"] = out[pattern_cols].any(axis=1)

    return out


# ─────────────────────────────────────────────────────────────
# Occurrence statistics helper
# ─────────────────────────────────────────────────────────────

def pattern_occurrence_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a summary of how often each pattern occurs.

    Useful during validation to check whether thresholds are too tight
    (almost no detections) or too loose (too many detections).

    Parameters
    ----------
    df : DataFrame after detect_single_candle_patterns() has been run.

    Returns
    -------
    DataFrame with columns [pattern, count, pct_of_candles]
    """
    pattern_cols = [c for c in df.columns if c.startswith("pat_") and c != "pat_any_single"]
    total = len(df)
    rows = []
    for col in pattern_cols:
        count = df[col].sum()
        rows.append({
            "pattern"         : col.replace("pat_", ""),
            "count"           : int(count),
            "pct_of_candles"  : round(count / total * 100, 2),
        })
    stats = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return stats
