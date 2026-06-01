"""
zone_scorer.py
==============
Scores each zone by strength on a scale of 0.0 to 1.0.

All parameters are read from config/zone_config.yaml (scorer section).
Do not hardcode any threshold here — change zone_config.yaml instead.

Config parameters used (zone_config.yaml → scorer section):
    weights.*                  : component weights (must sum to 1.0)
    max_useful_touches         : cap for touch count scoring
    touch_tolerance_atr        : ATR multiple for touch detection
    strong_move_threshold_atr  : ATR move that earns full move score
    lookahead_candles          : candles ahead to measure originating move
    freshness_decay_candles    : candles over which freshness decays
    freshness_min_score        : minimum freshness (zones never → 0)
    ideal_width_atr_min/max    : ideal zone width range (score=1.0 here)
    min_width_atr              : below this → near-zero width score
    max_width_atr              : above this → declining width score
    min_strength_threshold     : drop zones below this final score

Scoring Components
------------------
1. Touch count score   (default weight 0.30)
2. Move magnitude score (default weight 0.35)
3. Freshness score     (default weight 0.20)
4. Width quality score (default weight 0.15)

Final: zone_strength = weighted sum of all four components.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


def _get_cfg():
    from src.zones import ZoneConfig
    return ZoneConfig()


# ─────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────

def _score_touch_count(n_touches: int, max_useful: int) -> float:
    """0 touches → 0.0, max_useful touches → 1.0, capped beyond."""
    return min(n_touches / max_useful, 1.0)


def _score_move_magnitude(move_atr: float, threshold: float) -> float:
    """0 ATR move → 0.0, threshold ATRs or more → 1.0."""
    return min(move_atr / threshold, 1.0)


def _score_freshness(
    formation_index: int,
    current_index: int,
    decay_candles: int,
    min_score: float,
) -> float:
    """
    Linear decay from 1.0 (just formed) to min_score (very old).
    Zones never become fully irrelevant (min_score > 0).
    """
    age = max(current_index - formation_index, 0)
    score = 1.0 - (1.0 - min_score) * min(age / decay_candles, 1.0)
    return max(score, min_score)


def _score_width(
    width_atr: float,
    ideal_min: float,
    ideal_max: float,
    min_width: float,
    max_width: float,
) -> float:
    """
    Trapezoidal scoring:
      < min_width           → near-zero (data artefact risk)
      min_width → ideal_min → rises from 0.1 to 1.0
      ideal_min → ideal_max → 1.0  (peak region)
      ideal_max → max_width → falls from 1.0 to 0.3
      > max_width           → low (zone too vague)
    """
    if width_atr < min_width:
        return 0.1
    elif width_atr <= ideal_min:
        return 0.1 + (width_atr - min_width) / (ideal_min - min_width) * 0.9
    elif width_atr <= ideal_max:
        return 1.0
    elif width_atr <= max_width:
        return 1.0 - (width_atr - ideal_max) / (max_width - ideal_max) * 0.7
    else:
        return max(0.3 - (width_atr - max_width) * 0.1, 0.1)


# ─────────────────────────────────────────────────────────────
# Touch count and move magnitude computation
# ─────────────────────────────────────────────────────────────

def compute_touch_counts(
    zones: pd.DataFrame,
    df: pd.DataFrame,
    touch_tolerance_atr: float,
) -> pd.Series:
    """
    For each zone, count how many post-formation candles touched the zone.

    Support  : a candle's Low entered the zone band (± tolerance)
    Resistance: a candle's High entered the zone band (± tolerance)
    """
    counts = []
    for _, zone in zones.iterrows():
        atr       = zone["atr_at_formation"]
        tol       = touch_tolerance_atr * atr
        upper     = zone["upper_boundary"] + tol
        lower     = zone["lower_boundary"] - tol
        future_df = df.iloc[zone["formation_index"] + 1 :]

        if zone["zone_type"] == "support":
            n = ((future_df["Low"] <= upper) & (future_df["Low"] >= lower - atr)).sum()
        else:
            n = ((future_df["High"] >= lower) & (future_df["High"] <= upper + atr)).sum()

        counts.append(n)
    return pd.Series(counts, index=zones.index, name="touch_count")


def compute_move_magnitudes(
    zones: pd.DataFrame,
    df: pd.DataFrame,
    lookahead_candles: int,
) -> pd.Series:
    """
    Magnitude of the move that originated from each zone, in ATR units.

    Support  : max High over next N candles - zone midpoint
    Resistance: zone midpoint - min Low over next N candles
    """
    magnitudes = []
    for _, zone in zones.iterrows():
        atr      = zone["atr_at_formation"]
        midpoint = zone["midpoint"]
        end_idx  = min(zone["formation_index"] + lookahead_candles + 1, len(df))
        future   = df.iloc[zone["formation_index"] + 1 : end_idx]

        if future.empty or atr == 0:
            magnitudes.append(0.0)
            continue

        if zone["zone_type"] == "support":
            move = future["High"].max() - midpoint
        else:
            move = midpoint - future["Low"].min()

        magnitudes.append(max(move / atr, 0.0))
    return pd.Series(magnitudes, index=zones.index, name="move_magnitude_atr")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def score_zones(
    zones: pd.DataFrame,
    df: pd.DataFrame,
    current_bar_index: Optional[int] = None,
    cfg=None,
) -> pd.DataFrame:
    """
    Compute composite strength score (0.0–1.0) for each zone.

    All scoring parameters are read from config/zone_config.yaml
    (scorer section). Override by passing a custom ZoneConfig.

    Parameters
    ----------
    zones             : DataFrame of zones (merged or raw)
    df                : Processed OHLCV DataFrame
    current_bar_index : Index of the "current" candle for freshness scoring.
                        Defaults to last candle in df.
    cfg               : ZoneConfig instance (created from config if None)

    Returns
    -------
    Original zones DataFrame with added columns:
        touch_count, move_magnitude_atr,
        score_touch, score_move, score_freshness, score_width,
        zone_strength

    Example
    -------
    >>> from src.zones import score_zones
    >>> zones_scored = score_zones(zones_merged, df)
    >>> zones_scored.sort_values('zone_strength', ascending=False).head(5)
    """
    if zones.empty:
        return zones

    if cfg is None:
        cfg = _get_cfg()

    sc = cfg.scorer       # shorthand
    w  = sc.weights       # weight sub-section

    if current_bar_index is None:
        current_bar_index = len(df) - 1

    zones = zones.copy()

    # ── Raw metrics ───────────────────────────────────────────
    zones["touch_count"]        = compute_touch_counts(zones, df, sc.touch_tolerance_atr)
    zones["move_magnitude_atr"] = compute_move_magnitudes(zones, df, sc.lookahead_candles)

    # ── Component scores ──────────────────────────────────────
    zones["score_touch"] = zones["touch_count"].apply(
        lambda x: _score_touch_count(x, sc.max_useful_touches)
    )
    zones["score_move"] = zones["move_magnitude_atr"].apply(
        lambda x: _score_move_magnitude(x, sc.strong_move_threshold_atr)
    )
    zones["score_freshness"] = zones["formation_index"].apply(
        lambda idx: _score_freshness(
            idx, current_bar_index,
            sc.freshness_decay_candles,
            sc.freshness_min_score,
        )
    )
    zones["score_width"] = zones["width_atr"].apply(
        lambda x: _score_width(
            x,
            sc.ideal_width_atr_min,
            sc.ideal_width_atr_max,
            sc.min_width_atr,
            sc.max_width_atr,
        )
    )

    # ── Composite score ───────────────────────────────────────
    zones["zone_strength"] = (
        w.touch     * zones["score_touch"]
        + w.move      * zones["score_move"]
        + w.freshness * zones["score_freshness"]
        + w.width     * zones["score_width"]
    ).round(4)

    # ── Optional: drop weak zones ─────────────────────────────
    min_thresh = sc.min_strength_threshold
    if min_thresh > 0.0:
        zones = zones[zones["zone_strength"] >= min_thresh].reset_index(drop=True)

    return zones
