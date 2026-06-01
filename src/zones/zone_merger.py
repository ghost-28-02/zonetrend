"""
zone_merger.py
==============
Merges overlapping or nearby zones of the same type.

All parameters are read from config/zone_config.yaml (merger section).
Do not hardcode any threshold here — change zone_config.yaml instead.

Config parameters used (zone_config.yaml → merger section):
    merge_distance_atr       : max gap in ATR units to still trigger a merge
    iterate_to_convergence   : whether to repeat until no further merges occur

Why merging is necessary
------------------------
Swing detection often finds multiple closely spaced swing lows that represent
the same support zone. Merging them produces a single accurate zone record
with the correct boundaries, rather than three separate weak-looking zones.

Merge condition
---------------
Two zones of the SAME type merge when:
    - They overlap (ranges intersect), OR
    - The gap between them is < merge_distance_atr * average ATR

Merged zone properties:
    - lower_boundary  = min of both lower boundaries
    - upper_boundary  = max of both upper boundaries
    - formation_date  = earliest of the two (first appearance)
    - atr_at_formation = average of the two ATR values
"""

from __future__ import annotations
from typing import Optional

import pandas as pd


def _get_cfg():
    from src.zones import ZoneConfig
    return ZoneConfig()


# ─────────────────────────────────────────────────────────────
# Merge helpers
# ─────────────────────────────────────────────────────────────

def _should_merge(a: pd.Series, b: pd.Series, merge_distance_atr: float) -> bool:
    """Return True if zones a and b should be merged."""
    if a["zone_type"] != b["zone_type"]:
        return False

    lo_a, hi_a = a["lower_boundary"], a["upper_boundary"]
    lo_b, hi_b = b["lower_boundary"], b["upper_boundary"]

    # Overlap: one zone's lower is below the other's upper
    if lo_b <= hi_a and lo_a <= hi_b:
        return True

    # Proximity: gap is within merge_distance_atr * average ATR
    gap     = max(lo_b - hi_a, lo_a - hi_b, 0.0)
    avg_atr = (a["atr_at_formation"] + b["atr_at_formation"]) / 2.0
    return gap <= merge_distance_atr * avg_atr


def _merge_two(a: pd.Series, b: pd.Series) -> pd.Series:
    """Combine two zones into one spanning both ranges."""
    merged = a.copy()
    merged["lower_boundary"]   = min(a["lower_boundary"],   b["lower_boundary"])
    merged["upper_boundary"]   = max(a["upper_boundary"],   b["upper_boundary"])
    merged["midpoint"]         = (merged["upper_boundary"] + merged["lower_boundary"]) / 2
    merged["width"]            = merged["upper_boundary"] - merged["lower_boundary"]
    merged["atr_at_formation"] = (a["atr_at_formation"] + b["atr_at_formation"]) / 2
    merged["width_atr"]        = merged["width"] / merged["atr_at_formation"]
    merged["formation_date"]   = min(a["formation_date"],   b["formation_date"])
    merged["formation_index"]  = min(a["formation_index"],  b["formation_index"])
    return merged


def _single_pass(zones_df: pd.DataFrame, merge_distance_atr: float) -> list:
    """
    One sweep through zones sorted by lower_boundary.
    Merges each zone into the running zone if the merge condition is met.
    """
    sorted_z  = zones_df.sort_values("lower_boundary").reset_index(drop=True)
    result    = [sorted_z.iloc[0].copy()]

    for i in range(1, len(sorted_z)):
        current = sorted_z.iloc[i]
        if _should_merge(result[-1], current, merge_distance_atr):
            result[-1] = _merge_two(result[-1], current)
        else:
            result.append(current.copy())

    return result


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def merge_zones(
    zones: pd.DataFrame,
    merge_distance_atr: Optional[float] = None,
    iterate_to_convergence: Optional[bool] = None,
    cfg=None,
) -> pd.DataFrame:
    """
    Merge overlapping and nearby zones of the same type.

    Parameters are loaded from config/zone_config.yaml by default.
    You can override any parameter by passing it explicitly.

    Parameters
    ----------
    zones                  : DataFrame of zones from detect_swing_zones()
    merge_distance_atr     : Override config merger.merge_distance_atr
    iterate_to_convergence : Override config merger.iterate_to_convergence
    cfg                    : ZoneConfig instance (created from config if None)

    Returns
    -------
    DataFrame of merged zones (fewer rows, wider zones).

    Example
    -------
    >>> from src.zones import merge_zones
    >>> zones_merged = merge_zones(zones_raw)
    >>> # Aggressive merging
    >>> zones_merged = merge_zones(zones_raw, merge_distance_atr=1.0)
    """
    if zones.empty:
        return zones

    if cfg is None:
        cfg = _get_cfg()

    _dist    = merge_distance_atr       if merge_distance_atr       is not None else cfg.merger.merge_distance_atr
    _iterate = iterate_to_convergence   if iterate_to_convergence   is not None else cfg.merger.iterate_to_convergence

    def _merge_type(type_df: pd.DataFrame) -> pd.DataFrame:
        """Merge one zone type, optionally repeating until stable."""
        current = type_df.copy()
        if not _iterate:
            return pd.DataFrame(_single_pass(current, _dist))

        # Repeat until no further merges occur (handles 3+ overlapping zones)
        prev_count = len(current) + 1
        while len(current) < prev_count:
            prev_count = len(current)
            merged     = _single_pass(current, _dist)
            current    = pd.DataFrame(merged)
        return current

    parts = []
    for zone_type in ("support", "resistance"):
        subset = zones[zones["zone_type"] == zone_type]
        if not subset.empty:
            parts.append(_merge_type(subset))

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result.sort_values("formation_date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
