"""
zone_detector.py
================
Detects supply and demand zones using the DBR/RBD (reversal) and
RBR/DBD (continuation) structural patterns (controlled by arrival_mode).

Theory
------
Price moves are driven by imbalances between buyers and sellers.
When a strong directional move (rally or drop) originates from a tight
consolidation (the "base"), it means one side overwhelmed the other at
that price area. Unfilled institutional orders remain in the base.
When price returns to the base, those orders re-activate — creating
reliable support (demand) or resistance (supply).

The four patterns (reversal: DBR/RBD — continuation: RBR/DBD):

  DBR — Drop → Base → Rally  →  Demand Zone  (support)
  ─────────────────────────────────────────
  Price falls, enters a tight consolidation, then launches upward.
  The base is where buyers absorbed all selling and price exploded up.
  Zone top    = max(High  of base candles)
  Zone bottom = min(Low   of base candles)

  RBD — Rally → Base → Drop  →  Supply Zone  (resistance)
  ─────────────────────────────────────────
  Price rises, enters a tight consolidation, then collapses.
  The base is where sellers absorbed all buying and price collapsed.
  Zone top    = max(High  of base candles)
  Zone bottom = min(Low   of base candles)

Detection algorithm (no look-ahead bias)
-----------------------------------------
Step 1  Scan every candle for base candle eligibility:
           range (H-L)  <=  base_range_multiplier × ATR
           body  |C-O|  <=  base_body_multiplier  × ATR

Step 2  Group consecutive eligible candles into a base.
        Accept if: 1 <= length <= max_base_length
                   total base range <= base_range_multiplier × avg_ATR

Step 3  Check the MULTI-CANDLE departure / "leg-out" (review #2):
        First post-base candle must close in-direction in the stronger half
        of its range, with body >= departure_strength × ATR (a soft floor);
        AND within departure_leg_max candles the CLOSE must be displaced
        >= departure_leg_disp × ATR beyond the base edge.
        DBR → bullish leg-out (close pushes above base top)
        RBD → bearish leg-out (close pushes below base bottom)

Step 4  Measure the leg-in (candles before the base) over a FIXED window
        (zone_config.yaml → leg_in.lookback). Hard gate only when
        leg_in.min_move_atr > 0: net |displacement| over the window must be
        >= min_move_atr × ATR. Direction = sign of the net move. When
        leg_in.enabled is false the leg-in is still MEASURED (structure
        labels stay DBR/RBD/RBR/DBD, CSV format unchanged) but it neither
        gates zones nor contributes to the quality score.
        Candle-majority is returned as "arrival cleanliness" for scoring.

Step 5  If steps 3 and 4 pass → record the zone.
        Zone is considered "formed" at the CLOSE of the departure candle.
        (No future information is used — the zone is only known after the
        departure candle has fully closed.)

Step 6  Score each zone 0–1:
        Weighted combination of departure strength, base tightness,
        arrival momentum, and departure volume ratio.

Step 7  Track zone status forward in time:
        Demand zone invalidated when Close < zone bottom
        Supply zone invalidated when Close > zone top
        Each visit into the zone increments test_count.

Parameters
----------
All thresholds are in zone_config.yaml under the `zones:` key.
They are all expressed as ATR multiples — never fixed INR values —
so the detector works consistently across different volatility regimes.

Output
------
pd.DataFrame indexed by zone_id with columns:
    type, top, bottom, midpoint, width, width_atr,
    formation_date, base_start_date, base_end_date, base_length,
    avg_atr, departure_body_atr, departure_leg_atr, departure_close_ratio,
    departure_volume_ratio, base_volume_ratio, arrival_move_atr,
    arrival_cleanliness, trend_score, trend_aligned,
    weekly_trend_align, weekly_in_zone, weekly_dist_atr, weekly_zone_strength,
    weekly_zone_fresh, weekly_confluence_score, weekly_confirmed,
    strength, strength_pit, freshness_score, adjusted_strength_posthoc,
    status, test_count, last_test_date, invalidation_date

Strength columns (review #6 — look-ahead quarantine)
    strength                   formation-time, leak-free        → ML-safe
    strength_pit               strength + causal weekly bonus    → ML-safe
    adjusted_strength_posthoc  strength_pit × freshness          → ANALYSIS ONLY
                               (freshness uses future test_count; never a feature)

Improvements over v1
---------------------
  Priority 1 — Weekly-timeframe CONFLUENCE (multi-timeframe)
      Weekly OHLCV is fetched directly from Yahoo (1wk) by the data pipeline
      and passed in as `weekly_df`. Zones are detected on it, and each daily
      zone receives a CAUSAL confluence feature block — trend alignment, in
      a weekly zone, distance to the nearest weekly zone, weekly zone strength
      and freshness — plus a 0..1 `weekly_confluence_score`. Only weekly
      bars/zones known by the daily zone's formation date are used (no
      look-ahead). This replaces the old single `weekly_confirmed` flag
      (still emitted, = `weekly_in_zone`). Resampling is a fallback only.

  Priority 2 — Zone freshness decay
      After status tracking, a `freshness_score` is computed from
      test_count using a configurable decay schedule. `adjusted_strength`
      multiplies raw strength by the freshness multiplier and adds the
      weekly bonus. Use `adjusted_strength` for ranking and filtering.

  Priority 3 — Base volume ratio
      The average volume of base candles relative to the 20-day volume MA
      is recorded as `base_volume_ratio`. Low base volume is a positive
      signal (unfilled institutional orders remain). This metric is
      incorporated into the strength score alongside departure volume.

Usage
-----
    from src.zones.zone_detector import detect, save_zones, load_zones

    zones = detect(proc_df, zone_cfg)
    save_zones(zones, symbol, zones_dir)
    zones = load_zones(symbol, zones_dir)
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
MAIN_CFG_PATH   = PROJECT_ROOT / "config" / "config.yaml"
ZONE_CFG_PATH   = PROJECT_ROOT / "config" / "zone_config.yaml"


# ── Config & logging ──────────────────────────────────────────────────────────

def load_configs() -> tuple[dict, dict]:
    """Load main config and zone config. Returns (main_cfg, zone_cfg)."""
    with open(MAIN_CFG_PATH)  as f: main_cfg = yaml.safe_load(f)
    with open(ZONE_CFG_PATH)  as f: zone_cfg = yaml.safe_load(f)["zones"]
    return main_cfg, zone_cfg


def setup_logging(main_cfg: dict) -> logging.Logger:
    log_dir  = PROJECT_ROOT / main_cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    level    = getattr(logging, main_cfg["logging"]["log_level"].upper(), logging.INFO)
    handlers = []
    if main_cfg["logging"]["log_to_console"]:
        handlers.append(logging.StreamHandler())
    if main_cfg["logging"]["log_to_file"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(log_dir / f"zone_detector_{ts}.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("zone_detector")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _symbol_to_stem(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "IDX_")


def save_zones(
    zones: pd.DataFrame,
    symbol: str,
    zones_dir: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """Save detected zones DataFrame to CSV."""
    zones_dir.mkdir(parents=True, exist_ok=True)
    path = zones_dir / (_symbol_to_stem(symbol) + "_zones.csv")
    zones.to_csv(path)
    if logger:
        logger.info(f"[{symbol}] {len(zones)} zones saved → {path.relative_to(PROJECT_ROOT)}")
    return path


def load_zones(symbol: str, zones_dir: Path) -> pd.DataFrame:
    """Load a previously saved zones CSV back into a DataFrame."""
    path = zones_dir / (_symbol_to_stem(symbol) + "_zones.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"No zone file found for {symbol} at {path}. "
            "Run zone_detector.py first."
        )
    df = pd.read_csv(path, index_col="zone_id")
    for date_col in ["formation_date", "base_start_date", "base_end_date",
                     "last_test_date", "invalidation_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_processed(symbol: str, processed_dir: Path, suffix: str = "") -> pd.DataFrame:
    """Load a preprocessed OHLCV CSV for a single symbol/timeframe.

    `suffix` selects the timeframe file, e.g. suffix='_weekly' loads the
    Yahoo-fetched weekly data IDX_NSEI_weekly.csv.
    """
    path = processed_dir / (_symbol_to_stem(symbol) + suffix + ".csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. Run preprocessor.py first."
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ── Step 1 & 2: Base detection ────────────────────────────────────────────────

def _is_base_candle(
    high: float, low: float, open_: float, close: float, atr: float,
    range_mult: float, body_mult: float,
) -> bool:
    """
    Return True if this candle qualifies as a base candle.

    A base candle represents market indecision / equilibrium.
    Both its total range and its body must be tight relative to ATR.

    Parameters
    ----------
    range_mult : max allowed (H-L) / ATR
    body_mult  : max allowed |C-O| / ATR
    """
    if pd.isna(atr) or atr <= 0:
        return False
    return (
        (high - low)        <= range_mult * atr and
        abs(close - open_)  <= body_mult  * atr
    )


def _find_bases(df: pd.DataFrame, zcfg: dict) -> list[dict]:
    """
    Collect candidate base groups using an END-ANCHORED windowed scan.

    [review #4] The previous implementation grew one maximal run per start
    candle and then jumped the cursor (``i = j``) PAST the whole run — even
    when the run failed the aggregate range gate. That:
      * swallowed tight sub-bases that were never re-examined, and
      * tied base boundaries to grid alignment rather than to the actual
        departure point.

    This version anchors on every base candle as a potential base END (the
    candle immediately before a departure). For each end it takes the
    maximal run of consecutive base candles ending there (capped at
    max_base_length); if that window's aggregate range is too wide it
    retries progressively shorter windows ending at the same candle, so a
    valid tight sub-base is never lost.

    Each end yields at most one base, so a single thrust produces exactly
    one zone (no duplicates): interior base candles fail the departure test
    because their successor is itself a base candle.

    Returns
    -------
    List of dicts, each with: start_idx, end_idx, length, top, bottom, avg_atr
    """
    range_mult   = zcfg["base_range_multiplier"]                       # per-candle gate
    cluster_mult = zcfg.get("base_cluster_multiplier", range_mult)     # aggregate gate
    body_mult    = zcfg["base_body_multiplier"]
    min_len      = zcfg["min_base_length"]
    max_len      = zcfg["max_base_length"]

    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    atrs   = df["ATR"].values
    n      = len(df)

    def _window(start: int, end: int) -> dict | None:
        """Build a base dict for [start, end] if its TOTAL span passes the
        aggregate (cluster) gate. This is intentionally more generous than the
        per-candle gate so a multi-candle consolidation is not truncated."""
        top     = float(np.max(highs[start:end + 1]))
        bottom  = float(np.min(lows[start:end + 1]))
        avg_atr = float(np.nanmean(atrs[start:end + 1]))
        if avg_atr > 0 and (top - bottom) <= cluster_mult * avg_atr:
            return {
                "start_idx": start,
                "end_idx":   end,
                "length":    end - start + 1,
                "top":       top,
                "bottom":    bottom,
                "avg_atr":   avg_atr,
            }
        return None

    bases: list[dict] = []
    for end in range(n):
        if not _is_base_candle(
            highs[end], lows[end], opens[end], closes[end], atrs[end],
            range_mult, body_mult,
        ):
            continue

        # Maximal run of consecutive base candles ending at `end`,
        # capped so the window length never exceeds max_base_length.
        start = end
        while (
            start - 1 >= 0
            and (end - (start - 1) + 1) <= max_len
            and _is_base_candle(
                highs[start - 1], lows[start - 1], opens[start - 1],
                closes[start - 1], atrs[start - 1], range_mult, body_mult,
            )
        ):
            start -= 1

        max_window_len = end - start + 1
        if max_window_len < min_len:
            continue

        # Try the widest admissible window first, then shrink toward the
        # most recent candles until the aggregate range gate passes.
        for length in range(max_window_len, min_len - 1, -1):
            base = _window(end - length + 1, end)
            if base is not None:
                bases.append(base)
                break

    return bases


# ── Dynamic leg tracing (deep-analysis §4) ────────────────────────────────────

def _peak(x: float, lo: float, hi: float, width: float) -> float:
    """Peaked response: 1.0 inside [lo, hi], decaying linearly to 0 over `width`
    on each side. Used so the score rewards a MODERATE value (e.g. leg-out
    displacement 0.8-1.5 ATR), not an ever-larger one."""
    if np.isnan(x):
        return 0.5
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, 1.0 - (lo - x) / width)
    return max(0.0, 1.0 - (x - hi) / width)


def _trace_leg_out(opens, highs, lows, closes, vols, start_idx, direction,
                   atr, n, max_len, pullback_atr):
    """
    Trace the leg-OUT forward from `start_idx` until the impulse exhausts: price
    closes back `pullback_atr × ATR` from the running extreme, or `max_len` is hit.
    Returns the TRUE leg (1..N candles), not a fixed window. (deep-analysis §4)

    Returns dict: end_idx, candles, disp_atr, extreme_close, body_avg_atr,
                  vol_sum, velocity
    """
    if start_idx >= n or atr <= 0:
        return None
    ext = closes[start_idx]
    ext_idx = start_idx
    body = 0.0
    vol = 0.0
    last = start_idx
    for i in range(start_idx, min(n, start_idx + max_len)):
        body += abs(closes[i] - opens[i])
        vol += vols[i] if not np.isnan(vols[i]) else 0.0
        last = i
        if direction == "up":
            if closes[i] > ext:
                ext, ext_idx = closes[i], i
            elif (ext - closes[i]) >= pullback_atr * atr:
                break
        else:
            if closes[i] < ext:
                ext, ext_idx = closes[i], i
            elif (closes[i] - ext) >= pullback_atr * atr:
                break
    candles = ext_idx - start_idx + 1
    disp = abs(ext - closes[start_idx]) / atr
    n_used = last - start_idx + 1
    return {
        "end_idx":      ext_idx,
        "candles":      candles,
        "disp_atr":     round(disp, 4),
        "extreme_close": float(ext),
        "body_avg_atr": round((body / n_used) / atr, 4) if n_used > 0 else 0.0,
        "vol_sum":      vol,
        "velocity":     round(disp / candles, 4) if candles > 0 else 0.0,
    }


# ── Step 3: Departure check ───────────────────────────────────────────────────

def _check_departure(
    opens, highs, lows, closes, volratios,
    base_end_idx: int,
    zone_type: str,
    base_top: float,
    base_bottom: float,
    avg_atr: float,
    n: int,
    dep_floor: float,
    dep_close_ratio: float,
    leg_max: int,
    leg_disp: float,
    pullback_atr: float,
) -> tuple[bool, dict | None]:
    """
    Validate a DYNAMIC leg-out after the base (deep-analysis §4).

    Gates (first post-base candle): in-direction, closes in the stronger half of
    its range (>= dep_close_ratio), body >= dep_floor × ATR. Then the leg-out is
    traced to its TRUE length (1..N candles, ending when the impulse exhausts),
    and the leg's extreme close must clear the base edge by >= leg_disp × ATR.

    No look-ahead: the zone is "formed" at the FIRST candle whose close clears the
    base by leg_disp (the confirmation candle), not at base_end+1.

    Returns (is_valid, dict with leg-out metrics).
    """
    first = base_end_idx + 1
    if first >= n or avg_atr <= 0:
        return False, None

    o, c, h, l = opens[first], closes[first], highs[first], lows[first]
    rng = h - l
    if rng < 1e-10:
        return False, None

    direction = "up" if zone_type == "demand" else "down"

    # 1. First leg-out candle: direction + close strength
    if zone_type == "demand":
        if c < o:
            return False, None
        close_ratio = (c - l) / rng
    else:
        if c >= o:
            return False, None
        close_ratio = (h - c) / rng
    if close_ratio < dep_close_ratio:
        return False, None

    # 2. Soft body floor on the first leg-out candle
    first_body_atr = abs(c - o) / avg_atr
    if first_body_atr < dep_floor:
        return False, None

    # 3. Dynamic leg-out, then base-clearance test on its extreme close
    leg = _trace_leg_out(opens, highs, lows, closes, volratios, first,
                         direction, avg_atr, n, leg_max, pullback_atr)
    if leg is None:
        return False, None
    if zone_type == "demand":
        clearance = (leg["extreme_close"] - base_top) / avg_atr
    else:
        clearance = (base_bottom - leg["extreme_close"]) / avg_atr
    if clearance < leg_disp:
        return False, None

    # Formation candle = the first leg-out candle (base_end+1). The zone's price
    # is fixed by the base; the leg-out only confirms it. (Kept at base_end+1 for
    # continuity with the validated detector; the leg is still traced for metrics.)
    form_idx = first

    leg_vol_expansion = (leg["vol_sum"] / leg["candles"]) if leg["candles"] > 0 else np.nan

    return True, {
        "dep_idx":           form_idx,
        "first_idx":         first,
        "dep_body_atr":      round(first_body_atr, 4),
        "dep_close_ratio":   round(close_ratio, 4),
        "leg_out_clear":     round(clearance, 4),     # base-clearance (gate metric)
        "leg_out_disp":      leg["disp_atr"],         # true leg displacement
        "leg_out_candles":   leg["candles"],
        "leg_out_velocity":  leg["velocity"],
        "leg_out_end":       leg["end_idx"],
        "leg_out_vol_exp":   round(float(leg_vol_expansion), 4) if not np.isnan(leg_vol_expansion) else np.nan,
        "dep_volume_ratio":  float(volratios[first]) if not np.isnan(volratios[first]) else np.nan,
    }


# ── Step 4: Arrival check ─────────────────────────────────────────────────────

def _check_arrival(
    opens, highs, lows, closes, volratios,
    base_start_idx: int,
    leg_out_dir: str,
    base_top: float,
    base_bottom: float,
    avg_atr: float,
    n: int,
    lookback: int,
    min_move: float,
) -> tuple[bool, dict | None]:
    """
    Measure the leg-in over a FIXED, config-tunable window before the base.

    [User decision] The previous DYNAMIC backward trace (leg_pullback_atr
    based) was REMOVED in favour of explicit parameters, mirroring how the
    leg-out is tuned (zone_config.yaml → leg_in.lookback / leg_in.min_move_atr).

    Measurement: net displacement from the OPEN of the first window candle to
    the CLOSE of the last candle before the base, over the last `lookback`
    candles:

        net  = close[base_start - 1] - open[base_start - lookback]
        disp = |net| / ATR        arr_dir = sign(net)

    A leg-in is "valid" when disp >= min_move × ATR. Set min_move_atr to 0.0
    to effectively disable the hard gate (any non-flat approach qualifies) —
    the leg-in then only contributes its (down-weighted) quality component.

    Returns (has_arrival, dict: arr_dir, leg_in_disp, leg_in_candles,
             leg_in_velocity, arrival_cleanliness, origin_ok).
    """
    end = base_start_idx - 1
    if end < 0 or avg_atr <= 0:
        return False, None

    s = max(0, end - max(int(lookback), 1) + 1)
    m = end - s + 1
    net  = float(closes[end] - opens[s])
    disp = abs(net) / avg_atr
    if net == 0.0 or disp < min_move:
        return False, None

    arr_dir   = "up" if net > 0 else "down"
    origin    = float(opens[s])
    origin_ok = (origin > base_top) if arr_dir == "down" else (origin < base_bottom)
    if arr_dir == "down":
        clean = float(np.sum(closes[s:end + 1] < opens[s:end + 1])) / m
    else:
        clean = float(np.sum(closes[s:end + 1] >= opens[s:end + 1])) / m
    return True, {
        "arr_dir":             arr_dir,
        "leg_in_disp":         round(disp, 4),
        "leg_in_candles":      m,
        "leg_in_velocity":     round(disp / m, 4),
        "arrival_cleanliness": round(clean, 4),
        "origin_ok":           bool(origin_ok),
    }


# ── Priority 1 helper: Weekly timeframe ──────────────────────────────────────

def _resample_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a daily OHLCV DataFrame to weekly bars (week ending Friday).

    ATR is recomputed on the weekly bars using a 14-period Wilder's smoothing.
    VolumeRatio is recomputed as weekly Volume / 20-week rolling Volume MA.

    The weekly df has identical column names to the daily df so the existing
    `detect()` function can run on it unchanged.
    """
    weekly = daily_df.resample("W-FRI").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])

    # True Range on weekly bars (gap-aware)
    prev_close = weekly["Close"].shift(1)
    tr = pd.concat([
        weekly["High"] - weekly["Low"],
        (weekly["High"] - prev_close).abs(),
        (weekly["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's ATR (14 periods)
    atr = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    weekly["ATR"] = atr

    # VolumeRatio (weekly volume / 20-week MA)
    vol_ma = weekly["Volume"].rolling(20, min_periods=1).mean()
    weekly["VolumeRatio"] = weekly["Volume"] / vol_ma.replace(0, np.nan)

    return weekly.dropna(subset=["ATR"])


_WEEKLY_FEATURE_DEFAULTS = {
    "weekly_trend_align":      np.nan,
    "weekly_in_zone":          False,
    "weekly_dist_atr":         np.nan,
    "weekly_zone_strength":    np.nan,
    "weekly_zone_fresh":       False,
    "weekly_confluence_score": 0.0,
    "weekly_confirmed":        False,   # kept for backward compatibility (= weekly_in_zone)
}


def _add_weekly_confluence(
    zones: list[dict],
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame | None,
    zcfg: dict,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """
    Attach a CAUSAL weekly-confluence feature block to each daily zone,
    replacing the old single `weekly_confirmed` boolean.

    Weekly zones are detected on the higher-timeframe data (fetched directly
    from Yahoo as 1wk bars; resampled only as a fallback). For each daily zone
    we then compute, using ONLY weekly bars/zones knowable at the daily zone's
    formation date (no look-ahead):

        weekly_trend_align       signed HTF trend distance at formation, in
                                 weekly-ATR units, oriented so >0 = the weekly
                                 trend agrees with the zone (demand↔uptrend,
                                 supply↔downtrend).
        weekly_in_zone           the daily zone overlaps a same-type weekly zone
                                 by >= overlap_tolerance of the daily height.
        weekly_dist_atr          distance from the daily midpoint to the nearest
                                 same-type weekly zone, in daily-ATR units
                                 (0.0 when the midpoint is inside it).
        weekly_zone_strength     formation-time strength of the matched weekly
                                 zone (inherit the HTF imbalance quality).
        weekly_zone_fresh        the matched weekly zone was still untested as of
                                 the daily zone's formation date.
        weekly_confluence_score  0..1 blend of the above (ranking / strength_pit).
        weekly_confirmed         = weekly_in_zone (backward compatibility).

    Causality: weekly zones are filtered to formation_date <= the daily zone's
    formation_date; trend uses the last weekly bar at/<= that date; freshness
    counts only weekly touches strictly after the weekly zone formed and up to
    the daily formation date. Nothing from the future leaks into these features.
    """
    cfg_wc   = zcfg.get("weekly_confirmation", {})
    tol      = cfg_wc.get("overlap_tolerance", 0.30)
    trend_ma = cfg_wc.get("trend_ma", "EMA20")

    def _defaults_for_all():
        for z in zones:
            z.update(_WEEKLY_FEATURE_DEFAULTS)
        return zones

    if not cfg_wc.get("enabled", True) or weekly_df is None or len(weekly_df) == 0:
        if logger and cfg_wc.get("enabled", True):
            logger.info("  Weekly confluence: no weekly data available; features default.")
        return _defaults_for_all()

    # Detect weekly zones (no nested weekly confluence → no recursion).
    try:
        weekly_zones = detect(weekly_df, zcfg, logger=None, enable_weekly=False)
    except Exception as e:
        if logger:
            logger.warning(f"  Weekly confluence failed ({e}); features default.")
        return _defaults_for_all()

    # Weekly series used for trend + freshness (all backward-looking).
    w_index = weekly_df.index
    w_close = weekly_df["Close"].values
    w_atrw  = weekly_df["ATR"].values
    w_ma    = weekly_df[trend_ma].values if trend_ma in weekly_df.columns else w_close
    w_high  = weekly_df["High"].values
    w_low   = weekly_df["Low"].values

    has_wz    = not weekly_zones.empty
    confirmed = 0

    for z in zones:
        z.update(_WEEKLY_FEATURE_DEFAULTS)   # start from defaults each zone

        T      = z["formation_date"]
        a      = z["avg_atr"]
        mid    = z["midpoint"]
        d_top  = z["top"]
        d_bot  = z["bottom"]
        d_h    = d_top - d_bot
        d_type = z["type"]

        # ── Weekly trend alignment at the last completed weekly bar <= T ──
        prior = np.where(w_index <= T)[0]
        if prior.size:
            i = prior[-1]
            c, m, wa = float(w_close[i]), float(w_ma[i]), float(w_atrw[i])
            if wa > 0 and not (np.isnan(m) or np.isnan(c)):
                signed = (c - m) / wa
                z["weekly_trend_align"] = round(signed if d_type == "demand" else -signed, 4)

        # ── Confluence with a same-type weekly zone already formed by T ──
        if has_wz:
            cand = weekly_zones[
                (weekly_zones["type"] == d_type) &
                (weekly_zones["formation_date"] <= T)
            ]
            if len(cand):
                in_zone   = False
                best      = None
                best_dist = np.inf
                for w in cand.itertuples():
                    inter = min(d_top, w.top) - max(d_bot, w.bottom)
                    if d_h > 0 and inter > 0 and (inter / d_h) >= tol:
                        in_zone = True
                    if w.bottom <= mid <= w.top:
                        dist = 0.0
                    elif mid < w.bottom:
                        dist = w.bottom - mid
                    else:
                        dist = mid - w.top
                    if dist < best_dist:
                        best_dist, best = dist, w

                z["weekly_in_zone"]   = bool(in_zone)
                z["weekly_confirmed"] = bool(in_zone)
                if in_zone:
                    confirmed += 1
                if best is not None and a and a > 0:
                    z["weekly_dist_atr"]      = round(best_dist / a, 4)
                    z["weekly_zone_strength"] = round(float(best.strength), 4)
                    # Causal freshness: weekly touches in (weekly_formation, T].
                    wf     = best.formation_date
                    tmask  = (w_index > wf) & (w_index <= T)
                    touched = int(np.sum(
                        (w_low[tmask] <= best.top) & (w_high[tmask] >= best.bottom)
                    ))
                    z["weekly_zone_fresh"] = bool(touched == 0)

        # ── Combined 0..1 confluence score (causal) ──────────────────────
        ta         = z["weekly_trend_align"]
        trend_term = 0.5 if (ta is None or np.isnan(ta)) else float(np.clip(0.5 + ta / 4.0, 0.0, 1.0))
        prox       = z["weekly_dist_atr"]
        prox_term  = 0.0 if (prox is None or np.isnan(prox)) else float(np.clip(1.0 - prox / 2.0, 0.0, 1.0))
        ws         = z["weekly_zone_strength"]
        ws_term    = 0.0 if (ws is None or np.isnan(ws)) else float(ws)
        inzone_term = 1.0 if z["weekly_in_zone"]    else 0.0
        fresh_term  = 1.0 if z["weekly_zone_fresh"] else 0.0
        z["weekly_confluence_score"] = round(
            0.30 * inzone_term + 0.25 * prox_term + 0.20 * ws_term
            + 0.15 * trend_term + 0.10 * fresh_term, 4
        )

    if logger:
        logger.info(
            f"  Weekly zones: {len(weekly_zones)} | daily zones inside a weekly "
            f"zone: {confirmed}/{len(zones)}"
        )
    return zones


# ── Priority 3 helper: Base volume ratio ──────────────────────────────────────

def _get_base_volume_ratio(
    df: pd.DataFrame,
    base_start_idx: int,
    base_end_idx: int,
) -> float:
    """
    Compute the average VolumeRatio across all base candles.

    VolumeRatio = candle volume / 20-day volume MA (precomputed in preprocessor).

    Interpretation:
      < 0.70  → Very low volume in base. Institutional orders barely touched.
                Strong sign that unfilled orders remain intact. (POSITIVE)
      0.70–1.30 → Average volume. Neutral.
      > 1.30  → High volume in base. Orders may have been partially filled. (NEGATIVE)

    Returns NaN if VolumeRatio is not available in the DataFrame.
    """
    if "VolumeRatio" not in df.columns:
        return np.nan
    ratios = df.iloc[base_start_idx : base_end_idx + 1]["VolumeRatio"]
    valid  = ratios.replace(0, np.nan).dropna()
    return float(valid.mean()) if len(valid) > 0 else np.nan


# ── Priority 10 helper: trend alignment ───────────────────────────────────────

def _trend_score(
    df: pd.DataFrame,
    dep_idx: int,
    zone_type: str,
    zcfg: dict,
) -> float | None:
    """
    [review #10] Score how well a zone aligns with the prevailing trend,
    measured by the close's signed distance from a moving average at the
    departure (formation) candle. Demand is favoured in an uptrend, supply in
    a downtrend.

    Uses only data at the formation candle → no look-ahead. Returned as a
    SCORING input (and a ``trend_aligned`` flag in detect), never as a hard
    gate: counter-trend zones are down-weighted, not discarded.

    Returns a value in [0, 1] (0.5 = at the MA / neutral), or None when trend
    scoring is disabled or the MA column is unavailable — in which case the
    scorer omits the trend component and renormalises the remaining weights.
    """
    tcfg = zcfg.get("trend", {})
    if not tcfg.get("enabled", True):
        return None
    col = tcfg.get("ma_column", "EMA200")
    if col not in df.columns:
        return None

    row   = df.iloc[dep_idx]
    ma    = row.get(col, np.nan)
    close = row.get("Close", np.nan)
    atr   = row.get("ATR", np.nan)
    if pd.isna(ma) or pd.isna(close) or pd.isna(atr) or atr <= 0:
        return None

    dist_atr = (close - ma) / atr                    # >0 uptrend, <0 downtrend
    aligned  = dist_atr if zone_type == "demand" else -dist_atr
    full     = tcfg.get("full_align_atr", 2.0)       # ATR distance for full alignment
    return float(np.clip(0.5 + aligned / (2.0 * full), 0.0, 1.0))


# ── Step 6: Zone scoring ──────────────────────────────────────────────────────

def _score_zone(
    dep_body_atr: float,
    dep_leg_atr: float,
    dep_close_ratio: float,
    dep_volume_ratio: float,
    base_volume_ratio: float,
    base_range: float,
    avg_atr: float,
    arrival_move_atr: float,
    arrival_cleanliness: float,
    has_arrival: bool,
    trend_score: float | None,
    zcfg: dict,
) -> float:
    """
    Compute a 0.0–1.0 formation-time strength score.

    [review #9/#10] Components are assembled as (score, weight) pairs and only
    the AVAILABLE ones are kept, then their weights are renormalised to sum to
    1.0. This means:
      * untrusted/missing volume (e.g. spot indices, ``_volume_trust=False``)
        is dropped ENTIRELY rather than injected as a constant 0.5 (which would
        bias every score toward the middle), and
      * the trend component is included only when computable.

    Components (each normalised to [0, 1])
    --------------------------------------
    departure : 0.5 × leg displacement + 0.3 × first-candle body + 0.2 × close ratio
    base      : tightness (1.0 at zero width → 0.0 at the max allowed width)
    arrival   : 0.7 × momentum + 0.3 × cleanliness (the folded-in majority)
    volume    : departure-volume (high = good) + base-volume (low = good)
    trend     : alignment with the prevailing trend (optional, review #10)
    """
    w         = zcfg["scoring"]
    bv_cfg    = zcfg.get("base_volume", {})
    bv_low    = bv_cfg.get("low_threshold",  0.70)
    bv_high   = bv_cfg.get("high_threshold", 1.30)
    bv_weight = bv_cfg.get("weight", 0.10) if bv_cfg.get("enabled", True) else 0.0
    vol_trust = zcfg.get("_volume_trust", True)
    t_cfg     = zcfg.get("trend", {})
    t_weight  = t_cfg.get("weight", 0.10) if t_cfg.get("enabled", True) else 0.0

    # ── Departure: cumulative leg displacement is the primary evidence ────
    leg_target = zcfg.get("departure_leg_disp", 1.0) * 2.0
    dep_floor  = zcfg.get("departure_strength", 0.5)
    leg_norm   = min(dep_leg_atr / leg_target, 1.0) if leg_target > 0 else min(dep_leg_atr, 1.0)
    body_norm  = min(dep_body_atr / (dep_floor * 2.0), 1.0) if dep_floor > 0 else min(dep_body_atr, 1.0)
    dep_score  = 0.5 * leg_norm + 0.3 * body_norm + 0.2 * min(dep_close_ratio, 1.0)

    # ── Base tightness ────────────────────────────────────────────────────
    base_range_atr = base_range / avg_atr if avg_atr > 0 else 1.0
    base_denom     = zcfg.get("base_cluster_multiplier", zcfg["base_range_multiplier"])
    base_score     = max(1.0 - base_range_atr / base_denom, 0.0)

    # ── Arrival: momentum + cleanliness (the soft, folded-in majority) ────
    # Normalised against leg_in.min_move_atr (floored at 0.5 ATR so a 0.0
    # "no-gate" setting doesn't make every arrival score saturate at 1.0).
    arr_ref   = max(float(zcfg.get("leg_in", {}).get("min_move_atr", 0.0)), 0.5)
    arr_mom   = min(arrival_move_atr / (arr_ref * 2.0), 1.0)
    arr_score = 0.7 * arr_mom + 0.3 * min(max(arrival_cleanliness, 0.0), 1.0)

    components: list[tuple[float, float]] = [
        (dep_score,  w["departure_weight"]),
        (base_score, w["base_tightness_weight"]),
    ]
    # Arrival is included only when present (review #1: continuation/optional
    # bases may have no clear arrival leg); its weight is then renormalised away.
    if has_arrival:
        components.append((arr_score, w["arrival_weight"]))

    # ── Volume (only if trusted for this instrument) ──────────────────────
    if vol_trust:
        if pd.isna(dep_volume_ratio) or dep_volume_ratio <= 0:
            dep_vol_score = 0.5
        else:
            dep_vol_score = min(dep_volume_ratio / 3.0, 1.0)

        if pd.isna(base_volume_ratio) or base_volume_ratio <= 0:
            base_vol_score = 0.5
        elif base_volume_ratio <= bv_low:
            base_vol_score = 1.0
        elif base_volume_ratio >= bv_high:
            base_vol_score = 0.0
        else:
            base_vol_score = 1.0 - (base_volume_ratio - bv_low) / (bv_high - bv_low)

        dep_vol_weight = max(w["volume_weight"] - bv_weight, 0.0)
        components.append((dep_vol_score,  dep_vol_weight))
        components.append((base_vol_score, bv_weight))

    # ── Trend (only if available) ─────────────────────────────────────────
    if trend_score is not None and t_weight > 0:
        components.append((float(trend_score), t_weight))

    weight_sum = sum(wt for _, wt in components)
    if weight_sum <= 0:
        return 0.0
    total = sum(sc * wt for sc, wt in components) / weight_sum   # renormalise to 1.0
    return round(min(float(total), 1.0), 4)


def _quality_score(
    *,
    base_length: int,
    base_wick_frac: float,
    leg_out_clear: float,
    leg_out_disp: float,
    leg_in_velocity: float,
    arrival_cleanliness: float,
    has_arrival: bool,
    disp_base_ratio: float,
    leg_out_vol_exp: float,
    base_volume_ratio: float,
    trend_score: float | None,
    zcfg: dict,
) -> float:
    """
    Rebuilt 0..1 quality score (deep-analysis §7) — scaled to 0..100 as
    `quality_score`. Fixes the inverted legacy `strength`: it favours longer/
    wickier bases and MODERATE (peaked) leg-out displacement instead of raw
    departure magnitude, which was anti-predictive. Freshness is intentionally
    excluded (post-hoc / future). Available components are renormalised.
    """
    qw        = zcfg.get("quality_weights", {})
    vol_trust = zcfg.get("_volume_trust", True)

    # Base structure: length (1→0.4 … 4-5→1.0) + wickiness
    len_score  = {1: 0.4, 2: 0.7, 3: 0.8}.get(int(base_length), 1.0)
    wick_score = (0.5 if (base_wick_frac is None or np.isnan(base_wick_frac))
                  else float(np.clip((base_wick_frac - 0.3) / 0.5, 0.0, 1.0)))
    base_struct = 0.6 * len_score + 0.4 * wick_score

    # Leg-out: PEAKED at 0.8–1.5 ATR clearance (bigger is NOT better, §5)
    legout = _peak(leg_out_clear, 0.8, 1.5, 0.8)

    # Leg-in: velocity + cleanliness (only when an arrival leg exists)
    legin = (0.6 * float(np.clip(leg_in_velocity / 0.8, 0.0, 1.0))
             + 0.4 * float(np.clip(arrival_cleanliness, 0.0, 1.0))) if has_arrival else None

    # Displacement/base ratio: peaked near 1–1.5
    dispb = (_peak(disp_base_ratio, 1.0, 1.5, 1.5)
             if (disp_base_ratio is not None and not np.isnan(disp_base_ratio)) else None)

    # Volume (only if trusted): leg-out expansion + low base volume
    vol = None
    if vol_trust:
        vv = []
        if leg_out_vol_exp is not None and not np.isnan(leg_out_vol_exp):
            vv.append(min(leg_out_vol_exp / 1.5, 1.0))
        if base_volume_ratio is not None and not np.isnan(base_volume_ratio):
            vv.append(float(np.clip(1.0 - (base_volume_ratio - 0.7) / 0.6, 0.0, 1.0)))
        vol = float(np.mean(vv)) if vv else None

    comps = [(base_struct, qw.get("base_structure", 0.26)),
             (legout,      qw.get("leg_out", 0.24))]
    if legin is not None:
        comps.append((legin, qw.get("leg_in", 0.12)))
    if dispb is not None:
        comps.append((dispb, qw.get("disp_base", 0.10)))
    if trend_score is not None:
        comps.append((float(trend_score), qw.get("trend", 0.10)))
    if vol is not None:
        comps.append((vol, qw.get("volume", 0.08)))

    ws = sum(w for _, w in comps)
    base_q = (sum(s * w for s, w in comps) / ws) if ws > 0 else 0.0
    return round(min(base_q, 1.0), 4)


# ── Zone merging (deep-analysis §9) ───────────────────────────────────────────

def _merge_zones(zones: list[dict], zcfg: dict, logger=None) -> list[dict]:
    """
    Consolidate heavily-overlapping same-type zones that form close in time
    (~18% of raw zones are such duplicates). Keep the highest-`quality_score`
    member, record `merged_count`. Order-independent: greedy over formation time.
    """
    mcfg = zcfg.get("merge", {})
    if not mcfg.get("enabled", True) or not zones:
        for z in zones:
            z.setdefault("merged_count", 1)
        return zones
    tol  = mcfg.get("overlap_tolerance", 0.5)
    gap  = mcfg.get("time_gap", 60)

    kept: list[dict] = []
    for z in sorted(zones, key=lambda d: d["formation_idx"]):
        z["merged_count"] = 1
        hit = None
        for m in kept:
            if m["type"] != z["type"]:
                continue
            inter = min(m["top"], z["top"]) - max(m["bottom"], z["bottom"])
            h     = min(m["top"] - m["bottom"], z["top"] - z["bottom"])
            if h > 0 and (inter / h) >= tol and abs(z["formation_idx"] - m["formation_idx"]) <= gap:
                hit = m
                break
        if hit is None:
            kept.append(z)
        else:
            hit["merged_count"] += 1
            if z["quality_score"] > hit["quality_score"]:   # promote the stronger zone
                z["merged_count"] = hit["merged_count"]
                kept[kept.index(hit)] = z
    if logger:
        logger.info(f"  Zone merge: {len(zones)} → {len(kept)} ({len(zones) - len(kept)} merged away)")
    return kept


# ── Step 7: Zone status tracking ──────────────────────────────────────────────

def _track_zone_status(zones: list[dict], df: pd.DataFrame, zcfg: dict) -> list[dict]:
    """
    Walk forward through the dataframe after each zone's formation date
    and update its status, test count, and key event dates.

    Statuses:
        'active'    — no tests, no invalidation
        'tested'    — price has entered the zone at least once but not broken
        'broken'    — price closed beyond the zone boundary

    Invalidation modes (zone_config.yaml → invalidation_mode):
        'close'  — triggered when Close crosses the zone boundary
        'wick'   — triggered when High/Low crosses the zone boundary

    Look-ahead note: this function uses future data ONLY for tracking purposes
    (to update status). The zone itself was "formed" at formation_idx.
    When using zones as features for ML, always use only status and test_count
    as they were known at time t, not at the end of the dataset.
    """
    mode = zcfg.get("invalidation_mode", "close")

    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(df)

    for zone in zones:
        form_idx  = zone["formation_idx"]
        top       = zone["top"]
        bottom    = zone["bottom"]
        z_type    = zone["type"]

        test_count        = 0
        status            = "active"
        last_test_date    = None
        invalidation_date = None
        in_zone           = False  # track whether price is currently inside the zone

        for i in range(form_idx + 1, n):
            # ── Invalidation check ────────────────────────────
            if mode == "close":
                invalidated = (
                    (z_type == "demand" and closes[i] < bottom) or
                    (z_type == "supply" and closes[i] > top)
                )
            else:  # wick
                invalidated = (
                    (z_type == "demand" and lows[i]  < bottom) or
                    (z_type == "supply" and highs[i] > top)
                )

            if invalidated:
                status            = "broken"
                invalidation_date = df.index[i]
                break

            # ── Test / re-entry tracking ──────────────────────
            # Price is "in the zone" when the candle overlaps with [bottom, top]
            price_in_zone = lows[i] <= top and highs[i] >= bottom

            if price_in_zone and not in_zone:
                # Entering the zone for a new test
                test_count    += 1
                last_test_date = df.index[i]
                in_zone        = True
            elif not price_in_zone:
                in_zone = False

        if status == "active" and test_count > 0:
            status = "tested"

        zone["test_count"]        = test_count
        zone["status"]            = status
        zone["last_test_date"]    = last_test_date
        zone["invalidation_date"] = invalidation_date

    # ── Priority 2: Freshness score ────────────────────────────
    # Computed here (after test_count is finalised) for all zones.
    # adjusted_strength is computed later in detect() once weekly
    # confirmation is also available.
    decay_vals = zcfg.get("freshness", {}).get("decay_values", [1.0, 0.85, 0.65, 0.45])
    enabled    = zcfg.get("freshness", {}).get("enabled", True)
    for zone in zones:
        if enabled:
            tc  = int(zone["test_count"])
            idx = min(tc, len(decay_vals) - 1)
            zone["freshness_score"] = decay_vals[idx]
        else:
            zone["freshness_score"] = 1.0

    return zones


# ── Main detection function ───────────────────────────────────────────────────

def detect(
    df: pd.DataFrame,
    zcfg: dict,
    logger: logging.Logger | None = None,
    enable_weekly: bool = True,
    weekly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run the RBD/DBR zone detection pipeline on a preprocessed OHLCV DataFrame.

    Parameters
    ----------
    df            : Preprocessed DataFrame (Open, High, Low, Close, ATR, VolumeRatio).
    zcfg          : Zone configuration dict (zone_config.yaml → "zones:" key).
    logger        : Optional logger for progress messages.
    enable_weekly : When False, the weekly-confluence step is skipped. The
                    weekly sub-detection calls detect(..., enable_weekly=False)
                    so the pipeline recurses exactly ONE level.
    weekly_df     : Preprocessed WEEKLY OHLCV (fetched directly from Yahoo as
                    1wk bars via the data pipeline). Used to compute the causal
                    weekly-confluence features. If None and enable_weekly is
                    True, the detector falls back to resampling `df` to weekly
                    (with a warning) so it still works without a weekly file.

    Returns
    -------
    pd.DataFrame indexed by zone_id, sorted by formation_date (empty if none).

    No look-ahead bias
    ------------------
    A zone is "formed" at the close of its first leg-out (departure) candle.
    `strength` and `strength_pit` use ONLY formation-time information.
    `adjusted_strength_posthoc` additionally folds in freshness (which depends
    on FUTURE test_count) and is therefore ANALYSIS / RANKING ONLY — never use
    it as an ML feature at time t (see review #6).
    """
    if logger:
        logger.info(f"Running zone detection on {len(df)} candles...")

    # Verify required columns are present
    required = {"Open", "High", "Low", "Close", "ATR"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Run preprocessor.py first."
        )

    # ── [review #9] Decide whether volume is trustworthy here ──
    # Spot indices (e.g. ^NSEI) report zero/NaN volume; if too many candles are
    # affected, drop volume scoring (the scorer renormalises the other weights).
    max_zero_frac = zcfg.get("volume", {}).get("max_zero_fraction", 0.01)
    if "Volume" in df.columns:
        zero_frac = float((df["Volume"].fillna(0) <= 0).mean())
    else:
        zero_frac = 1.0
    volume_trust = zero_frac <= max_zero_frac
    if logger and not volume_trust:
        logger.info(
            f"  Volume untrusted ({zero_frac:.1%} zero/NaN candles > "
            f"{max_zero_frac:.1%}); volume scoring disabled, weights renormalised."
        )
    zcfg = {**zcfg, "_volume_trust": volume_trust}

    # ── Step 1 & 2: Find all base groups ──────────────────────
    bases = _find_bases(df, zcfg)
    if logger:
        logger.info(f"  Candidate bases found: {len(bases)}")

    # Pull numpy arrays once (deep-analysis §11 — avoid per-row df.iloc).
    O  = df["Open"].values;  H = df["High"].values;  L = df["Low"].values
    C  = df["Close"].values; A = df["ATR"].values
    VR = df["VolumeRatio"].values if "VolumeRatio" in df.columns else np.full(len(df), np.nan)

    dep_floor    = zcfg["departure_strength"]
    dep_cr       = zcfg["departure_close_ratio"]
    leg_max      = zcfg.get("departure_leg_max", 12)
    leg_disp     = zcfg.get("departure_leg_disp", 0.6)
    pullback     = zcfg.get("leg_pullback_atr", 0.6)
    licfg         = zcfg.get("leg_in", {})
    li_enabled    = bool(licfg.get("enabled", True))
    li_lookback   = int(licfg.get("lookback", 5))
    li_min_move   = float(licfg.get("min_move_atr",
                          zcfg.get("arrival_min_move", 0.0)))  # legacy fallback
    arrival_mode  = zcfg.get("arrival_mode", "any")
    require_origin = zcfg.get("arrival_require_origin", False)
    require_bc    = zcfg.get("require_base_containment", True)
    min_q         = zcfg.get("min_quality_threshold", 0.0)
    nrows        = len(df)

    zones    = []
    zone_num = 1

    for base in bases:
        base_start = base["start_idx"]
        base_end   = base["end_idx"]
        avg_atr    = base["avg_atr"]
        top        = base["top"]
        bottom     = base["bottom"]
        base_range = top - bottom
        if avg_atr <= 0:
            continue

        # ── Steps 3 & 4: departure sets polarity (dynamic leg-out) ──
        dep_demand, info_d = _check_departure(O, H, L, C, VR, base_end, "demand",
                                              top, bottom, avg_atr, nrows,
                                              dep_floor, dep_cr, leg_max, leg_disp, pullback)
        dep_supply, info_s = _check_departure(O, H, L, C, VR, base_end, "supply",
                                              top, bottom, avg_atr, nrows,
                                              dep_floor, dep_cr, leg_max, leg_disp, pullback)
        if dep_demand:
            zone_type, dep_info, dep_dir = "demand", info_d, "up"
        elif dep_supply:
            zone_type, dep_info, dep_dir = "supply", info_s, "down"
        else:
            continue

        # ── Arrival: fixed-window leg-in + origin check ──
        # The leg-in is ALWAYS measured (the structure label and the CSV
        # feature columns keep the same format either way). leg_in.enabled
        # only controls whether it GATES zones and contributes to quality:
        #   enabled: true  → arrival_mode gate + leg-in quality component
        #   enabled: false → measured for labelling only (no gate, no score)
        has_arr, arr = _check_arrival(O, H, L, C, VR, base_start, dep_dir, top, bottom,
                                      avg_atr, nrows, li_lookback,
                                      li_min_move if li_enabled else 0.0)
        arr_dir = arr["arr_dir"] if has_arr else None
        if li_enabled and arrival_mode == "reversal":
            opposite = has_arr and (
                (dep_dir == "up" and arr_dir == "down") or
                (dep_dir == "down" and arr_dir == "up"))
            if not opposite:
                continue
            if require_origin and not arr["origin_ok"]:
                continue
        elif li_enabled and arrival_mode == "any":
            if not has_arr:
                continue

        # Structure label is ALWAYS the full 3-letter form (DBR/RBD/RBR/DBD) —
        # downstream consumers (web app) rely on this format. In the rare case
        # the leg-in nets to exactly zero, assume the classic reversal arrival.
        d_char = "R" if dep_dir == "up" else "D"
        if has_arr:
            structure = f"{'R' if arr_dir == 'up' else 'D'}B{d_char}"
        else:
            structure = ("D" if d_char == "R" else "R") + "B" + d_char

        # ── Base-containment gate (direction-aware) ─────────────
        # The base must sit inside the imbalance. The departure-side check is
        # the same for all structures (first leg-out close clears the base);
        # the arrival-side check depends on which way the leg-in came:
        #   DBR (demand, reversal):     base HIGH < open[prev]  (arrival dropped in from above)
        #   RBD (supply, reversal):     base LOW  > open[prev]  (arrival rallied in from below)
        #   RBR (demand, continuation): base LOW  > open[prev]  (arrival rallied in from below)
        #   DBD (supply, continuation): base HIGH < open[prev]  (arrival dropped in from above)
        if require_bc:
            prev_idx  = base_start - 1
            first_dep = dep_info["first_idx"]            # = base_end + 1
            if prev_idx < 0 or first_dep >= nrows:
                continue
            # containment_price (zone_config.yaml):
            #   "body" → contain the base BODIES only (wicks may poke out).
            #            Sweep wicks are REJECTION, not acceptance — a long
            #            sweep wick should not disqualify an otherwise clean
            #            base (e.g. SONATSOFTW Jun-2022 quarterly demand).
            #   "wick" → strict: the full candle range must be contained.
            bb_c = slice(base_start, base_end + 1)
            if zcfg.get("containment_price", "wick") == "body":
                c_top = float(np.max(np.maximum(O[bb_c], C[bb_c])))
                c_bot = float(np.min(np.minimum(O[bb_c], C[bb_c])))
            else:
                c_top, c_bot = top, bottom
            # Departure side: first leg-out close must clear the base.
            if zone_type == "demand":
                dep_ok = c_top < C[first_dep]
            else:
                dep_ok = c_bot > C[first_dep]
            # Arrival side: base must sit beyond the arrival candle's open
            # in the arrival's direction of travel.
            if arr_dir is None:
                arr_ok = True            # bare base→leg-out (arrival_mode "optional")
            elif arr_dir == "down":
                arr_ok = c_top < O[prev_idx]
            else:                        # arr_dir == "up"
                arr_ok = c_bot > O[prev_idx]
            if not (dep_ok and arr_ok):
                continue

        dep_idx        = dep_info["dep_idx"]
        base_vol_ratio = _get_base_volume_ratio(df, base_start, base_end)
        trend_sc       = _trend_score(df, dep_idx, zone_type, zcfg)

        # ── Base wickiness + hybrid distal/proximal boundaries (§6) ──
        bb        = slice(base_start, base_end + 1)
        body_mean = float(np.mean(np.abs(C[bb] - O[bb])))
        rng_mean  = float(np.mean(H[bb] - L[bb]))
        wick_frac = (1.0 - body_mean / rng_mean) if rng_mean > 0 else np.nan
        body_top  = float(np.max(np.maximum(O[bb], C[bb])))
        body_bot  = float(np.min(np.minimum(O[bb], C[bb])))
        if zone_type == "demand":
            proximal, distal = body_top, bottom     # entry near body top; stop at the low
        else:
            proximal, distal = body_bot, top        # entry near body bottom; stop at the high

        # ── Per-leg metrics (§5) ──
        leg_out_clear = dep_info["leg_out_clear"]
        leg_out_disp  = dep_info["leg_out_disp"]
        disp_base     = (leg_out_disp * avg_atr / base_range) if base_range > 0 else np.nan
        leg_in_disp = arr["leg_in_disp"]     if has_arr else np.nan
        leg_in_cnd  = arr["leg_in_candles"]  if has_arr else 0
        leg_in_vel  = arr["leg_in_velocity"] if has_arr else 0.0
        arr_clean   = arr["arrival_cleanliness"] if has_arr else 0.0

        # ── Legacy strength (kept) + NEW quality score (§7) ──
        strength = _score_zone(
            dep_body_atr=dep_info["dep_body_atr"], dep_leg_atr=leg_out_clear,
            dep_close_ratio=dep_info["dep_close_ratio"], dep_volume_ratio=dep_info["dep_volume_ratio"],
            base_volume_ratio=base_vol_ratio, base_range=base_range, avg_atr=avg_atr,
            arrival_move_atr=leg_in_disp if has_arr else 0.0, arrival_cleanliness=arr_clean,
            has_arrival=(has_arr and li_enabled), trend_score=trend_sc, zcfg=zcfg)
        quality01 = _quality_score(
            base_length=base["length"], base_wick_frac=wick_frac, leg_out_clear=leg_out_clear,
            leg_out_disp=leg_out_disp, leg_in_velocity=leg_in_vel, arrival_cleanliness=arr_clean,
            has_arrival=(has_arr and li_enabled),  # leg-in scored only when enabled
            disp_base_ratio=disp_base, leg_out_vol_exp=dep_info["leg_out_vol_exp"],
            base_volume_ratio=base_vol_ratio, trend_score=trend_sc, zcfg=zcfg)

        if quality01 < min_q:
            continue

        zones.append({
            "zone_id":                  f"Z{zone_num:04d}",
            "type":                     zone_type,
            "structure":                structure,
            "top":                      round(top, 4),
            "bottom":                   round(bottom, 4),
            "midpoint":                 round((top + bottom) / 2, 4),
            "proximal":                 round(proximal, 4),
            "distal":                   round(distal, 4),
            "width":                    round(base_range, 4),
            "width_atr":                round(base_range / avg_atr, 4),
            "formation_date":           df.index[dep_idx],
            "formation_idx":            dep_idx,
            "base_start_date":          df.index[base_start],
            "base_end_date":            df.index[base_end],
            "base_length":              base["length"],
            "base_wick_frac":           round(wick_frac, 4) if not np.isnan(wick_frac) else np.nan,
            "avg_atr":                  round(avg_atr, 4),
            "departure_body_atr":       dep_info["dep_body_atr"],
            "departure_close_ratio":    dep_info["dep_close_ratio"],
            "leg_out_clear_atr":        leg_out_clear,
            "leg_out_disp_atr":         leg_out_disp,
            "leg_out_candles":          dep_info["leg_out_candles"],
            "leg_out_velocity":         dep_info["leg_out_velocity"],
            "leg_out_vol_exp":          dep_info["leg_out_vol_exp"],
            "leg_in_disp_atr":          round(float(leg_in_disp), 4) if has_arr else np.nan,
            "leg_in_candles":           leg_in_cnd if has_arr else np.nan,
            "leg_in_velocity":          round(float(leg_in_vel), 4) if has_arr else np.nan,
            "disp_base_ratio":          round(float(disp_base), 4) if not np.isnan(disp_base) else np.nan,
            "arrival_cleanliness":      arr_clean if has_arr else np.nan,
            "departure_volume_ratio":   dep_info["dep_volume_ratio"],
            "base_volume_ratio":        round(float(base_vol_ratio), 4)
                                        if not np.isnan(base_vol_ratio) else np.nan,
            "trend_score":              round(float(trend_sc), 4) if trend_sc is not None else np.nan,
            "trend_aligned":            bool(trend_sc >= 0.5) if trend_sc is not None else False,
            "strength":                 strength,                 # legacy (deprecated for ranking)
            "quality_score":            round(quality01 * 100, 2),  # NEW primary 0–100 (§7)
            # Weekly-confluence features — populated by _add_weekly_confluence
            "weekly_trend_align":        np.nan,
            "weekly_in_zone":            False,
            "weekly_dist_atr":           np.nan,
            "weekly_zone_strength":      np.nan,
            "weekly_zone_fresh":         False,
            "weekly_confluence_score":   0.0,
            "weekly_confirmed":          False,
            # Post-status columns
            "freshness_score":           1.0,
            "strength_pit":              round(quality01, 4),
            "adjusted_strength_posthoc": round(quality01, 4),
            "merged_count":              1,
            # Status fields
            "test_count":               0,
            "status":                   "active",
            "last_test_date":           None,
            "invalidation_date":        None,
        })
        zone_num += 1

    if logger:
        logger.info(f"  Zones detected (pre-status): {len(zones)}")

    if not zones:
        if logger:
            logger.warning("  No zones detected. Try relaxing parameters in zone_config.yaml.")
        return pd.DataFrame()

    # ── Zone merging (deep-analysis §9): consolidate overlapping duplicates,
    # keeping the highest-quality member (re-IDs are stable by formation order). ──
    zones = _merge_zones(zones, zcfg, logger=logger)

    # ── Step 7: Track zone status over time ────────────────────
    # Also computes freshness_score for each zone (Priority 2).
    zones = _track_zone_status(zones, df, zcfg)

    # ── Weekly-timeframe confluence features (causal) ──────────
    # Uses weekly data fetched from Yahoo (1wk). Falls back to resampling the
    # daily df only if no weekly_df was supplied, so the detector still runs.
    if enable_weekly:
        wdf = weekly_df
        if wdf is None:
            if logger:
                logger.info("  No weekly_df supplied; resampling daily→weekly as "
                            "fallback (prefer fetching 1wk via data_pipeline.py).")
            try:
                wdf = _resample_to_weekly(df)
            except Exception:
                wdf = None
        zones = _add_weekly_confluence(zones, df, wdf, zcfg, logger=logger)

    # ── PIT / post-hoc columns (review #6: quarantine the look-ahead) ─
    #   quality_score (0–100)     : NEW formation-time quality (§7), leak-free → primary ranker
    #   strength_pit              : quality(0–1) + CAUSAL weekly-confluence bonus → ML-safe
    #   adjusted_strength_posthoc : strength_pit × freshness (freshness uses
    #                               FUTURE test_count)                          → ANALYSIS ONLY
    #   strength                  : legacy score, kept for reference (do NOT rank on it)
    wc_bonus = zcfg.get("weekly_confirmation", {}).get("strength_bonus", 0.08)
    for z in zones:
        q01   = z["quality_score"] / 100.0
        bonus = wc_bonus * float(z.get("weekly_confluence_score", 0.0))
        z["strength_pit"] = round(min(q01 + bonus, 1.0), 4)
        z["adjusted_strength_posthoc"] = round(
            min(z["strength_pit"] * z["freshness_score"], 1.0), 4
        )

    result = (
        pd.DataFrame(zones)
        .set_index("zone_id")
        .sort_values("formation_date")
    )

    # ── Strict DBR/RBD enforcement ─────────────────────────────
    # In reversal mode only the two reversal structures are ever produced
    # (arrival opposite to the leg-out). This is a hard safety net so a stray
    # config change can never leak RBR/DBD continuation zones into the output.
    # (Skipped when leg_in.enabled is false — no arrival info exists then.)
    if (zcfg.get("leg_in", {}).get("enabled", True)
            and zcfg.get("arrival_mode", "any") == "reversal"
            and "structure" in result.columns):
        keep = result["structure"].isin(["DBR", "RBD"])
        if (~keep).any():
            if logger:
                logger.warning(
                    f"  Strict reversal mode: dropped {int((~keep).sum())} "
                    "non-DBR/RBD zone(s)."
                )
            result = result[keep]

    if logger:
        _log_summary(result, logger)

    return result


def _log_summary(zones: pd.DataFrame, logger: logging.Logger):
    """Log a compact summary of detection results."""
    n_demand     = (zones["type"] == "demand").sum()
    n_supply     = (zones["type"] == "supply").sum()
    n_active     = (zones["status"] == "active").sum()
    n_tested     = (zones["status"] == "tested").sum()
    n_broken     = (zones["status"] == "broken").sum()
    n_weekly     = zones.get("weekly_confirmed", pd.Series(dtype=bool)).sum()
    n_wfresh     = zones.get("weekly_zone_fresh", pd.Series(dtype=bool)).sum()
    avg_conf     = zones.get("weekly_confluence_score", pd.Series(dtype=float)).mean()
    avg_strength = zones["strength"].mean()
    avg_pit      = zones.get("strength_pit", zones["strength"]).mean()
    avg_adjusted = zones.get("adjusted_strength_posthoc", zones["strength"]).mean()

    logger.info("  ─── Zone Detection Summary ───────────────────")
    logger.info(f"  Total zones          : {len(zones)}")
    logger.info(f"  Demand / Supply      : {n_demand} / {n_supply}")
    logger.info(f"  Active               : {n_active}")
    logger.info(f"  Tested               : {n_tested}")
    logger.info(f"  Broken               : {n_broken}")
    logger.info(f"  In weekly zone       : {n_weekly} (fresh weekly: {n_wfresh})")
    logger.info(f"  Avg weekly confluence: {avg_conf:.3f}")
    logger.info(f"  Avg raw strength     : {avg_strength:.3f}")
    logger.info(f"  Avg strength (PIT)   : {avg_pit:.3f}")
    logger.info(f"  Avg adjusted (posthoc): {avg_adjusted:.3f}")
    logger.info(f"  Date range           : {zones['formation_date'].min().date()} → "
                f"{zones['formation_date'].max().date()}")
    logger.info("  ──────────────────────────────────────────────")


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_detection(symbol: str, main_cfg: dict, zcfg: dict, logger: logging.Logger) -> bool:
    """
    Load processed data for symbol, run detection, save zones.
    Returns True on success, False on failure.
    """
    processed_dir = PROJECT_ROOT / main_cfg["data"]["processed_dir"]
    zones_dir     = PROJECT_ROOT / main_cfg["data"]["zones_dir"]

    try:
        df = load_processed(symbol, processed_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    logger.info(f"[{symbol}] Loaded {len(df)} rows from processed data.")

    # Load the weekly (1wk) processed data fetched by data_pipeline.py.
    # If absent, detect() falls back to resampling the daily data.
    try:
        weekly_df = load_processed(symbol, processed_dir, suffix="_weekly")
        logger.info(f"[{symbol}] Loaded {len(weekly_df)} weekly rows for confluence.")
    except FileNotFoundError:
        logger.warning(
            f"[{symbol}] No weekly processed file — run data_pipeline.py to fetch "
            "1wk data. Falling back to resampling for weekly confluence."
        )
        weekly_df = None

    zones = detect(df, zcfg, logger=logger, weekly_df=weekly_df)

    if zones.empty:
        logger.warning(f"[{symbol}] No zones detected.")
        return False

    save_zones(zones, symbol, zones_dir, logger=logger)
    return True


def main():
    import argparse
    main_cfg, zcfg = load_configs()
    logger = setup_logging(main_cfg)

    parser = argparse.ArgumentParser(
        description="ZoneTrend — detect RBD/DBR supply and demand zones"
    )
    parser.add_argument(
        "--symbol",
        default=main_cfg["data"]["symbol"],
        help="Yahoo Finance ticker (default: value in config.yaml)",
    )
    args = parser.parse_args()

    logger.info(f"ZoneTrend zone_detector | symbol={args.symbol}")

    ok = run_detection(args.symbol, main_cfg, zcfg, logger)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
