"""
zone_detector.py
================
Detects supply and demand zones using the RBD and DBR structural patterns.

Theory
------
Price moves are driven by imbalances between buyers and sellers.
When a strong directional move (rally or drop) originates from a tight
consolidation (the "base"), it means one side overwhelmed the other at
that price area. Unfilled institutional orders remain in the base.
When price returns to the base, those orders re-activate — creating
reliable support (demand) or resistance (supply).

The two patterns:

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

Step 3  Check the departure candle (candle immediately after base end):
        DBR: must be strongly bullish —
             body  >= departure_strength × ATR
             close ratio >= departure_close_ratio  (closed high within its range)
        RBD: must be strongly bearish —
             body  >= departure_strength × ATR
             close ratio >= departure_close_ratio  (closed low within its range)

Step 4  Check the arrival leg (candles before the base):
        DBR: net close move over arrival_lookback candles must be negative
             and its magnitude >= arrival_min_move × ATR
        RBD: net close move must be positive and >= arrival_min_move × ATR

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
        After max_test_count tests, status becomes 'consumed'.

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
    avg_atr, departure_body_atr, departure_close_ratio, arrival_move_atr,
    strength, status, test_count, last_test_date, invalidation_date

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


def load_processed(symbol: str, processed_dir: Path) -> pd.DataFrame:
    """Load a preprocessed OHLCV CSV for a single symbol."""
    path = processed_dir / (_symbol_to_stem(symbol) + ".csv")
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
    Scan the full dataframe and collect all candidate base groups.

    A base group is a maximal run of consecutive base candles,
    subject to min/max length constraints and a total-range check.

    Returns
    -------
    List of dicts, each with:
        start_idx, end_idx, length, top, bottom, avg_atr
    """
    range_mult   = zcfg["base_range_multiplier"]
    body_mult    = zcfg["base_body_multiplier"]
    min_len      = zcfg["min_base_length"]
    max_len      = zcfg["max_base_length"]

    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    atrs   = df["ATR"].values

    n      = len(df)
    bases  = []
    i      = 0

    while i < n:
        # Check if candle i can start a base
        if not _is_base_candle(
            highs[i], lows[i], opens[i], closes[i], atrs[i],
            range_mult, body_mult,
        ):
            i += 1
            continue

        # Extend forward while candles remain base-like
        base_indices = [i]
        j = i + 1
        while j < n and len(base_indices) < max_len:
            if _is_base_candle(
                highs[j], lows[j], opens[j], closes[j], atrs[j],
                range_mult, body_mult,
            ):
                base_indices.append(j)
                j += 1
            else:
                break

        if len(base_indices) >= min_len:
            top        = float(np.max(highs[base_indices]))
            bottom     = float(np.min(lows[base_indices]))
            avg_atr    = float(np.nanmean(atrs[base_indices]))
            total_range = top - bottom

            # Total base range must still be within ATR threshold
            if avg_atr > 0 and total_range <= range_mult * avg_atr:
                bases.append({
                    "start_idx": base_indices[0],
                    "end_idx":   base_indices[-1],
                    "length":    len(base_indices),
                    "top":       top,
                    "bottom":    bottom,
                    "avg_atr":   avg_atr,
                })

            # Advance past this base to prevent overlapping detections
            i = j
        else:
            i += 1

    return bases


# ── Step 3: Departure check ───────────────────────────────────────────────────

def _check_departure(
    df: pd.DataFrame,
    base_end_idx: int,
    zone_type: str,
    avg_atr: float,
    dep_strength: float,
    dep_close_ratio: float,
) -> tuple[bool, dict | None]:
    """
    Examine the candle immediately after the base and decide whether it
    constitutes a valid departure for the given zone type.

    For a demand zone (DBR): departure must be strongly bullish.
    For a supply zone (RBD): departure must be strongly bearish.

    Checks:
      1. Body size  >= departure_strength × ATR
      2. Close ratio >= departure_close_ratio
         (bullish: (Close-Low)/(High-Low), bearish: (High-Close)/(High-Low))
         Ensures the candle CLOSED strong, not just had a wide range.

    Returns
    -------
    (is_valid, departure_info_dict | None)
    """
    dep_idx = base_end_idx + 1
    if dep_idx >= len(df):
        return False, None

    row   = df.iloc[dep_idx]
    high  = float(row["High"])
    low   = float(row["Low"])
    open_ = float(row["Open"])
    close = float(row["Close"])
    body  = abs(close - open_)
    rng   = high - low

    # Avoid division by zero on doji-like departure candle
    if rng < 1e-10:
        return False, None

    body_atr_ratio = body / avg_atr if avg_atr > 0 else 0.0

    if zone_type == "demand":
        # Bullish departure: Close must be above Open and strong
        is_correct_direction = close >= open_
        close_ratio          = (close - low) / rng
    else:
        # Bearish departure: Close must be below Open and strong
        is_correct_direction = close < open_
        close_ratio          = (high - close) / rng

    is_valid = (
        is_correct_direction
        and body_atr_ratio  >= dep_strength
        and close_ratio     >= dep_close_ratio
    )

    if not is_valid:
        return False, None

    return True, {
        "dep_idx":          dep_idx,
        "dep_body_atr":     round(body_atr_ratio, 4),
        "dep_close_ratio":  round(close_ratio, 4),
        "dep_volume_ratio": float(row.get("VolumeRatio", np.nan)),
    }


# ── Step 4: Arrival check ─────────────────────────────────────────────────────

def _check_arrival(
    df: pd.DataFrame,
    base_start_idx: int,
    zone_type: str,
    avg_atr: float,
    lookback: int,
    min_move: float,
) -> tuple[bool, float]:
    """
    Examine the candles before the base to confirm a directional arrival leg.

    For a demand zone (DBR): price must have been falling into the base.
    For a supply zone (RBD): price must have been rising into the base.

    The net close-to-close move over `lookback` candles must exceed
    arrival_min_move × ATR in the required direction.

    Returns
    -------
    (is_valid, arrival_move_in_atr_units)
    """
    start_idx = max(0, base_start_idx - lookback)
    end_idx   = base_start_idx          # first base candle is excluded

    if end_idx - start_idx < 1:
        return False, 0.0

    close_start = float(df.iloc[start_idx]["Close"])
    close_end   = float(df.iloc[end_idx - 1]["Close"])
    net_move    = close_end - close_start
    net_move_atr = abs(net_move) / avg_atr if avg_atr > 0 else 0.0

    if zone_type == "demand":
        # Arrival must be a downward leg (price falling into the base)
        is_valid = (net_move < 0) and (net_move_atr >= min_move)
    else:
        # Arrival must be an upward leg (price rising into the base)
        is_valid = (net_move > 0) and (net_move_atr >= min_move)

    return is_valid, round(net_move_atr, 4)


# ── Step 6: Zone scoring ──────────────────────────────────────────────────────

def _score_zone(
    dep_body_atr: float,
    dep_close_ratio: float,
    dep_volume_ratio: float,
    base_range: float,
    avg_atr: float,
    arrival_move_atr: float,
    zcfg: dict,
) -> float:
    """
    Compute a zone strength score between 0.0 and 1.0.

    Four components, each normalised to [0, 1]:

    departure_score
        How strong was the departure relative to the threshold?
        dep_body_atr / departure_strength, capped at 1 when body = 2× threshold.

    base_tightness_score
        How tight was the base?
        1.0 when base range = 0, 0.0 when base range = base_range_multiplier × ATR.

    arrival_score
        How strong was the arrival leg?
        arrival_move_atr / (arrival_min_move × 2), capped at 1.

    volume_score
        How elevated was volume on the departure candle?
        dep_volume_ratio / 3.0, capped at 1. Neutral (0.5) when unknown.

    The final score is a weighted sum using weights from zone_config.yaml.
    """
    weights = zcfg["scoring"]

    # Departure strength (normalised against twice the threshold)
    dep_score = min(dep_body_atr / (zcfg["departure_strength"] * 2.0), 1.0)

    # Departure close ratio contribution (already in [0,1])
    # Blend dep_score with close_ratio for a single departure component
    dep_score = 0.7 * dep_score + 0.3 * min(dep_close_ratio, 1.0)

    # Base tightness (tighter base → higher score)
    base_range_atr = base_range / avg_atr if avg_atr > 0 else 1.0
    base_score     = max(1.0 - base_range_atr / zcfg["base_range_multiplier"], 0.0)

    # Arrival momentum
    arrival_score = min(arrival_move_atr / (zcfg["arrival_min_move"] * 2.0), 1.0)

    # Volume
    if pd.isna(dep_volume_ratio) or dep_volume_ratio <= 0:
        vol_score = 0.5  # neutral when no volume data
    else:
        vol_score = min(dep_volume_ratio / 3.0, 1.0)

    total = (
        weights["departure_weight"]      * dep_score     +
        weights["base_tightness_weight"] * base_score    +
        weights["arrival_weight"]        * arrival_score +
        weights["volume_weight"]         * vol_score
    )
    return round(float(total), 4)


# ── Step 7: Zone status tracking ──────────────────────────────────────────────

def _track_zone_status(zones: list[dict], df: pd.DataFrame, zcfg: dict) -> list[dict]:
    """
    Walk forward through the dataframe after each zone's formation date
    and update its status, test count, and key event dates.

    Statuses:
        'active'    — no tests, no invalidation
        'tested'    — price has entered the zone at least once but not invalidated
        'consumed'  — test_count >= max_test_count (zone likely exhausted)
        'invalid'   — price closed beyond the zone boundary

    Invalidation modes (zone_config.yaml → invalidation_mode):
        'close'  — triggered when Close crosses the zone boundary
        'wick'   — triggered when High/Low crosses the zone boundary

    Look-ahead note: this function uses future data ONLY for tracking purposes
    (to update status). The zone itself was "formed" at formation_idx.
    When using zones as features for ML, always use only status and test_count
    as they were known at time t, not at the end of the dataset.
    """
    mode      = zcfg.get("invalidation_mode", "close")
    max_tests = zcfg["max_test_count"]

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
                status            = "invalid"
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

                if test_count >= max_tests:
                    status = "consumed"
                    break
            elif not price_in_zone:
                in_zone = False

        if status == "active" and test_count > 0:
            status = "tested"

        zone["test_count"]        = test_count
        zone["status"]            = status
        zone["last_test_date"]    = last_test_date
        zone["invalidation_date"] = invalidation_date

    return zones


# ── Main detection function ───────────────────────────────────────────────────

def detect(
    df: pd.DataFrame,
    zcfg: dict,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Run the full RBD/DBR zone detection pipeline on a preprocessed OHLCV DataFrame.

    Parameters
    ----------
    df    : Preprocessed DataFrame from preprocessor.py.
            Must contain: Open, High, Low, Close, ATR, VolumeRatio.
    zcfg  : Zone configuration dict (from zone_config.yaml → "zones:" key).
    logger: Optional logger for progress messages.

    Returns
    -------
    pd.DataFrame indexed by zone_id, sorted by formation_date.
    Empty DataFrame if no zones are detected.

    No look-ahead bias:
    A zone is considered "formed" at the close of its departure candle.
    All status tracking uses forward data only for post-hoc analysis.
    When using zones as ML features, use the time-indexed version of
    this output — only include zone data that was available at time t.
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

    # ── Step 1 & 2: Find all base groups ──────────────────────
    bases = _find_bases(df, zcfg)
    if logger:
        logger.info(f"  Candidate bases found: {len(bases)}")

    zones    = []
    zone_num = 1

    for base in bases:
        base_start = base["start_idx"]
        base_end   = base["end_idx"]
        avg_atr    = base["avg_atr"]
        top        = base["top"]
        bottom     = base["bottom"]
        base_range = top - bottom

        # ── Steps 3 & 4: Try both zone types ──────────────────
        # Each base can only produce one zone (the departure candle
        # is either bullish or bearish, not both), so trying both
        # ensures we capture whichever direction applies.
        for zone_type in ("demand", "supply"):

            # Step 3 — departure check
            dep_valid, dep_info = _check_departure(
                df, base_end, zone_type, avg_atr,
                zcfg["departure_strength"],
                zcfg["departure_close_ratio"],
            )
            if not dep_valid:
                continue

            # Step 4 — arrival check
            arr_valid, arr_move_atr = _check_arrival(
                df, base_start, zone_type, avg_atr,
                zcfg["arrival_lookback"],
                zcfg["arrival_min_move"],
            )
            if not arr_valid:
                continue

            dep_idx = dep_info["dep_idx"]

            # ── Step 6: Score ──────────────────────────────────
            strength = _score_zone(
                dep_body_atr     = dep_info["dep_body_atr"],
                dep_close_ratio  = dep_info["dep_close_ratio"],
                dep_volume_ratio = dep_info["dep_volume_ratio"],
                base_range       = base_range,
                avg_atr          = avg_atr,
                arrival_move_atr = arr_move_atr,
                zcfg             = zcfg,
            )

            # Apply minimum strength filter
            if strength < zcfg["min_strength_threshold"]:
                continue

            zones.append({
                "zone_id":           f"Z{zone_num:04d}",
                "type":              zone_type,
                "top":               round(top, 4),
                "bottom":            round(bottom, 4),
                "midpoint":          round((top + bottom) / 2, 4),
                "width":             round(base_range, 4),
                "width_atr":         round(base_range / avg_atr, 4),
                # Formation date = close of departure candle (zone becomes "known")
                "formation_date":    df.index[dep_idx],
                "formation_idx":     dep_idx,
                "base_start_date":   df.index[base_start],
                "base_end_date":     df.index[base_end],
                "base_length":       base["length"],
                "avg_atr":           round(avg_atr, 4),
                "departure_body_atr":   dep_info["dep_body_atr"],
                "departure_close_ratio": dep_info["dep_close_ratio"],
                "departure_volume_ratio": dep_info["dep_volume_ratio"],
                "arrival_move_atr":  arr_move_atr,
                "strength":          strength,
                # Status fields — populated in step 7
                "test_count":        0,
                "status":            "active",
                "last_test_date":    None,
                "invalidation_date": None,
            })
            zone_num += 1

    if logger:
        logger.info(f"  Zones before strength filter: {zone_num - 1}")
        logger.info(f"  Zones after  strength filter: {len(zones)}")

    if not zones:
        if logger:
            logger.warning("  No zones detected. Try relaxing parameters in zone_config.yaml.")
        return pd.DataFrame()

    # ── Step 7: Track zone status over time ────────────────────
    zones = _track_zone_status(zones, df, zcfg)

    result = (
        pd.DataFrame(zones)
        .set_index("zone_id")
        .sort_values("formation_date")
    )

    if logger:
        _log_summary(result, logger)

    return result


def _log_summary(zones: pd.DataFrame, logger: logging.Logger):
    """Log a compact summary of detection results."""
    n_demand    = (zones["type"] == "demand").sum()
    n_supply    = (zones["type"] == "supply").sum()
    n_active    = (zones["status"] == "active").sum()
    n_tested    = (zones["status"] == "tested").sum()
    n_consumed  = (zones["status"] == "consumed").sum()
    n_invalid   = (zones["status"] == "invalid").sum()
    avg_strength = zones["strength"].mean()

    logger.info("  ─── Zone Detection Summary ───────────────────")
    logger.info(f"  Total zones   : {len(zones)}")
    logger.info(f"  Demand zones  : {n_demand}  |  Supply zones : {n_supply}")
    logger.info(f"  Active        : {n_active}")
    logger.info(f"  Tested        : {n_tested}")
    logger.info(f"  Consumed      : {n_consumed}")
    logger.info(f"  Invalid       : {n_invalid}")
    logger.info(f"  Avg strength  : {avg_strength:.3f}")
    logger.info(f"  Date range    : {zones['formation_date'].min().date()} → "
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

    zones = detect(df, zcfg, logger=logger)

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
