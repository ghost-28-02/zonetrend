"""
ml_zone_detector.py
===================
Uses the trained zone detection model to scan a price history and
produce a zones DataFrame compatible with the existing pipeline.

How it works
------------
1. Build candle windows over the full processed history (same logic as
   candle_window_builder.py — no re-training needed).
2. Run the trained XGBoost model on every window.
3. Where P(zone) >= threshold, the model has detected a zone.
4. Reconstruct zone boundaries from the local window at that point.
5. Output a zones_df with the same columns as zone_detector.py so all
   downstream code (labeler, notebook) works without modification.

Zone boundary reconstruction
-----------------------------
When the model fires at candle i (departure candle), we look back
WINDOW_SIZE candles and:

Step 1 — Identify base candles
  Base candles are the tightest candles in the window.
  We define "tight" as: body_atr < median(body_atr) × BASE_BODY_FACTOR
                    AND range_atr < median(range_atr) × BASE_RANGE_FACTOR
  We then find the longest consecutive run of tight candles within
  the middle/end portion of the window (not the very first candles).

Step 2 — Zone boundaries
  top    = max(High  of base candles)
  bottom = min(Low   of base candles)

Step 3 — Zone type (demand vs supply)
  We look at the leg-in direction: the net move of the first half
  of the window before the base.
  net_leg_in > 0 (rally into base) → SUPPLY zone (RBD pattern)
  net_leg_in < 0 (drop into base)  → DEMAND zone (DBR pattern)

Step 4 — Structure label
  DBR : drop → base → rally  (demand reversal)
  RBD : rally → base → drop  (supply reversal)
  We detect the departure direction from the last 3 candles of the window.
  Bullish departure → DBR (demand)
  Bearish departure → RBD (supply)

Step 5 — Proximal / distal
  Demand: proximal = top (closest to price), distal = bottom
  Supply: proximal = bottom,                distal = top

Deduplication
-------------
Multiple consecutive high-probability windows may fire for the same
zone. We deduplicate by:
  - Sorting by probability descending
  - Suppressing any detection within DEDUP_WINDOW candles of a higher-
    probability detection whose zone boundaries overlap by > OVERLAP_PCT

Output columns (matches zone_detector.py)
------------------------------------------
zone_id, type, structure, top, bottom, midpoint, width, proximal, distal,
formation_date, base_start_date, base_end_date, base_length,
avg_atr, ml_prob, status, test_count

Note: advanced scoring columns (strength, weekly_confluence, etc.) are
set to NaN — they require the full rule-based pipeline. The notebook
handles this gracefully.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

# ── Reconstruction constants ──────────────────────────────────────────────────
WINDOW_SIZE        = 20    # must match candle_window_builder
BASE_BODY_FACTOR   = 1.2   # body <= median_body × factor → base candle
BASE_RANGE_FACTOR  = 1.3   # range <= median_range × factor → base candle
MIN_BASE_LEN       = 1     # minimum base candles
MAX_BASE_LEN       = 6     # maximum base candles
MAX_BASE_LOOKBACK  = 4     # base must end within this many candles of departure
DEDUP_WINDOW       = 5     # suppress detections within N candles of better one
DEFAULT_THRESHOLD  = 0.45  # P(zone) >= threshold → zone detected


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _symbol_to_stem(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "IDX_")


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_processed(symbol: str, cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["data"]["processed_dir"] / (_symbol_to_stem(symbol) + ".csv")
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def save_ml_zones(zones_df: pd.DataFrame, symbol: str, cfg: dict, logger: logging.Logger) -> Path:
    out_dir = PROJECT_ROOT / cfg["data"]["zones_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (_symbol_to_stem(symbol) + "_ml_zones.csv")
    zones_df.to_csv(path, index=False)
    logger.info(f"ML zones saved → {path.relative_to(PROJECT_ROOT)}  ({len(zones_df)} zones)")
    return path


def load_ml_zones(symbol: str, cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["data"]["zones_dir"] / (_symbol_to_stem(symbol) + "_ml_zones.csv")
    if not path.exists():
        raise FileNotFoundError(f"ML zones not found: {path}")
    df = pd.read_csv(path)
    df["formation_date"] = pd.to_datetime(df["formation_date"])
    return df


# ── Window feature extraction (mirrors candle_window_builder) ────────────────

def _extract_window_features(window: pd.DataFrame) -> np.ndarray | None:
    """Extract feature vector for one window (same logic as candle_window_builder)."""
    anchor_atr   = window["ATR"].iloc[-1]
    anchor_close = window["Close"].iloc[-1]
    if pd.isna(anchor_atr) or anchor_atr <= 0:
        return None

    features = []
    for _, row in window.iterrows():
        body       = abs(row["Close"] - row["Open"])
        rng        = row["High"] - row["Low"]
        upper_wick = row["High"] - max(row["Open"], row["Close"])
        lower_wick = min(row["Open"], row["Close"]) - row["Low"]
        vol_ratio  = row.get("VolumeRatio", 1.0)
        features.extend([
            (row["Close"] - anchor_close) / anchor_atr,
            body       / anchor_atr,
            rng        / anchor_atr,
            max(upper_wick, 0) / anchor_atr,
            max(lower_wick, 0) / anchor_atr,
            1.0 if row["Close"] >= row["Open"] else 0.0,
            float(vol_ratio) if not pd.isna(vol_ratio) else 1.0,
        ])

    start_close = window["Close"].iloc[0]
    start_atr   = window["ATR"].iloc[0]
    net_move    = (anchor_close - start_close) / anchor_atr
    vol_ratio_w = anchor_atr / start_atr if (not pd.isna(start_atr) and start_atr > 0) else 1.0
    ema20_dist  = (anchor_close - window["EMA20"].iloc[-1])  / anchor_atr if "EMA20"  in window.columns else 0.0
    ema50_dist  = (anchor_close - window["EMA50"].iloc[-1])  / anchor_atr if "EMA50"  in window.columns else 0.0
    ema200_dist = (anchor_close - window["EMA200"].iloc[-1]) / anchor_atr if "EMA200" in window.columns else 0.0
    features.extend([net_move, vol_ratio_w, ema20_dist, ema50_dist, ema200_dist])

    return np.array(features, dtype=float)


# ── Zone boundary reconstruction ─────────────────────────────────────────────

def _reconstruct_zone(window: pd.DataFrame, atr: float) -> dict | None:
    """
    Given a window ending at the departure candle, reconstruct zone boundaries.

    Returns dict with zone geometry or None if reconstruction fails.
    """
    n = len(window)

    # ── Identify base candles (tight body + tight range) ──────────────────
    bodies = np.array([abs(r["Close"] - r["Open"]) for _, r in window.iterrows()])
    ranges = np.array([r["High"] - r["Low"] for _, r in window.iterrows()])

    med_body  = np.median(bodies[bodies > 0]) if (bodies > 0).any() else atr * 0.3
    med_range = np.median(ranges[ranges > 0]) if (ranges > 0).any() else atr * 0.5

    is_base = (
        (bodies <= med_body  * BASE_BODY_FACTOR) &
        (ranges <= med_range * BASE_RANGE_FACTOR)
    )

    # Only look for base in positions 3 to n-2 (not the oldest leg-in or very last departure)
    is_base[:3]   = False
    is_base[-1]   = False   # last candle is the departure candle itself

    # ── Find base nearest to departure (work backwards from n-2) ─────────────
    # This guarantees the structure is: leg-in → [base] → departure.
    # Looking for the longest run gave bases near the start of the window
    # (inside the leg-in), producing a visually wrong gap to the departure.
    best_start, best_end = -1, -1

    # The base end must be within MAX_BASE_LOOKBACK candles of the departure
    earliest_base_pos = max(3, n - 1 - MAX_BASE_LOOKBACK)

    # Scan backwards from n-2 to find the last tight candle before departure
    for k in range(n - 2, earliest_base_pos - 1, -1):
        if is_base[k]:
            best_end = k
            break

    if best_end != -1:
        # Extend the run backwards (consecutive tight candles, up to MAX_BASE_LEN)
        best_start = best_end
        while (best_start > earliest_base_pos and
               is_base[best_start - 1] and
               (best_end - best_start + 1) < MAX_BASE_LEN):
            best_start -= 1

    best_len = (best_end - best_start + 1) if best_end != -1 else 0

    if best_len == 0:
        # Fallback: pick the 2 tightest candles in the last MAX_BASE_LOOKBACK region
        search_region = list(range(earliest_base_pos, n - 1))
        if len(search_region) < 1:
            return None
        combined = bodies + ranges
        tightest = sorted(search_region, key=lambda k: combined[k])[:2]
        best_start = min(tightest)
        best_end   = max(tightest)

    base_slice = window.iloc[best_start : best_end + 1]
    top        = float(base_slice["High"].max())
    bottom     = float(base_slice["Low"].min())

    if top <= bottom:
        return None

    # ── Determine structure using both leg-in and departure direction ─────────
    #
    # Departure candle body direction (primary signal)
    dep_slice = window.iloc[best_end + 1:]
    if len(dep_slice) >= 2:
        dep_body = dep_slice["Close"].iloc[-1] - dep_slice["Close"].iloc[0]
    elif len(dep_slice) == 1:
        dep_body = dep_slice["Close"].iloc[0] - dep_slice["Open"].iloc[0]
    else:
        dep_body = base_slice["Close"].iloc[-1] - base_slice["Open"].iloc[-1]

    # Leg-in net direction (candles before the base)
    leg_in_slice = window.iloc[:best_start]
    if len(leg_in_slice) >= 2:
        leg_net = leg_in_slice["Close"].iloc[-1] - leg_in_slice["Close"].iloc[0]
    else:
        leg_net = window["Close"].iloc[-1] - window["Close"].iloc[0]

    # Departure direction determines zone type (demand = bullish, supply = bearish)
    # Leg-in direction determines structure (reversal vs continuation)
    #
    # All 4 patterns:
    #   DBR : Drop   → Base → Rally  (demand reversal)
    #   RBR : Rally  → Base → Rally  (demand continuation)
    #   RBD : Rally  → Base → Drop   (supply reversal)
    #   DBD : Drop   → Base → Drop   (supply continuation)
    if dep_body > 0:
        zone_type = "demand"
        structure = "DBR" if leg_net < 0 else "RBR"
        proximal  = top
        distal    = bottom
    elif dep_body < 0:
        zone_type = "supply"
        structure = "RBD" if leg_net > 0 else "DBD"
        proximal  = bottom
        distal    = top
    else:
        # Doji departure — infer from leg-in
        if leg_net <= 0:
            zone_type, structure = "demand", "DBR"
            proximal, distal = top, bottom
        else:
            zone_type, structure = "supply", "RBD"
            proximal, distal = bottom, top

    return {
        "type":      zone_type,
        "structure": structure,
        "top":       round(top, 2),
        "bottom":    round(bottom, 2),
        "midpoint":  round((top + bottom) / 2, 2),
        "width":     round(top - bottom, 2),
        "proximal":  round(proximal, 2),
        "distal":    round(distal, 2),
        "base_length": best_end - best_start + 1,
        "base_start_date": str(window.index[best_start] if isinstance(window.index[0], pd.Timestamp)
                               else window["Date"].iloc[best_start].date()),
        "base_end_date":   str(window.index[best_end]   if isinstance(window.index[0], pd.Timestamp)
                               else window["Date"].iloc[best_end].date()),
        "avg_atr": round(float(window["ATR"].mean()), 4),
    }


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate_zones(candidates: list, dedup_window: int = DEDUP_WINDOW) -> list:
    """
    Remove overlapping zone detections, keeping the highest-probability one.
    Two zones overlap if their date indices are within dedup_window of each other
    AND their price ranges overlap.
    """
    if not candidates:
        return []

    # Sort by probability descending
    candidates = sorted(candidates, key=lambda x: x["ml_prob"], reverse=True)
    kept = []

    for cand in candidates:
        overlap = False
        for k in kept:
            # Date proximity check
            if abs(cand["_idx"] - k["_idx"]) <= dedup_window:
                # Price range overlap check
                if cand["bottom"] <= k["top"] and cand["top"] >= k["bottom"]:
                    overlap = True
                    break
        if not overlap:
            kept.append(cand)

    return kept


# ── Forward status tracking ────────────────────────────────────────────────────

def _track_zone_status(zones_df: pd.DataFrame, proc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk forward through price history and update:
      status       : 'active' | 'invalidated'
      test_count   : number of times price entered the zone
      last_test_date
      invalidation_date
    """
    zones_df = zones_df.copy()
    zones_df["status"]            = "active"
    zones_df["test_count"]        = 0
    zones_df["last_test_date"]    = pd.NaT
    zones_df["invalidation_date"] = pd.NaT

    price_df = proc_df.set_index("Date")[["High", "Low", "Close"]].sort_index()

    for idx, zone in zones_df.iterrows():
        fdate    = pd.to_datetime(zone["formation_date"])
        future   = price_df[price_df.index > fdate]
        z_top    = zone["top"]
        z_bot    = zone["bottom"]
        z_type   = zone["type"]
        tc       = 0
        last_td  = pd.NaT
        inv_date = pd.NaT

        for date, row in future.iterrows():
            # Test: price enters zone
            if row["Low"] <= z_top and row["High"] >= z_bot:
                tc      += 1
                last_td  = date

            # Invalidation: close beyond distal
            if z_type == "demand" and row["Close"] < z_bot:
                inv_date = date
                break
            elif z_type == "supply" and row["Close"] > z_top:
                inv_date = date
                break

        zones_df.at[idx, "test_count"]        = tc
        zones_df.at[idx, "last_test_date"]    = last_td
        zones_df.at[idx, "invalidation_date"] = inv_date
        zones_df.at[idx, "status"]            = "active" if pd.isna(inv_date) else "invalidated"

    return zones_df


# ── Main detection function ───────────────────────────────────────────────────

def detect_zones_with_model(
    proc_df:   pd.DataFrame,
    model,
    threshold: float = DEFAULT_THRESHOLD,
    logger:    logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Slide the trained model over the full price history and return a
    zones DataFrame.

    Parameters
    ----------
    proc_df   : preprocessed OHLCV DataFrame
    model     : trained XGBoost (or RF fallback) model
    threshold : P(zone) >= threshold → zone detected
    logger    : optional logger

    Returns
    -------
    pd.DataFrame with one row per detected zone, same column structure
    as zone_detector.py output.
    """
    if logger is None:
        logger = logging.getLogger("ml_zone_detector")

    n = len(proc_df)
    n_classes = len(model.classes_) if hasattr(model, "classes_") else 2
    zone_class_idx = 1   # index of the positive (zone) class

    candidates = []
    skipped    = 0

    logger.info(f"Scanning {n} candles | threshold={threshold}")

    for i in range(WINDOW_SIZE - 1, n):
        window = proc_df.iloc[i - WINDOW_SIZE + 1 : i + 1].copy()
        feats  = _extract_window_features(window)

        if feats is None:
            skipped += 1
            continue

        prob = model.predict_proba(feats.reshape(1, -1))[0][zone_class_idx]

        if prob < threshold:
            continue

        # Reconstruct zone geometry from this window
        atr  = float(proc_df["ATR"].iloc[i])
        zone = _reconstruct_zone(window, atr)
        if zone is None:
            continue

        formation_date = proc_df["Date"].iloc[i]
        zone.update({
            "formation_date": formation_date,
            "ml_prob":        round(float(prob), 4),
            "_idx":           i,          # used for deduplication, removed later
        })
        candidates.append(zone)

    logger.info(f"Raw detections before dedup: {len(candidates)}  (skipped_nan={skipped})")

    # Deduplicate overlapping detections
    kept = _deduplicate_zones(candidates, dedup_window=DEDUP_WINDOW)
    logger.info(f"Zones after deduplication: {len(kept)}")

    if not kept:
        return pd.DataFrame()

    # Remove internal dedup field and build DataFrame
    for z in kept:
        z.pop("_idx", None)

    zones_df = pd.DataFrame(kept).reset_index(drop=True)
    zones_df.index.name = "zone_id"
    zones_df = zones_df.reset_index()
    zones_df["zone_id"] = "ML_" + zones_df["zone_id"].astype(str)

    # Add NaN columns that rule-based detector provides (for compatibility)
    for col in ["strength", "strength_pit", "adjusted_strength_posthoc",
                "trend_aligned", "weekly_confluence_score"]:
        if col not in zones_df.columns:
            zones_df[col] = np.nan

    # Track zone status forward in time
    zones_df = _track_zone_status(zones_df, proc_df)

    demand = (zones_df["type"] == "demand").sum()
    supply = (zones_df["type"] == "supply").sum()
    active = (zones_df["status"] == "active").sum()
    logger.info(f"ML zones: {len(zones_df)} total  "
                f"(demand={demand}, supply={supply}, active={active})")

    return zones_df


# ── Pipeline entry ────────────────────────────────────────────────────────────

def run_ml_detection(
    symbol:    str,
    cfg:       dict,
    logger:    logging.Logger,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Called from zone_pipeline.py."""
    from src.models.zone_detection_model import load_model

    proc_df = load_processed(symbol, cfg)
    model   = load_model(symbol, cfg)

    logger.info(f"Running ML zone detection | symbol={symbol} | threshold={threshold}")
    zones_df = detect_zones_with_model(proc_df, model, threshold=threshold, logger=logger)

    if not zones_df.empty:
        save_ml_zones(zones_df, symbol, cfg, logger)

    return zones_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ML zone detection.")
    parser.add_argument("--symbol",    type=str,   default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args   = parser.parse_args()
    cfg    = load_config()
    import logging as _log
    logging.basicConfig(level=_log.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger = _log.getLogger("ml_zone_detector")
    symbol = args.symbol or cfg["data"]["symbol"]
    run_ml_detection(symbol, cfg, logger, threshold=args.threshold)
