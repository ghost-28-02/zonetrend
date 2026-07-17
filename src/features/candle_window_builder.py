"""
candle_window_builder.py
========================
Builds the ML training dataset for the zone detection model.

Approach
--------
For every candle in the processed history we extract a fixed-size
backward-looking window of WINDOW_SIZE candles and compute a feature
vector that describes the price action pattern in that window.

The label for each window is whether the rule-based zone_detector
identified a zone formation at that candle's close.

Why windows instead of single candles
--------------------------------------
A zone is a multi-candle pattern: leg-in → base → departure.
A single candle has no memory of what came before it.
By feeding the last WINDOW_SIZE candles as a feature vector we give
the model the full structural context it needs to recognise the pattern.
This is the key difference from Option B — the model sees a sequence,
not a snapshot.

Feature engineering (per window of size W)
-------------------------------------------
For each of the W candles in the window we compute 7 ATR-normalised features:
  1. close_rel      (Close[j] - Close[window_end]) / ATR — price level relative to anchor
  2. body_atr       |Close[j] - Open[j]| / ATR           — candle body size
  3. range_atr      (High[j]  - Low[j])  / ATR           — candle total range
  4. upper_wick_atr upper wick / ATR                      — rejection above
  5. lower_wick_atr lower wick / ATR                      — rejection below
  6. is_bullish      1 if Close >= Open else 0
  7. volume_ratio   Volume[j] / VolumeMA20[j]             — relative volume

Plus W-level summary features:
  8. net_move       (Close[end] - Close[start]) / ATR    — net trend over window
  9. volatility_atr ATR[end] / ATR[start]                — volatility change
  10. ema20_dist    (Close[end] - EMA20[end]) / ATR
  11. ema50_dist    (Close[end] - EMA50[end]) / ATR
  12. ema200_dist   (Close[end] - EMA200[end]) / ATR

Total features: W × 7 + 5

All features are ATR-normalised → scale-free and volatility-agnostic.
No future information is used: every feature is computed using data
available at (and before) the window's last candle close.

Labels
------
label = 1  if zone_detector found a zone whose formation_date equals
           the window's last candle date
label = 0  otherwise

The dataset is highly imbalanced (~100 positives vs ~2700 negatives
for Nifty daily). We store all rows and let the model handle imbalance
via scale_pos_weight.

Output
------
data/labeled/<SYMBOL>_zone_windows_v2.csv   feature matrix + label
data/labeled/<SYMBOL>_window_features.txt   feature name list
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 20          # candles per window (leg-in + base + departure)
PER_CANDLE_FEATS = 7      # features computed for each candle in window
SUMMARY_FEATS    = 5      # window-level summary features


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _symbol_to_stem(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "IDX_")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_processed(symbol: str, cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["data"]["processed_dir"] / (_symbol_to_stem(symbol) + ".csv")
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_zones(symbol: str, cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["data"]["zones_dir"] / (_symbol_to_stem(symbol) + "_zones.csv")
    if not path.exists():
        raise FileNotFoundError(f"Zones not found: {path}")
    df = pd.read_csv(path)
    df["formation_date"] = pd.to_datetime(df["formation_date"])
    return df


def save_dataset(df: pd.DataFrame, symbol: str, cfg: dict, logger: logging.Logger) -> Path:
    out_dir = PROJECT_ROOT / cfg.get("features", {}).get("labeled_dir", "data/labeled")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (_symbol_to_stem(symbol) + "_zone_windows_v2.csv")
    df.to_csv(path, index=False)
    logger.info(f"Window dataset saved → {path.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")
    return path


def save_feature_names(names: list, symbol: str, cfg: dict, logger: logging.Logger) -> Path:
    out_dir = PROJECT_ROOT / cfg.get("features", {}).get("labeled_dir", "data/labeled")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (_symbol_to_stem(symbol) + "_window_features.txt")
    path.write_text("\n".join(names))
    logger.info(f"Feature list saved  → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── Feature names ─────────────────────────────────────────────────────────────

def get_feature_names(window_size: int = WINDOW_SIZE) -> list:
    """Return ordered list of feature names for a window of given size."""
    names = []
    for i in range(window_size):
        pos = i - window_size + 1   # negative: i=0 → oldest candle (-(W-1))
        names += [
            f"c{pos:+d}_close_rel",
            f"c{pos:+d}_body_atr",
            f"c{pos:+d}_range_atr",
            f"c{pos:+d}_upper_wick_atr",
            f"c{pos:+d}_lower_wick_atr",
            f"c{pos:+d}_is_bullish",
            f"c{pos:+d}_volume_ratio",
        ]
    names += ["net_move_atr", "volatility_ratio", "ema20_dist", "ema50_dist", "ema200_dist"]
    return names


# ── Per-window feature extraction ─────────────────────────────────────────────

def _extract_window_features(window: pd.DataFrame) -> np.ndarray | None:
    """
    Extract the feature vector for a single window DataFrame.

    Parameters
    ----------
    window : DataFrame of exactly WINDOW_SIZE rows, columns:
             Open, High, Low, Close, Volume, ATR, VolumeMA20 (or VolumeRatio),
             EMA20, EMA50, EMA200

    Returns
    -------
    1-D numpy array of length WINDOW_SIZE * 7 + 5, or None if ATR is zero/NaN.
    """
    anchor_atr  = window["ATR"].iloc[-1]
    anchor_close = window["Close"].iloc[-1]

    if pd.isna(anchor_atr) or anchor_atr <= 0:
        return None

    features = []

    for _, row in window.iterrows():
        body       = abs(row["Close"] - row["Open"])
        rng        = row["High"] - row["Low"]
        upper_wick = row["High"] - max(row["Open"], row["Close"])
        lower_wick = min(row["Open"], row["Close"]) - row["Low"]
        vol_ratio  = row.get("VolumeRatio", row["Volume"] / max(row.get("VolumeMA20", row["Volume"]), 1))

        features.extend([
            (row["Close"] - anchor_close) / anchor_atr,   # close_rel
            body       / anchor_atr,                       # body_atr
            rng        / anchor_atr,                       # range_atr
            max(upper_wick, 0) / anchor_atr,               # upper_wick_atr
            max(lower_wick, 0) / anchor_atr,               # lower_wick_atr
            1.0 if row["Close"] >= row["Open"] else 0.0,   # is_bullish
            float(vol_ratio) if not pd.isna(vol_ratio) else 1.0,  # volume_ratio
        ])

    # Summary features
    start_close = window["Close"].iloc[0]
    start_atr   = window["ATR"].iloc[0]
    net_move    = (anchor_close - start_close) / anchor_atr
    vol_ratio_w = anchor_atr / start_atr if (not pd.isna(start_atr) and start_atr > 0) else 1.0

    ema20_dist  = (anchor_close - window["EMA20"].iloc[-1])  / anchor_atr if "EMA20"  in window.columns else 0.0
    ema50_dist  = (anchor_close - window["EMA50"].iloc[-1])  / anchor_atr if "EMA50"  in window.columns else 0.0
    ema200_dist = (anchor_close - window["EMA200"].iloc[-1]) / anchor_atr if "EMA200" in window.columns else 0.0

    features.extend([net_move, vol_ratio_w, ema20_dist, ema50_dist, ema200_dist])

    return np.array(features, dtype=float)


# ── Build full dataset ────────────────────────────────────────────────────────

def build_window_dataset(
    proc_df:  pd.DataFrame,
    zones_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Slide a window over the full processed history and build the training dataset.

    For each candle i (where i >= window_size):
      - Extract window[i-window_size : i]  (exclusive end → last candle is i-1
        shifted so window[-1] = candle i)

    Wait — we include candle i itself as the anchor (departure candle).
    The window is proc_df[i-window_size+1 : i+1] (inclusive), so:
      - window[0]  = oldest candle (potential leg-in start)
      - window[-1] = candle i (potential departure candle)

    Label = 1 if a zone's formation_date == date of candle i.

    Parameters
    ----------
    proc_df     : preprocessed OHLCV with ATR, EMA20/50/200, VolumeRatio
    zones_df    : rule-based zone detections
    window_size : number of candles per window
    logger      : optional logger

    Returns
    -------
    DataFrame with feature columns + ['date', 'label', 'zone_type']
    """
    if logger is None:
        logger = logging.getLogger("candle_window_builder")

    # Build a set of zone formation dates for fast O(1) lookup
    zone_date_set = set(pd.to_datetime(zones_df["formation_date"]).dt.normalize())

    # Map date → zone type for the label column
    zone_type_map: dict = {}
    for _, z in zones_df.iterrows():
        d = pd.to_datetime(z["formation_date"]).normalize()
        # If multiple zones on same date, take the first
        if d not in zone_type_map:
            zone_type_map[d] = z.get("type", "demand")

    feature_names = get_feature_names(window_size)
    rows = []
    skipped_nan = 0

    for i in range(window_size - 1, len(proc_df)):
        window = proc_df.iloc[i - window_size + 1 : i + 1]
        feats  = _extract_window_features(window)

        if feats is None:
            skipped_nan += 1
            continue

        candle_date = proc_df["Date"].iloc[i].normalize()
        is_zone     = 1 if candle_date in zone_date_set else 0
        z_type      = zone_type_map.get(candle_date, "none")

        row = dict(zip(feature_names, feats))
        row["date"]      = candle_date
        row["label"]     = is_zone
        row["zone_type"] = z_type   # 'demand' / 'supply' / 'none' — for 3-class variant
        rows.append(row)

    df = pd.DataFrame(rows)

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"Window dataset built: {len(df)} rows "
                f"(positive={n_pos}, negative={n_neg}, skipped_nan={skipped_nan})")
    logger.info(f"Imbalance ratio: {n_neg}/{n_pos} = {n_neg/max(n_pos,1):.1f}:1")
    logger.info(f"Feature count: {len(feature_names)}")

    return df


# ── Pipeline entry ────────────────────────────────────────────────────────────

def run_window_builder(
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Called from zone_pipeline.py."""
    proc_df  = load_processed(symbol, cfg)
    zones_df = load_zones(symbol, cfg)

    logger.info(f"Building candle windows | symbol={symbol} | W={window_size}")
    dataset = build_window_dataset(proc_df, zones_df, window_size=window_size, logger=logger)

    save_dataset(dataset, symbol, cfg, logger)
    save_feature_names(get_feature_names(window_size), symbol, cfg, logger)

    return dataset


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build candle window training dataset.")
    parser.add_argument("--symbol",  type=str, default=None)
    parser.add_argument("--window",  type=int, default=WINDOW_SIZE)
    args   = parser.parse_args()
    cfg    = load_config()
    import logging as _log
    logging.basicConfig(level=_log.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger = _log.getLogger("candle_window_builder")
    symbol = args.symbol or cfg["data"]["symbol"]
    run_window_builder(symbol, cfg, logger, window_size=args.window)
