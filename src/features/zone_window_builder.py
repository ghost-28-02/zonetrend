"""
zone_window_builder.py
======================
Builds the training dataset for zone prediction.

Goal
----
Train a Random Forest that answers:
  "Given what the market looks like right now, will a zone form here —
   and if yes, what type (DBR / RBD / RBR / DBD)?"

The model sees only information available UP TO the last base candle.
It does NOT see the departure leg. This is what makes it genuinely
predictive — a trader sitting at the last base candle cannot see what
comes next, and neither does the model during prediction.

How it works
------------
A "zone window" consists of two parts:

  [  leg-in candles  ][  base candles  ]
  |<-- 3 candles -->||<-- 1-3 candles -->|

  For POSITIVE examples: we use confirmed zones from zone_detector.py.
  The zone already tells us the leg-in length, base length, and all
  computed features. We just need to add market context from the
  processed data at base_end_date.

  For NEGATIVE examples: we slide the same window across all candles
  that are NOT near any zone formation and compute the same features
  from raw candle data. These are windows that "look like they could
  become zones" but did not.

Look-ahead policy (strict)
--------------------------
Only features computable from candles UP TO base_end_date are included
in the ML feature set. Departure / leg-out features (departure_body_atr,
leg_out_disp_atr, etc.) are logged separately as "analysis-only" columns
with an _ao suffix and EXCLUDED from the default feature set.

Why this matters: the departure leg happens AFTER the base closes.
Including departure features in X would mean the model is "predicting"
something it already has the answer to — useless for real trading.

What the model predicts
-----------------------
  is_zone     (binary)    → 1 if a zone formed, 0 if not
  structure   (5-class)   → DBR / RBD / RBR / DBD / no_zone

Use is_zone first (binary classifier). Once the model says "yes, zone
forming", a second model trained only on positive examples can refine
the structure type.

Feature set (pre-departure only)
---------------------------------
  Leg-in:
    leg_in_disp_atr       Net displacement of arrival leg in ATR units
    leg_in_direction      +1 (rising into base) or -1 (falling into base)
    leg_in_velocity       Displacement per candle (leg_in_disp_atr / candles)
    leg_in_bullish_ratio  Fraction of bullish candles in the arrival leg

  Base:
    base_length           Number of base candles (1-3)
    base_body_atr         Mean |close-open| of base candles / ATR
    base_wick_frac        Mean wick fraction: (range - body) / range
    base_range_atr        (max high - min low) of base candles / ATR
    base_volume_ratio     Mean base volume / VolumeMA20

  Derived:
    disp_base_ratio       |leg_in_disp_atr| / base_range_atr
                          High value = strong arrival into tight base = quality signal

  Market context at base_end_date:
    atr_level             ATR value (absolute volatility)
    close_vs_ema200       (close - EMA200) / ATR — positive = above EMA = uptrend
    close_vs_ema50        (close - EMA50)  / ATR
    volume_ratio          VolumeRatio from processed data
    rolling_volatility    Rolling std of log returns
    body_to_atr           Body / ATR of the last base candle
    is_bullish_last       1 if last base candle is bullish

  Labels:
    is_zone               1 = confirmed zone, 0 = no zone
    structure             DBR / RBD / RBR / DBD / no_zone

Output files
------------
  data/labeled/<SYMBOL>_zone_windows.csv   — full training dataset
  data/labeled/<SYMBOL>_features.txt       — list of ML feature column names

Usage
-----
    # Run standalone
    python src/features/zone_window_builder.py

    # Override symbol and negative sample count
    python src/features/zone_window_builder.py --symbol RELIANCE.NS --n_neg 300
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

STRUCTURE_LABELS = ["DBR", "RBD", "RBR", "DBD"]

# Features the Random Forest will receive (pre-departure, no look-ahead)
ML_FEATURES = [
    # Leg-in
    "leg_in_disp_atr",
    "leg_in_direction",
    "leg_in_velocity",
    "leg_in_bullish_ratio",
    # Base
    "base_length",
    "base_body_atr",
    "base_wick_frac",
    "base_range_atr",
    "base_volume_ratio",
    # Derived
    "disp_base_ratio",
    # Market context
    "atr_level",
    "close_vs_ema200",
    "close_vs_ema50",
    "volume_ratio",
    "rolling_volatility",
    "body_to_atr",
    "is_bullish_last",
]

# Post-departure features — kept in dataset with _ao suffix for analysis,
# excluded from ML_FEATURES
ANALYSIS_ONLY_FEATURES = [
    "departure_body_atr",
    "departure_close_ratio",
    "leg_out_disp_atr",
    "leg_out_velocity",
    "strength_ao",
    "quality_score_ao",
    "trend_score_ao",
    "weekly_confluence_score_ao",
]


# ── Config & logging ────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = PROJECT_ROOT / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    level    = getattr(logging, cfg["logging"]["log_level"].upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if cfg["logging"].get("log_to_file", True):
        handlers.append(logging.FileHandler(log_dir / "zone_window_builder.log"))
    if cfg["logging"].get("log_to_console", True):
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=level, handlers=handlers,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("zone_window_builder")


# ── Path helpers ────────────────────────────────────────────────────────────

def _symbol_to_stem(symbol: str) -> str:
    if symbol.startswith("^"):
        return "IDX_" + symbol[1:].replace(".", "_")
    return symbol.replace(".", "_")


def _labeled_dir(cfg: dict) -> Path:
    d = PROJECT_ROOT / cfg.get("features", {}).get("labeled_dir", "data/labeled")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Processed data helpers ──────────────────────────────────────────────────

def _ema200_dist(row_idx: int, proc: pd.DataFrame) -> float:
    """(close - EMA200) / ATR at row_idx. NaN-safe."""
    row  = proc.iloc[row_idx]
    atr  = row.get("ATR", np.nan)
    ema  = row.get("EMA200", np.nan)
    cls  = row.get("Close", np.nan)
    if pd.isna(atr) or atr == 0 or pd.isna(ema):
        return np.nan
    return (cls - ema) / atr


def _ema50_dist(row_idx: int, proc: pd.DataFrame) -> float:
    row  = proc.iloc[row_idx]
    atr  = row.get("ATR", np.nan)
    ema  = row.get("EMA50", np.nan)
    cls  = row.get("Close", np.nan)
    if pd.isna(atr) or atr == 0 or pd.isna(ema):
        return np.nan
    return (cls - ema) / atr


# ── Positive examples (confirmed zones) ────────────────────────────────────

def extract_positive_examples(
    zones_df: pd.DataFrame,
    proc: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Build one training row per confirmed zone.

    Uses pre-departure features directly from zones_df (already computed
    by zone_detector.py) plus market context from proc at base_end_date.

    No departure features are included in ML_FEATURES. They are copied
    with an _ao suffix for analysis only.
    """
    proc = proc.copy()
    proc["Date"] = pd.to_datetime(proc["Date"])
    date_to_idx  = {row["Date"]: i for i, row in proc.iterrows()}

    rows = []
    skipped = 0

    for _, zone in zones_df.iterrows():
        base_end_date = pd.to_datetime(zone["base_end_date"])

        if base_end_date not in date_to_idx:
            logger.debug(f"base_end_date {base_end_date} not in processed data — skipped")
            skipped += 1
            continue

        idx  = date_to_idx[base_end_date]
        prow = proc.iloc[idx]
        atr  = float(zone.get("avg_atr", prow.get("ATR", np.nan)))

        # ── Leg-in features (pre-departure, computed by zone_detector) ──
        leg_in_disp_atr = float(zone.get("leg_in_disp_atr", np.nan))

        # Infer leg-in direction from leg-in displacement sign
        # AND from the zone type (demand = bullish departure, supply = bearish)
        # We infer leg-in direction from structure:
        #   DBR → D = drop → leg-in was DOWN  (-1)
        #   RBD → R = rally → leg-in was UP   (+1)
        #   RBR → R = rally → leg-in was UP   (+1)
        #   DBD → D = drop → leg-in was DOWN  (-1)
        structure        = zone["structure"]
        leg_in_direction = -1 if structure in ("DBR", "DBD") else +1

        leg_in_candles  = int(zone.get("leg_in_candles", 3))
        leg_in_velocity = float(zone.get("leg_in_velocity", np.nan))

        # Compute leg-in bullish ratio from raw candle data
        leg_in_end_idx   = idx - int(zone.get("base_length", 1))
        leg_in_start_idx = max(0, leg_in_end_idx - leg_in_candles)
        leg_in_window    = proc.iloc[leg_in_start_idx:leg_in_end_idx]
        leg_in_bullish_ratio = (
            leg_in_window["IsBullish"].mean()
            if len(leg_in_window) > 0 and "IsBullish" in leg_in_window.columns
            else np.nan
        )

        # ── Base features (pre-departure, computed by zone_detector) ──
        base_length     = int(zone.get("base_length", 1))
        base_wick_frac  = float(zone.get("base_wick_frac", np.nan))
        base_volume_ratio = float(zone.get("base_volume_ratio", np.nan))

        # Extract base candles from processed data for body/range features
        base_start_idx = idx - base_length + 1
        base_candles   = proc.iloc[base_start_idx : idx + 1]

        if len(base_candles) == 0:
            skipped += 1
            continue

        if atr > 0 and not np.isnan(atr):
            base_body_atr  = float(base_candles["CandleBody"].mean() / atr) \
                if "CandleBody" in base_candles.columns else np.nan
            base_range_atr = float(
                (base_candles["High"].max() - base_candles["Low"].min()) / atr
            )
        else:
            base_body_atr  = np.nan
            base_range_atr = float(zone.get("width_atr", np.nan))

        # disp_base_ratio: how large was the arrival move vs how tight the base
        disp_base_ratio = float(zone.get("disp_base_ratio", np.nan))
        if np.isnan(disp_base_ratio) and not np.isnan(base_range_atr) and base_range_atr > 0:
            disp_base_ratio = abs(leg_in_disp_atr) / base_range_atr

        # ── Market context at base_end_date ──
        atr_level          = float(prow.get("ATR", np.nan))
        close_vs_ema200    = _ema200_dist(idx, proc)
        close_vs_ema50     = _ema50_dist(idx, proc)
        volume_ratio       = float(prow.get("VolumeRatio", np.nan))
        rolling_volatility = float(prow.get("RollingVolatility", np.nan))
        body_to_atr        = float(prow.get("BodyToATR", np.nan))
        is_bullish_last    = int(prow.get("IsBullish", 0))

        # ── Labels ──
        row = {
            # Identifiers (not features)
            "zone_id":          zone.get("zone_id", ""),
            "symbol":           zone.get("symbol", ""),
            "base_end_date":    str(base_end_date.date()),
            "formation_date":   str(pd.to_datetime(zone["formation_date"]).date()),
            "example_type":     "positive",

            # ML features (pre-departure)
            "leg_in_disp_atr":       leg_in_disp_atr,
            "leg_in_direction":      leg_in_direction,
            "leg_in_velocity":       leg_in_velocity,
            "leg_in_bullish_ratio":  leg_in_bullish_ratio,
            "base_length":           base_length,
            "base_body_atr":         base_body_atr,
            "base_wick_frac":        base_wick_frac,
            "base_range_atr":        base_range_atr,
            "base_volume_ratio":     base_volume_ratio,
            "disp_base_ratio":       disp_base_ratio,
            "atr_level":             atr_level,
            "close_vs_ema200":       close_vs_ema200,
            "close_vs_ema50":        close_vs_ema50,
            "volume_ratio":          volume_ratio,
            "rolling_volatility":    rolling_volatility,
            "body_to_atr":           body_to_atr,
            "is_bullish_last":       is_bullish_last,

            # Analysis-only post-departure features (excluded from ML training)
            "departure_body_atr":          float(zone.get("departure_body_atr", np.nan)),
            "departure_close_ratio":       float(zone.get("departure_close_ratio", np.nan)),
            "leg_out_disp_atr":            float(zone.get("leg_out_disp_atr", np.nan)),
            "leg_out_velocity":            float(zone.get("leg_out_velocity", np.nan)),
            "strength_ao":                 float(zone.get("strength_pit", np.nan)),
            "quality_score_ao":            float(zone.get("quality_score", np.nan)),
            "trend_score_ao":              float(zone.get("trend_score", np.nan)),
            "weekly_confluence_score_ao":  float(zone.get("weekly_confluence_score", np.nan)),

            # Labels
            "is_zone":   1,
            "structure": structure,
        }
        rows.append(row)

    logger.info(
        f"Positive examples: {len(rows)} extracted, {skipped} skipped "
        f"(base_end_date not in processed data)"
    )
    return pd.DataFrame(rows)


# ── Negative examples (non-zone windows) ───────────────────────────────────

def extract_negative_examples(
    proc: pd.DataFrame,
    zones_df: pd.DataFrame,
    n_samples: int,
    leg_in_lookback: int,
    max_base_length: int,
    logger: logging.Logger,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate negative training examples by sliding a fixed window across
    all candles that are NOT near any actual zone formation.

    Window layout (fixed total size = leg_in_lookback + max_base_length):
      [  leg-in: leg_in_lookback candles  ][  base: max_base_length candles  ]

    For negative examples, the "base" is the last max_base_length candles
    of the window. They didn't become a zone, but we measure their features
    the same way so the model can compare them to true zone bases.

    Exclusion zone:
    Any candle within (leg_in_lookback + max_base_length + departure_buffer)
    candles of an actual zone formation_idx is excluded from negative sampling.
    This prevents mislabeling near-miss candidates as negatives.
    """
    proc = proc.copy()
    proc["Date"] = pd.to_datetime(proc["Date"])
    total_window = leg_in_lookback + max_base_length
    # Buffer: exclude candles within this many positions of any zone
    exclusion_buffer = total_window + 5

    # Mark excluded indices (around every zone formation)
    excluded = set()
    zone_formation_idxs = zones_df["formation_idx"].dropna().astype(int).tolist()
    for fidx in zone_formation_idxs:
        for offset in range(-exclusion_buffer, exclusion_buffer + 1):
            excluded.add(fidx + offset)

    # Valid "window end" indices: last candle of the simulated base.
    # Also exclude rows where ATR is NaN (warm-up period at the start of history).
    min_idx   = total_window - 1      # need full window before this point
    max_idx   = len(proc) - 1
    candidates = [
        i for i in range(min_idx, max_idx + 1)
        if i not in excluded
        and not pd.isna(proc.iloc[i].get("ATR", np.nan))
    ]

    logger.info(
        f"Negative sampling: {len(candidates)} valid window positions "
        f"out of {len(proc)} candles ({len(excluded)} excluded near zones)"
    )

    rng = np.random.default_rng(random_seed)
    if len(candidates) < n_samples:
        logger.warning(
            f"Only {len(candidates)} valid positions available; "
            f"requested {n_samples}. Using all available."
        )
        sampled_idxs = candidates
    else:
        sampled_idxs = rng.choice(candidates, size=n_samples, replace=False).tolist()

    rows = []

    for end_idx in sampled_idxs:
        # Window slices
        base_start_idx   = end_idx - max_base_length + 1
        base_end_idx     = end_idx                          # inclusive
        leg_in_end_idx   = base_start_idx - 1
        leg_in_start_idx = leg_in_end_idx - leg_in_lookback + 1

        base_candles   = proc.iloc[base_start_idx : base_end_idx + 1]
        leg_in_candles = proc.iloc[leg_in_start_idx : leg_in_end_idx + 1]

        if len(base_candles) == 0 or len(leg_in_candles) == 0:
            continue

        prow = proc.iloc[end_idx]
        atr  = float(prow.get("ATR", np.nan))

        # ── Leg-in features ──
        if not pd.isna(atr) and atr > 0:
            leg_in_net = float(
                leg_in_candles["Close"].iloc[-1] - leg_in_candles["Open"].iloc[0]
            )
            leg_in_disp_atr = leg_in_net / atr
        else:
            leg_in_disp_atr = np.nan

        leg_in_direction     = int(np.sign(leg_in_disp_atr)) if not np.isnan(leg_in_disp_atr) else 0
        leg_in_velocity      = leg_in_disp_atr / leg_in_lookback if not np.isnan(leg_in_disp_atr) else np.nan
        leg_in_bullish_ratio = (
            leg_in_candles["IsBullish"].mean()
            if "IsBullish" in leg_in_candles.columns else np.nan
        )

        # ── Base features ──
        base_length = max_base_length

        if not pd.isna(atr) and atr > 0:
            base_body_atr = (
                float(base_candles["CandleBody"].mean() / atr)
                if "CandleBody" in base_candles.columns else np.nan
            )
            base_range    = float(base_candles["High"].max() - base_candles["Low"].min())
            base_range_atr = base_range / atr
        else:
            base_body_atr  = np.nan
            base_range_atr = np.nan

        # Wick fraction: (range - body) / range per candle, then average
        if "CandleBody" in base_candles.columns and "CandleRange" in base_candles.columns:
            wick_vals = (base_candles["CandleRange"] - base_candles["CandleBody"]) / \
                        base_candles["CandleRange"].replace(0, np.nan)
            base_wick_frac = float(wick_vals.mean())
        else:
            base_wick_frac = np.nan

        base_volume_ratio = (
            float(base_candles["VolumeRatio"].mean())
            if "VolumeRatio" in base_candles.columns else np.nan
        )

        disp_base_ratio = (
            abs(leg_in_disp_atr) / base_range_atr
            if not np.isnan(leg_in_disp_atr) and not np.isnan(base_range_atr) and base_range_atr > 0
            else np.nan
        )

        # ── Market context ──
        atr_level          = float(prow.get("ATR", np.nan))
        close_vs_ema200    = _ema200_dist(end_idx, proc)
        close_vs_ema50     = _ema50_dist(end_idx, proc)
        volume_ratio       = float(prow.get("VolumeRatio", np.nan))
        rolling_volatility = float(prow.get("RollingVolatility", np.nan))
        body_to_atr        = float(prow.get("BodyToATR", np.nan))
        is_bullish_last    = int(prow.get("IsBullish", 0))

        row = {
            "zone_id":          "",
            "symbol":           "",
            "base_end_date":    str(prow["Date"].date()),
            "formation_date":   "",
            "example_type":     "negative",

            # ML features
            "leg_in_disp_atr":       leg_in_disp_atr,
            "leg_in_direction":      leg_in_direction,
            "leg_in_velocity":       leg_in_velocity,
            "leg_in_bullish_ratio":  leg_in_bullish_ratio,
            "base_length":           base_length,
            "base_body_atr":         base_body_atr,
            "base_wick_frac":        base_wick_frac,
            "base_range_atr":        base_range_atr,
            "base_volume_ratio":     base_volume_ratio,
            "disp_base_ratio":       disp_base_ratio,
            "atr_level":             atr_level,
            "close_vs_ema200":       close_vs_ema200,
            "close_vs_ema50":        close_vs_ema50,
            "volume_ratio":          volume_ratio,
            "rolling_volatility":    rolling_volatility,
            "body_to_atr":           body_to_atr,
            "is_bullish_last":       is_bullish_last,

            # Analysis-only (not available for negatives — set NaN)
            "departure_body_atr":          np.nan,
            "departure_close_ratio":       np.nan,
            "leg_out_disp_atr":            np.nan,
            "leg_out_velocity":            np.nan,
            "strength_ao":                 np.nan,
            "quality_score_ao":            np.nan,
            "trend_score_ao":              np.nan,
            "weekly_confluence_score_ao":  np.nan,

            # Labels
            "is_zone":   0,
            "structure": "no_zone",
        }
        rows.append(row)

    logger.info(f"Negative examples: {len(rows)} extracted")
    return pd.DataFrame(rows)


# ── Dataset assembly ────────────────────────────────────────────────────────

def build_training_dataset(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Combine positive and negative examples into one training dataset.
    Sort by base_end_date to maintain temporal order (critical for
    time-series cross-validation).
    """
    dataset = pd.concat([positive_df, negative_df], ignore_index=True)
    dataset  = dataset.sort_values("base_end_date").reset_index(drop=True)

    logger.info(
        f"Training dataset assembled: "
        f"{len(positive_df)} positive + {len(negative_df)} negative = "
        f"{len(dataset)} total rows"
    )

    # Class distribution
    dist = dataset["structure"].value_counts()
    logger.info(f"Class distribution:\n{dist.to_string()}")

    # Missing value report
    missing = dataset[ML_FEATURES].isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        logger.warning(
            f"Missing values in ML features:\n{missing.to_string()}"
        )

    return dataset


# ── I/O ────────────────────────────────────────────────────────────────────

def load_processed(symbol: str, cfg: dict) -> pd.DataFrame:
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    path = processed_dir / (_symbol_to_stem(symbol) + ".csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            "Run src/data/preprocessor.py first."
        )
    return pd.read_csv(path)


def load_zones(symbol: str, cfg: dict) -> pd.DataFrame:
    zones_dir = PROJECT_ROOT / cfg["data"]["zones_dir"]
    path = zones_dir / (_symbol_to_stem(symbol) + "_zones.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Zones not found: {path}\n"
            "Run src/zones/zone_detector.py first."
        )
    df = pd.read_csv(path)
    df["formation_date"]  = pd.to_datetime(df["formation_date"])
    df["base_start_date"] = pd.to_datetime(df["base_start_date"])
    df["base_end_date"]   = pd.to_datetime(df["base_end_date"])
    return df


def save_dataset(
    df: pd.DataFrame,
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
) -> Path:
    labeled_dir = _labeled_dir(cfg)
    path = labeled_dir / (_symbol_to_stem(symbol) + "_zone_windows.csv")
    df.to_csv(path, index=False)
    logger.info(f"Dataset saved → {path.relative_to(PROJECT_ROOT)}")
    return path


def save_feature_list(symbol: str, cfg: dict, logger: logging.Logger) -> Path:
    labeled_dir = _labeled_dir(cfg)
    path = labeled_dir / (_symbol_to_stem(symbol) + "_features.txt")
    with open(path, "w") as f:
        f.write("# ML feature columns for zone prediction Random Forest\n")
        f.write("# These are pre-departure features only (no look-ahead bias)\n\n")
        for feat in ML_FEATURES:
            f.write(feat + "\n")
        f.write("\n# Analysis-only post-departure features (DO NOT use as X)\n")
        for feat in ANALYSIS_ONLY_FEATURES:
            f.write(feat + "\n")
    logger.info(f"Feature list saved → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build zone-window training dataset for Random Forest."
    )
    parser.add_argument("--symbol",   type=str,  default=None)
    parser.add_argument(
        "--n_neg", type=int, default=300,
        help="Number of negative (non-zone) examples to sample. "
             "Default 300 gives a ~3:1 imbalance ratio; use class_weight='balanced' "
             "in RandomForestClassifier to handle it."
    )
    parser.add_argument("--seed",  type=int, default=42)
    args   = parser.parse_args()
    cfg    = load_config()
    logger = setup_logging(cfg)
    symbol = args.symbol or cfg["data"]["symbol"]

    logger.info(f"=== Zone Window Builder — {symbol} ===")

    proc     = load_processed(symbol, cfg)
    zones_df = load_zones(symbol, cfg)

    logger.info(f"Loaded {len(proc)} processed candles, {len(zones_df)} zones")

    # Read zone config for window sizes
    zone_cfg_path = PROJECT_ROOT / "config" / "zone_config.yaml"
    with open(zone_cfg_path) as f:
        zcfg = yaml.safe_load(f)
    leg_in_lookback  = zcfg["zones"]["leg_in"]["lookback"]    # 3
    max_base_length  = zcfg["zones"]["max_base_length"]        # 3

    logger.info(
        f"Window config: leg_in_lookback={leg_in_lookback}, "
        f"max_base_length={max_base_length}, "
        f"total window size={leg_in_lookback + max_base_length} candles"
    )

    positive_df = extract_positive_examples(zones_df, proc, logger)
    negative_df = extract_negative_examples(
        proc, zones_df,
        n_samples=args.n_neg,
        leg_in_lookback=leg_in_lookback,
        max_base_length=max_base_length,
        logger=logger,
        random_seed=args.seed,
    )

    dataset = build_training_dataset(positive_df, negative_df, logger)

    save_dataset(dataset, symbol, cfg, logger)
    save_feature_list(symbol, cfg, logger)

    # ── Console summary ──
    print("\n=== Zone Window Dataset Summary ===")
    print(f"Total rows        : {len(dataset)}")
    print(f"  Positive (zones): {dataset['is_zone'].sum()}")
    print(f"  Negative        : {(dataset['is_zone'] == 0).sum()}")
    print()
    print("Structure distribution:")
    for label, count in dataset["structure"].value_counts().items():
        print(f"  {label:<10}: {count}")
    print()
    print("ML feature set:")
    for f in ML_FEATURES:
        n_nan = dataset[f].isna().sum()
        print(f"  {f:<30} — {n_nan} NaN")
    print()
    print("Sample rows (first positive, first negative):")
    pos_sample = dataset[dataset["is_zone"] == 1].iloc[0]
    neg_sample = dataset[dataset["is_zone"] == 0].iloc[0]
    print("\n[Positive example]")
    print(pos_sample[["base_end_date", "structure"] + ML_FEATURES].to_string())
    print("\n[Negative example]")
    print(neg_sample[["base_end_date", "structure"] + ML_FEATURES].to_string())


if __name__ == "__main__":
    main()
