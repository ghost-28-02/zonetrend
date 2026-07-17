"""
zone_pipeline.py
================
ZoneTrend pipeline — zone detection → ML training → ML-based zone detection.

Steps
-----
    1. zone_detector        — rule-based detection (generates ground-truth labels)
    2. zone_labeler         — stamp zone-context columns onto every candle
    3. candle_window_builder— build sliding 20-candle window training dataset
    4. zone_detection_model — train XGBoost to detect zones from windows
    5. ml_zone_detector     — run model to produce ML-detected zones

Backtesting has been removed. The research focus is on zone detection quality.

Skip flags:
    --skip-detection   skip step 1 (rule-based zones already exist)
    --skip-labeler     skip step 2
    --skip-windows     skip step 3
    --skip-model       skip step 4
    --skip-ml-detect   skip step 5

Usage
-----
    python zone_pipeline.py                          # full run
    python zone_pipeline.py --skip-detection         # zones already exist
    python zone_pipeline.py --skip-detection --skip-labeler --skip-windows
    python zone_pipeline.py --symbol RELIANCE.NS

Output
------
    data/zones/<SYMBOL>_zones.csv             step 1  (rule-based ground truth)
    data/labeled/<SYMBOL>_labeled.csv         step 2
    data/labeled/<SYMBOL>_zone_windows_v2.csv step 3  (window training dataset)
    data/models/<SYMBOL>_zone_xgb.pkl         step 4  (trained XGBoost)
    data/models/<SYMBOL>_zone_xgb_report.txt  step 4
    data/zones/<SYMBOL>_ml_zones.csv          step 5  (ML-detected zones)
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT  = Path(__file__).resolve().parent
CONFIG_PATH   = PROJECT_ROOT / "config" / "config.yaml"
ZONE_CFG_PATH = PROJECT_ROOT / "config" / "zone_config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_data               import setup_logging as fetch_logging
from src.zones.zone_detector           import run_detection
from src.features.zone_labeler         import (
    ZoneLabeler, load_processed, load_zones as load_zones_labeler,
    save_labeled, save_zone_summary, build_zone_summary,
)
from src.features.candle_window_builder import run_window_builder
from src.models.zone_detection_model    import run_model_training
from src.zones.ml_zone_detector         import run_ml_detection


def load_configs():
    with open(CONFIG_PATH) as f:
        main_cfg = yaml.safe_load(f)
    with open(ZONE_CFG_PATH) as f:
        zone_cfg = yaml.safe_load(f)["zones"]
    return main_cfg, zone_cfg


# ── Step functions ───────────────────────────────────────────────────────────

def step_detection(symbol, main_cfg, zone_cfg, logger):
    logger.info("─" * 60)
    logger.info("STEP 1 — Rule-Based Zone Detection (ground truth labels)")
    logger.info("─" * 60)
    if not run_detection(symbol, main_cfg, zone_cfg, logger):
        logger.error("Zone detection failed — aborting.")
        sys.exit(1)
    logger.info("Step 1 complete.")


def step_labeler(symbol, main_cfg, logger):
    logger.info("─" * 60)
    logger.info("STEP 2 — Zone Labeler")
    logger.info("─" * 60)
    processed_df = load_processed(symbol, main_cfg)
    zones_df     = load_zones_labeler(symbol, main_cfg)
    labeler      = ZoneLabeler(main_cfg, logger=logger)
    labeled_df   = labeler.run(processed_df, zones_df)
    save_labeled(labeled_df, symbol, main_cfg, logger)
    save_zone_summary(build_zone_summary(labeled_df), symbol, main_cfg, logger)
    logger.info("Step 2 complete.")


def step_windows(symbol, main_cfg, logger):
    logger.info("─" * 60)
    logger.info("STEP 3 — Candle Window Builder (20-candle sliding windows)")
    logger.info("─" * 60)
    run_window_builder(symbol, main_cfg, logger)
    logger.info("Step 3 complete.")


def step_model(symbol, main_cfg, n_splits, logger):
    logger.info("─" * 60)
    logger.info("STEP 4 — Zone Detection Model Training (XGBoost)")
    logger.info("─" * 60)
    run_model_training(symbol, main_cfg, logger, n_splits=n_splits)
    logger.info("Step 4 complete.")


def step_ml_detect(symbol, main_cfg, threshold, logger):
    logger.info("─" * 60)
    logger.info("STEP 5 — ML Zone Detection")
    logger.info("─" * 60)
    run_ml_detection(symbol, main_cfg, logger, threshold=threshold)
    logger.info("Step 5 complete.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    main_cfg, zone_cfg = load_configs()

    parser = argparse.ArgumentParser(
        description="ZoneTrend — detect → label → train → ML-detect"
    )
    parser.add_argument("--symbol",          default=main_cfg["data"]["symbol"])
    parser.add_argument("--n-splits",        type=int,   default=4,
                        help="Walk-forward CV folds (default 4)")
    parser.add_argument("--threshold",       type=float, default=0.45,
                        help="P(zone) threshold for ML detector (default 0.45)")
    parser.add_argument("--skip-detection",  action="store_true")
    parser.add_argument("--skip-labeler",    action="store_true")
    parser.add_argument("--skip-windows",    action="store_true")
    parser.add_argument("--skip-model",      action="store_true")
    parser.add_argument("--skip-ml-detect",  action="store_true")

    args   = parser.parse_args()
    symbol = args.symbol
    logger = fetch_logging(main_cfg)
    logger.name = "zone_pipeline"
    safe   = symbol.replace(".", "_").replace("^", "IDX_")

    logger.info("=" * 60)
    logger.info(f"ZoneTrend pipeline  |  symbol = {symbol}")
    logger.info("=" * 60)

    if not args.skip_detection:
        step_detection(symbol, main_cfg, zone_cfg, logger)
    else:
        logger.info("STEP 1 — Rule-Based Detection  [skipped]")

    if not args.skip_labeler:
        step_labeler(symbol, main_cfg, logger)
    else:
        logger.info("STEP 2 — Zone Labeler  [skipped]")

    if not args.skip_windows:
        step_windows(symbol, main_cfg, logger)
    else:
        logger.info("STEP 3 — Window Builder  [skipped]")

    if not args.skip_model:
        step_model(symbol, main_cfg, args.n_splits, logger)
    else:
        logger.info("STEP 4 — Model Training  [skipped]")

    if not args.skip_ml_detect:
        step_ml_detect(symbol, main_cfg, args.threshold, logger)
    else:
        logger.info("STEP 5 — ML Detection  [skipped]")

    logger.info("=" * 60)
    logger.info(f"Pipeline complete  |  symbol = {symbol}")
    logger.info(f"  Rule zones : data/zones/{safe}_zones.csv")
    logger.info(f"  Windows    : data/labeled/{safe}_zone_windows_v2.csv")
    logger.info(f"  Model      : data/models/{safe}_zone_xgb.pkl")
    logger.info(f"  ML zones   : data/zones/{safe}_ml_zones.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
