"""
data_pipeline.py
================
Entry point for the ZoneTrend data pipeline.

Runs in order:
    1. fetch_data    — download raw OHLCV from Yahoo Finance
    2. preprocessor  — clean data and compute derived columns

The active symbol is set in config/config.yaml (data.symbol).
Override it on the command line with --symbol.

Usage
-----
    # Use symbol from config.yaml
    python data_pipeline.py

    # Override symbol
    python data_pipeline.py --symbol TCS.NS
    python data_pipeline.py --symbol ^NSEI

    # Skip fetch if raw data already exists
    python data_pipeline.py --skip-fetch

Output
------
    data/raw/<SYMBOL>.csv        — raw OHLCV
    data/processed/<SYMBOL>.csv  — cleaned + derived columns (28 cols)
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

sys.path.insert(0, str(PROJECT_ROOT))
from src.data.fetch_data   import fetch, setup_logging as fetch_logging
from src.data.preprocessor import run_preprocess


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="ZoneTrend — data pipeline: fetch + preprocess"
    )
    parser.add_argument(
        "--symbol",
        default=cfg["data"]["symbol"],
        help="Yahoo Finance ticker (default: value in config.yaml)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip download step; use existing raw CSV",
    )
    args = parser.parse_args()

    symbol = args.symbol
    logger = fetch_logging(cfg)
    logger.name = "data_pipeline"

    logger.info("=" * 60)
    logger.info(f"ZoneTrend data pipeline | symbol={symbol}")
    logger.info("=" * 60)

    # ── Step 1: Fetch ─────────────────────────────────────────
    if args.skip_fetch:
        logger.info("Step 1/2 — fetch skipped (--skip-fetch)")
    else:
        logger.info("Step 1/2 — Fetching raw OHLCV data...")
        ok = fetch(symbol, cfg, logger)
        if not ok:
            logger.error("Fetch failed. Aborting pipeline.")
            sys.exit(1)

    # ── Step 2: Preprocess ────────────────────────────────────
    logger.info("Step 2/2 — Preprocessing...")
    ok = run_preprocess(symbol, cfg, logger)
    if not ok:
        logger.error("Preprocessing failed. Aborting pipeline.")
        sys.exit(1)

    safe = symbol.replace(".", "_").replace("^", "IDX_")
    logger.info("=" * 60)
    logger.info(f"Data pipeline complete for {symbol}")
    logger.info(f"  Raw data  : data/raw/{safe}.csv")
    logger.info(f"  Processed : data/processed/{safe}.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
