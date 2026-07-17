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
        "--timeframe",
        default=cfg["data"].get("timeframe", "1d"),
        help="Bar interval, e.g. 1d, 1wk, 1mo, 3mo or aliases like 1month "
             "(default: data.timeframe in config.yaml)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip download step; use existing raw CSV",
    )
    parser.add_argument(
        "--skip-weekly",
        action="store_true",
        help="Skip the weekly (1wk) timeframe fetch+preprocess",
    )
    args = parser.parse_args()

    symbol = args.symbol
    logger = fetch_logging(cfg)
    logger.name = "data_pipeline"

    timeframe = args.timeframe

    logger.info("=" * 60)
    logger.info(f"ZoneTrend data pipeline | symbol={symbol} | timeframe={timeframe}")
    logger.info("=" * 60)

    # ── Step 1: Fetch ─────────────────────────────────────────
    if args.skip_fetch:
        logger.info("Step 1/2 — fetch skipped (--skip-fetch)")
    else:
        logger.info(f"Step 1/2 — Fetching raw OHLCV data ({timeframe})...")
        ok = fetch(symbol, cfg, logger, interval=timeframe)
        if not ok:
            logger.error("Fetch failed. Aborting pipeline.")
            sys.exit(1)

    # ── Step 2: Preprocess (daily) ────────────────────────────
    logger.info("Step 2/3 — Preprocessing daily...")
    ok = run_preprocess(symbol, cfg, logger)
    if not ok:
        logger.error("Preprocessing failed. Aborting pipeline.")
        sys.exit(1)

    # ── Step 3: Weekly timeframe ──────────────────────────────
    # Weekly OHLCV is fetched DIRECTLY from Yahoo Finance (interval=1wk), not
    # resampled from daily. It powers the weekly-confluence features in the
    # zone detector. Weekly failures are non-fatal: the detector falls back to
    # resampling if the weekly processed file is missing.
    fetch_weekly = cfg["data"].get("fetch_weekly", True) and not args.skip_weekly
    safe = symbol.replace(".", "_").replace("^", "IDX_")
    if fetch_weekly:
        wk_interval = cfg["data"].get("weekly_interval", "1wk")
        wk_min      = cfg["preprocessing"].get("min_candles_weekly", 60)
        logger.info(f"Step 3/3 — Weekly timeframe ({wk_interval})...")
        if args.skip_fetch:
            logger.info("  Weekly fetch skipped (--skip-fetch)")
        elif not fetch(symbol, cfg, logger, interval=wk_interval, suffix="_weekly"):
            logger.warning("  Weekly fetch failed; detector will resample as fallback.")
        if not run_preprocess(symbol, cfg, logger, suffix="_weekly",
                              min_candles_override=wk_min):
            logger.warning("  Weekly preprocess failed; detector will resample as fallback.")
    else:
        logger.info("Step 3/3 — Weekly timeframe skipped.")

    logger.info("=" * 60)
    logger.info(f"Data pipeline complete for {symbol}")
    logger.info(f"  Raw daily       : data/raw/{safe}.csv")
    logger.info(f"  Processed daily : data/processed/{safe}.csv")
    if fetch_weekly:
        logger.info(f"  Processed weekly: data/processed/{safe}_weekly.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
