"""
run_data_pipeline.py
====================
Single entry point to run the entire data pipeline in one command.

Steps executed in order:
    1. fetch_data.py    — Download OHLCV data from Yahoo Finance
    2. preprocessor.py  — Clean data and compute ATR, indicators, candle features

Usage:
    # Run dev symbols only (fast, 9 symbols) — recommended for first run
    python run_data_pipeline.py

    # Run all Nifty 50 symbols
    python run_data_pipeline.py --mode full

    # Run specific symbols
    python run_data_pipeline.py --symbols RELIANCE.NS TCS.NS HDFCBANK.NS
"""

import sys
import argparse
import logging
from pathlib import Path

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.data.fetch_data    import fetch_all,    setup_logging as fetch_logging
from src.data.preprocessor  import process_all,  setup_logging as proc_logging


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ZoneTrend — Full Data Pipeline")
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="dev",
        help="'dev' = 9 symbols (fast). 'full' = all Nifty 50 symbols.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Override: specific symbols e.g. RELIANCE.NS TCS.NS",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip download step (use existing raw data). Useful if already downloaded.",
    )
    args = parser.parse_args()

    cfg = load_config()

    # Determine symbol list
    if args.symbols:
        symbols = args.symbols
        label   = "custom"
    elif args.mode == "dev":
        symbols = cfg["data"]["dev_symbols"]
        label   = "dev"
    else:
        symbols = cfg["data"]["symbols"]
        label   = "full"

    print("=" * 60)
    print("ZoneTrend — Data Pipeline")
    print("=" * 60)
    print(f"Mode    : {label}")
    print(f"Symbols : {len(symbols)}")
    print(f"Period  : {cfg['data']['start_date']} → {cfg['data']['end_date']}")
    print("=" * 60)

    # ── Step 1: Fetch ──────────────────────────────────────────
    if args.skip_fetch:
        print("\n[STEP 1/2] FETCH — Skipped (--skip-fetch flag set)")
    else:
        print("\n[STEP 1/2] FETCH — Downloading OHLCV from Yahoo Finance...")
        logger = fetch_logging(cfg)
        success, failed = fetch_all(symbols, cfg, logger)
        print(f"           Done: {success}/{len(symbols)} downloaded")
        if failed:
            print(f"           Failed: {failed}")
            # Continue with whatever was downloaded successfully

    # ── Step 2: Preprocess ─────────────────────────────────────
    print("\n[STEP 2/2] PREPROCESS — Cleaning data and computing features...")
    logger = proc_logging(cfg)
    process_all(symbols, cfg, logger)
    print("           Done.")

    # ── Summary ────────────────────────────────────────────────
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    files = list(processed_dir.glob("*.csv"))
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Processed files saved in: {processed_dir.relative_to(PROJECT_ROOT)}/")
    for f in sorted(files):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:35s} {size_kb:5d} KB")
    print("=" * 60)
    print()
    print("Next step:")
    print("  jupyter notebook notebooks/01_data_exploration.ipynb")


if __name__ == "__main__":
    main()
