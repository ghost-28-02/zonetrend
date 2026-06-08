"""
zone_pipeline.py
================
Entry point for the ZoneTrend zone detection pipeline.
Requires processed data to exist (run data_pipeline.py first).

Runs:
    1. zone_detector — detect RBD/DBR supply and demand zones

Usage:
    python zone_pipeline.py
    python zone_pipeline.py --symbol TCS.NS
    python zone_pipeline.py --symbol ^NSEI

Output:
    data/zones/<SYMBOL>_zones.csv
"""
import argparse, sys
from pathlib import Path
import yaml

PROJECT_ROOT  = Path(__file__).resolve().parent
CONFIG_PATH   = PROJECT_ROOT / "config" / "config.yaml"
ZONE_CFG_PATH = PROJECT_ROOT / "config" / "zone_config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fetch_data     import setup_logging as fetch_logging
from src.zones.zone_detector import run_detection


def load_configs():
    with open(CONFIG_PATH) as f:   main_cfg = yaml.safe_load(f)
    with open(ZONE_CFG_PATH) as f: zone_cfg = yaml.safe_load(f)["zones"]
    return main_cfg, zone_cfg


def main():
    main_cfg, zone_cfg = load_configs()
    parser = argparse.ArgumentParser(description="ZoneTrend — zone detection pipeline")
    parser.add_argument("--symbol", default=main_cfg["data"]["symbol"],
                        help="Yahoo Finance ticker (default: value in config.yaml)")
    args   = parser.parse_args()
    symbol = args.symbol
    logger = fetch_logging(main_cfg)
    logger.name = "zone_pipeline"
    safe   = symbol.replace(".", "_").replace("^", "IDX_")

    logger.info("=" * 60)
    logger.info(f"ZoneTrend zone pipeline | symbol={symbol}")
    logger.info("=" * 60)

    logger.info("Detecting RBD/DBR zones...")
    if not run_detection(symbol, main_cfg, zone_cfg, logger):
        logger.error("Zone detection failed.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Zone pipeline complete for {symbol}")
    logger.info(f"  Zones : data/zones/{safe}_zones.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
