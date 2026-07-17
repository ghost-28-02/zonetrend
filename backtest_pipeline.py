"""
backtest_pipeline.py
====================
Entry point for the ZoneTrend P&L backtesting pipeline.

Runs the pnl_backtester on detected supply/demand zones and reports
actual Profit & Loss in Indian Rupees (₹).

Requires
--------
    data/processed/<SYMBOL>.csv   — run data_pipeline.py first
    data/zones/<SYMBOL>_zones.csv — run zone_pipeline.py first

Usage
-----
    # Default: ML (XGBoost) zones, ₹5L capital, 1% risk, 2:1 RR, 5-day hold
    python backtest_pipeline.py

    # ── Recommended settings (ML zones + filters) ──
    python backtest_pipeline.py --confirm --min-strength 0.7 --rr 3.0

    # Use rule-based (algo) zones instead of ML zones
    python backtest_pipeline.py --algo-zones

    # Custom capital and risk
    python backtest_pipeline.py --capital 1000000 --risk-pct 0.02

    # Different symbol
    python backtest_pipeline.py --symbol RELIANCE.NS

    # Adjust RR ratio and max hold period
    python backtest_pipeline.py --rr 3.0 --max-hold 10

    # Only trade high-quality zones
    python backtest_pipeline.py --min-strength 0.65 --max-strength 0.80

    # Wait for a confirmation/rejection candle before entering
    python backtest_pipeline.py --confirm

    # Add slippage and brokerage commission
    python backtest_pipeline.py --slippage 0.001 --commission 20

    # Enable trailing stop (overrides config.yaml trailing_stop.enabled)
    python backtest_pipeline.py --trail-atr-mult 1.5

    # Full improved run
    python backtest_pipeline.py --confirm --min-strength 0.65 --max-strength 0.80 --rr 3.0 --max-hold 10

Output
------
    data/backtest/<SYMBOL>_pnl_trades.csv     — one row per trade with ₹ P&L
    data/backtest/<SYMBOL>_pnl_report.txt     — full text summary report
    data/backtest/<SYMBOL>_equity_curve.csv   — running capital after each trade
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

sys.path.insert(0, str(PROJECT_ROOT))
from src.backtesting.pnl_backtester import run_pnl_backtest, setup_logging


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # Read backtest defaults from config.yaml [backtest] section.
    # CLI flags always override config values when explicitly provided.
    bt = cfg.get("backtest", {})
    cfg_capital      = bt.get("start_capital",    500_000.0)
    cfg_risk_pct     = bt.get("risk_pct",         0.01)
    cfg_rr           = bt.get("rr_ratio",         3.0)
    cfg_max_hold     = bt.get("max_hold_candles", 5)
    cfg_slippage     = bt.get("slippage_pct",     0.0)
    cfg_commission   = bt.get("commission",       0.0)
    cfg_confirm        = bt.get("confirm_entry",    True)
    cfg_min_strength   = bt.get("min_strength",     0.7)
    cfg_max_strength   = bt.get("max_strength",     1.0)
    cfg_zone_source    = bt.get("zone_source",      "ml")   # "ml" or "algo"
    cfg_structures     = bt.get("trade_structures", ["DBR", "RBD", "RBR", "DBD"])

    # Trailing stop — read from config.yaml [backtest.trailing_stop]
    # trail_atr_mult = 0.0 means disabled (fixed zone stop)
    ts_cfg              = bt.get("trailing_stop", {})
    ts_enabled          = ts_cfg.get("enabled",         False)
    ts_atr_mult         = ts_cfg.get("atr_mult",        1.5)
    ts_trigger_r        = ts_cfg.get("trail_trigger_r", 2.0)
    ts_floor_atr_mult   = ts_cfg.get("floor_atr_mult",  0.5)
    ts_stop_atr_mult    = ts_cfg.get("stop_atr_mult",   0.0)
    cfg_trail_atr_mult  = ts_atr_mult if ts_enabled else 0.0
    cfg_trail_trigger_r = ts_trigger_r
    cfg_floor_atr_mult  = ts_floor_atr_mult
    cfg_stop_atr_mult   = ts_stop_atr_mult

    parser = argparse.ArgumentParser(
        description="ZoneTrend — P&L backtest on supply/demand zones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",
        default=cfg["data"]["symbol"],
        help="Yahoo Finance ticker (default: value in config.yaml)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=cfg_capital,
        metavar="INR",
        help="Starting capital in ₹ (default from config.yaml)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=cfg_risk_pct,
        metavar="FRACTION",
        help="Fraction of capital to risk per trade (default from config.yaml)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=cfg_rr,
        metavar="RATIO",
        help="Reward-to-risk ratio for take-profit (default from config.yaml)",
    )
    parser.add_argument(
        "--max-hold",
        type=int,
        default=cfg_max_hold,
        metavar="DAYS",
        help="Maximum days to hold a trade before forced exit (default from config.yaml)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=cfg_slippage,
        metavar="FRACTION",
        help="Entry slippage as fraction (default from config.yaml)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=cfg_commission,
        metavar="INR",
        help="Flat ₹ brokerage per trade side (default from config.yaml)",
    )
    parser.add_argument(
        "--trail-atr-mult",
        type=float,
        default=cfg_trail_atr_mult,
        metavar="MULT",
        help=(
            "Trail factor: ATR multiplier for trail distance in Phase 2 (default from config.yaml). "
            "0 = disabled (fixed zone stop). "
            "Example: --trail-atr-mult 1.5  trails stop 1.5×ATR behind peak price. "
            "Tip: set trailing_stop.enabled: true in config.yaml to always use it."
        ),
    )
    parser.add_argument(
        "--trail-trigger-r",
        type=float,
        default=cfg_trail_trigger_r,
        metavar="R",
        help=(
            "R-multiple at which trail activates (default from config.yaml). "
            "2.0 = trail activates when price hits 2R target (safest, min win = 2R). "
            "1.0 = trail activates at 1R (min win = 1R, may cut some 2R winners). "
            "Only used when trail-atr-mult > 0."
        ),
    )
    parser.add_argument(
        "--floor-atr-mult",
        type=float,
        default=cfg_floor_atr_mult,
        metavar="MULT",
        help=(
            "Floor buffer below trigger_price (default from config.yaml). "
            "floor = trigger_price − floor_atr_mult × ATR. "
            "0.0 = no floor (pure ATR trail). "
            "0.5 = half-ATR buffer — avoids immediate exit on activation candle (recommended). "
            "Only used when trail-atr-mult > 0."
        ),
    )
    parser.add_argument(
        "--stop-atr-mult",
        type=float,
        default=cfg_stop_atr_mult,
        metavar="MULT",
        help=(
            "Stop factor: ATR multiplier for Phase 1 initial stop (default from config.yaml). "
            "0 = use zone distal as stop (original behaviour). "
            ">0 = initial stop = entry ± stop_atr_mult × ATR (overrides zone distal). "
            "Example: --stop-atr-mult 1.5"
        ),
    )
    parser.add_argument(
        "--algo-zones",
        action="store_true",
        default=(cfg_zone_source == "algo"),
        help="Use rule-based (algo) zones instead of ML (XGBoost) zones",
    )
    parser.add_argument(
        "--confirm",
        action=argparse.BooleanOptionalAction,
        default=cfg_confirm,
        help="Wait for a rejection candle before entering (default from config.yaml)",
    )
    parser.add_argument(
        "--min-strength",
        type=float,
        default=cfg_min_strength,
        metavar="SCORE",
        help="Minimum zone quality score to trade (default from config.yaml)",
    )
    parser.add_argument(
        "--max-strength",
        type=float,
        default=cfg_max_strength,
        metavar="SCORE",
        help="Maximum zone quality score to trade (default from config.yaml)",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        default=cfg_structures,
        metavar="STRUCTURE",
        help=(
            "Zone structures to trade. Space-separated list from: DBR RBD RBR DBD. "
            "Default from config.yaml. "
            "Example: --structures DBR RBD  (trade only reversal zones)"
        ),
    )
    args   = parser.parse_args()
    symbol = args.symbol

    logger = setup_logging(cfg)
    logger.name = "backtest_pipeline"

    safe = symbol.replace(".", "_").replace("^", "IDX_")

    if args.trail_atr_mult > 0:
        trail_log = (
            f"{args.trail_atr_mult}×ATR trail"
            f" | trigger={args.trail_trigger_r}R"
            f" | floor=trigger−{args.floor_atr_mult}×ATR"
            + (f" | init_stop={args.stop_atr_mult}×ATR" if args.stop_atr_mult > 0 else "")
        )
    else:
        trail_log = "OFF (fixed zone stop + fixed target)"

    logger.info("=" * 60)
    logger.info(f"ZoneTrend P&L Backtest Pipeline | symbol={symbol}")
    logger.info(f"  Capital      : ₹{args.capital:,.0f}")
    logger.info(f"  Risk/trade   : {args.risk_pct:.1%}")
    logger.info(f"  RR ratio     : {args.rr}:1")
    logger.info(f"  Max hold     : {args.max_hold} days")
    logger.info(f"  Zone source  : {'Rule-based (algo)' if args.algo_zones else 'ML-detected (XGBoost)'}")
    logger.info(f"  Confirm entry: {'YES' if args.confirm else 'NO'}")
    logger.info(f"  Trailing stop: {trail_log}")
    logger.info(f"  Strength     : [{args.min_strength:.2f} – {args.max_strength:.2f}]")
    logger.info(f"  Structures   : {', '.join(args.structures)}")
    logger.info(f"  Slippage     : {args.slippage:.3%}")
    logger.info(f"  Commission   : ₹{args.commission:.0f}/side")
    logger.info("=" * 60)

    # ── Preflight checks ──────────────────────────────────────────────────────
    processed_path = PROJECT_ROOT / cfg["data"]["processed_dir"] / f"{safe}.csv"
    zones_path     = PROJECT_ROOT / cfg["data"]["zones_dir"]     / f"{safe}_zones.csv"

    if not processed_path.exists():
        logger.error(f"Processed data not found: {processed_path}")
        logger.error("Run  python data_pipeline.py  first.")
        sys.exit(1)

    if not zones_path.exists():
        logger.error(f"Zones file not found: {zones_path}")
        logger.error("Run  python zone_pipeline.py  first.")
        sys.exit(1)

    logger.info(f"Processed data : {processed_path.relative_to(PROJECT_ROOT)}")
    logger.info(f"Zones file     : {zones_path.relative_to(PROJECT_ROOT)}")

    # ── Run backtest ──────────────────────────────────────────────────────────
    trades_df, metrics = run_pnl_backtest(
        symbol           = symbol,
        cfg              = cfg,
        logger           = logger,
        start_capital    = args.capital,
        risk_pct         = args.risk_pct,
        rr_ratio         = args.rr,
        max_hold_candles = args.max_hold,
        slippage_pct     = args.slippage,
        commission       = args.commission,
        confirm_entry    = args.confirm,
        min_strength     = args.min_strength,
        max_strength     = args.max_strength,
        trade_structures = args.structures,
        trail_atr_mult   = args.trail_atr_mult,
        trail_trigger_r  = args.trail_trigger_r,
        floor_atr_mult   = args.floor_atr_mult,
        stop_atr_mult    = args.stop_atr_mult,
        use_ml_zones     = not args.algo_zones,
    )

    if not metrics:
        logger.error("Backtest produced no results.")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    trail_exits = metrics.get("trail_wins", 0)
    trail_pnl   = metrics.get("trail_win_pnl_inr", 0.0)

    logger.info("=" * 60)
    logger.info(f"Backtest complete for {symbol}")
    logger.info(f"  Starting capital : ₹{metrics['start_capital']:,.2f}")
    logger.info(f"  Ending capital   : ₹{metrics['final_capital']:,.2f}")
    logger.info(f"  Net P&L          : ₹{metrics['total_return_inr']:+,.2f}  "
                f"({metrics['total_return_pct']:+.2f}%)")
    logger.info(f"  Win rate         : {metrics['win_rate']:.1%}  "
                f"({metrics['wins']}W / {metrics['losses']}L)")
    if args.trail_atr_mult > 0 and trail_exits > 0:
        logger.info(f"  Trail wins (>2R) : {trail_exits}  "
                    f"(P&L: ₹{trail_pnl:+,.2f})")
    logger.info(f"  Max Drawdown     : ₹{metrics['max_drawdown_inr']:,.2f}  "
                f"({metrics['max_drawdown_pct']:.2f}%)")
    logger.info(f"  Trades saved     : data/backtest/{safe}_pnl_trades.csv")
    logger.info(f"  Report saved     : data/backtest/{safe}_pnl_report.txt")
    logger.info(f"  Equity curve     : data/backtest/{safe}_equity_curve.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
