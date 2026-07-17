"""
zone_labeler.py
===============
Merges processed OHLCV data with detected zones and stamps each candle
with zone-context columns ready for Random Forest feature engineering.

What this module does
---------------------
After zone detection produces a zones DataFrame (one row per zone with
columns like formation_date, structure, type, top, bottom, strength),
this module answers the question: "For each candle in the processed data,
what zone context was present at that moment in time?"

It produces three categories of columns:

  1. Formation-stamp columns
     On the exact candle where a zone was formed (formation_date), we stamp
     the structure label (DBR / RBD / RBR / DBD) and a supply/demand flag.
     Multiple zones can form on the same date — see conflict_strategy below.

  2. Active-zone context columns (causal — no look-ahead)
     For every candle we scan all zones whose formation_date <= candle date
     AND whose status was still active at that date (i.e. not yet broken or
     invalidated). We record the nearest active demand zone above and the
     nearest active supply zone above, plus their distances and strengths.

  3. One-hot encoding of zone structure
     Four binary columns (is_DBR, is_RBD, is_RBR, is_DBD) mark whether a
     zone of that structure was formed on this candle. These are the direct
     inputs your Random Forest will use for zone-type context.

Why zone_type is a feature, not a label
----------------------------------------
DBR / RBD / RBR / DBD describe the price pattern that CREATED the zone.
That label is determined structurally (drop before base? rally after?),
not by future price behaviour. Using it as the model target would make
the model re-derive zone detection — circular and not predictive.

Use zone_type (one-hot) as a FEATURE alongside strength, ATR distance,
weekly confluence, etc.  The LABEL for your Random Forest should be a
forward-looking outcome: did price reverse or break when it next touched
this zone?  That label is generated in a separate label-generation step
(src/features/outcome_labeler.py — to be built next).

No-look-ahead guarantee
------------------------
All zone context is computed using only zones whose formation_date is
strictly <= the current candle date.  Status tracking uses only the
zone's own recorded invalidation_date column, which was derived from
historical price closes after the zone formed.  We never use future
price information.

Output columns added to processed data
---------------------------------------
  zone_formed_structure      str | NaN  — DBR / RBD / RBR / DBD if a zone
                                          formed today, else NaN
  zone_formed_type           str | NaN  — 'demand' | 'supply' | NaN
  zone_formed_strength       float      — strength of the zone formed today
                                          (0.0 if no zone formed)
  zone_formed_count          int        — number of zones formed today (0,1,2+)
  is_DBR                     int        — 1 if DBR zone formed today
  is_RBD                     int        — 1 if RBD zone formed today
  is_RBR                     int        — 1 if RBR zone formed today
  is_DBD                     int        — 1 if DBD zone formed today
  active_demand_count        int        — number of active demand zones at EOD
  active_supply_count        int        — number of active supply zones at EOD
  nearest_demand_top         float      — top of closest active demand zone
  nearest_demand_bottom      float      — bottom of closest active demand zone
  nearest_demand_dist_pct    float      — (close - demand_top) / close × 100
  nearest_demand_strength    float      — ML-safe strength of nearest demand
  nearest_demand_structure   str | NaN  — DBR or RBR
  nearest_supply_bottom      float      — bottom of closest active supply zone
  nearest_supply_top         float      — top of closest active supply zone
  nearest_supply_dist_pct    float      — (supply_bottom - close) / close × 100
  nearest_supply_strength    float      — ML-safe strength of nearest supply
  nearest_supply_structure   str | NaN  — RBD or DBD
  price_in_demand_zone       int        — 1 if close is inside any active demand
  price_in_supply_zone       int        — 1 if close is inside any active supply
  zone_confluence            int        — 1 if both demand and supply zones are
                                          active simultaneously

Usage
-----
    # Run standalone (uses config.yaml symbol)
    python src/features/zone_labeler.py

    # Override symbol
    python src/features/zone_labeler.py --symbol RELIANCE.NS

    # Import in other modules
    from src.features.zone_labeler import ZoneLabeler
    labeler = ZoneLabeler(config)
    labeled_df = labeler.run(processed_df, zones_df)

Output files
------------
  data/labeled/<SYMBOL>_labeled.csv   — processed data with zone columns
  data/labeled/<SYMBOL>_zone_summary.csv — per-zone summary (for inspection)
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
        fh = logging.FileHandler(log_dir / "zone_labeler.log")
        fh.setLevel(level)
        handlers.append(fh)

    if cfg["logging"].get("log_to_console", True):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        handlers.append(ch)

    logging.basicConfig(level=level, handlers=handlers,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("zone_labeler")


# ── Path helpers ────────────────────────────────────────────────────────────

def _symbol_to_stem(symbol: str) -> str:
    """'^NSEI' → 'IDX_NSEI',  'RELIANCE.NS' → 'RELIANCE_NS'."""
    if symbol.startswith("^"):
        return "IDX_" + symbol[1:].replace(".", "_")
    return symbol.replace(".", "_")


def _processed_path(symbol: str, cfg: dict) -> Path:
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    return processed_dir / (_symbol_to_stem(symbol) + ".csv")


def _zones_path(symbol: str, cfg: dict) -> Path:
    zones_dir = PROJECT_ROOT / cfg["data"]["zones_dir"]
    return zones_dir / (_symbol_to_stem(symbol) + "_zones.csv")


def _labeled_dir(cfg: dict) -> Path:
    # Falls back to data/labeled/ if not in config
    labeled_dir = PROJECT_ROOT / cfg.get("features", {}).get(
        "labeled_dir", "data/labeled"
    )
    labeled_dir.mkdir(parents=True, exist_ok=True)
    return labeled_dir


# ── Core labeler class ──────────────────────────────────────────────────────

class ZoneLabeler:
    """
    Merges processed OHLCV data with zone detections and stamps each
    candle with zone-context features.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    conflict_strategy : str
        How to handle multiple zones forming on the same date.
        'strongest'  — keep the zone with highest strength (default).
        'first'      — keep the first zone by base_start_date.
        'all'        — keep all (zone_formed_structure becomes
                       comma-separated, less clean for ML).
    logger : logging.Logger | None
    """

    def __init__(
        self,
        config: dict,
        conflict_strategy: str = "strongest",
        logger: logging.Logger | None = None,
    ):
        self.cfg               = config
        self.conflict_strategy = conflict_strategy
        self.logger            = logger or logging.getLogger("zone_labeler")

    # ── Public entry point ─────────────────────────────────────────────────

    def run(
        self,
        processed_df: pd.DataFrame,
        zones_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Stamp zone-context columns onto the processed DataFrame.

        Parameters
        ----------
        processed_df : pd.DataFrame
            Output of preprocessor.py.  Must have a 'Date' column
            (string or datetime) and a 'Close' column.
        zones_df : pd.DataFrame
            Output of zone_detector.py.  Must have columns:
            formation_date, structure, type, top, bottom,
            strength, status, invalidation_date.

        Returns
        -------
        pd.DataFrame
            A copy of processed_df with zone-context columns appended.
        """
        proc  = self._prepare_processed(processed_df)
        zones = self._prepare_zones(zones_df)

        self.logger.info(
            f"Labeling {len(proc)} candles against {len(zones)} zones "
            f"(conflict_strategy='{self.conflict_strategy}')"
        )

        proc = self._stamp_formation(proc, zones)
        proc = self._stamp_active_context(proc, zones)
        proc = self._stamp_price_position(proc, zones)

        # One-hot encoding of formed structure (primary ML feature columns)
        proc = self._one_hot_structure(proc)

        self.logger.info(
            f"Zone labeling complete. Columns added: {self._new_columns()}"
        )
        return proc

    # ── Preparation ────────────────────────────────────────────────────────

    def _prepare_processed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def _prepare_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["formation_date"]    = pd.to_datetime(df["formation_date"])
        df["invalidation_date"] = pd.to_datetime(
            df["invalidation_date"], errors="coerce"
        )

        # Validate structure labels
        invalid_struct = df[~df["structure"].isin(STRUCTURE_LABELS)]
        if len(invalid_struct):
            self.logger.warning(
                f"{len(invalid_struct)} zones have unknown structure labels "
                f"and will be skipped: {invalid_struct['structure'].unique()}"
            )
            df = df[df["structure"].isin(STRUCTURE_LABELS)].copy()

        # Use ML-safe strength (never adjusted_strength_posthoc which leaks)
        if "strength_pit" in df.columns:
            df["_strength_ml"] = df["strength_pit"]
        else:
            df["_strength_ml"] = df["strength"]

        return df.reset_index(drop=True)

    # ── Step 1: Formation stamp ─────────────────────────────────────────────

    def _stamp_formation(
        self,
        proc: pd.DataFrame,
        zones: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each candle date, record which zone (if any) was formed.

        A zone is considered 'formed' on formation_date — the close of the
        departure candle (no look-ahead: the zone is only known after the
        departure candle closes, which is exactly what formation_date
        captures in zone_detector.py).

        Conflict resolution (multiple zones same date):
          'strongest' — zone with highest _strength_ml wins
          'first'     — zone with the earliest base_start_date wins
        """
        # Initialise formation columns (object dtype for string columns avoids
        # pandas FutureWarning when assigning str to a float-initialised column)
        proc["zone_formed_structure"] = pd.array([None] * len(proc), dtype=object)
        proc["zone_formed_type"]      = pd.array([None] * len(proc), dtype=object)
        proc["zone_formed_strength"]  = 0.0
        proc["zone_formed_count"]     = 0

        grouped = zones.groupby("formation_date")

        for date, group in grouped:
            mask = proc["Date"] == date
            if not mask.any():
                # Zone formed on a non-trading day (holiday) — skip
                self.logger.debug(f"Zone formed on non-trading date {date} — skipped")
                continue

            count = len(group)
            idx   = proc.index[mask][0]
            proc.at[idx, "zone_formed_count"] = count

            if count == 1:
                row = group.iloc[0]
            elif self.conflict_strategy == "strongest":
                row = group.loc[group["_strength_ml"].idxmax()]
            elif self.conflict_strategy == "first":
                row = group.sort_values("base_start_date").iloc[0]
            else:
                # 'strongest' as default fallback
                row = group.loc[group["_strength_ml"].idxmax()]

            proc.at[idx, "zone_formed_structure"] = row["structure"]
            proc.at[idx, "zone_formed_type"]      = row["type"]
            proc.at[idx, "zone_formed_strength"]  = float(row["_strength_ml"])

            if count > 1:
                self.logger.debug(
                    f"Date {date}: {count} zones formed; "
                    f"kept {row['structure']} (strategy='{self.conflict_strategy}')"
                )

        formed_count = proc["zone_formed_count"].gt(0).sum()
        self.logger.info(
            f"Formation stamps: {formed_count} candles have ≥1 zone formed "
            f"({proc['zone_formed_count'].gt(1).sum()} had conflicts)"
        )
        return proc

    # ── Step 2: Active zone context ────────────────────────────────────────

    def _stamp_active_context(
        self,
        proc: pd.DataFrame,
        zones: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        For each candle date, find all zones that were ACTIVE at that
        point in time (formed before or on the date, not yet invalidated)
        and record the nearest demand and supply zones.

        Causal guarantee: we only include zones where formation_date <= date
        and invalidation_date is NaT or > date.

        Nearest demand = active demand zone whose TOP is closest to Close
                         (from below — the zone we might bounce from)
        Nearest supply = active supply zone whose BOTTOM is closest to Close
                         (from above — the zone we might stall under)
        """
        # Initialise active-context columns
        proc["active_demand_count"]      = 0
        proc["active_supply_count"]      = 0
        proc["nearest_demand_top"]       = np.nan
        proc["nearest_demand_bottom"]    = np.nan
        proc["nearest_demand_dist_pct"]  = np.nan
        proc["nearest_demand_strength"]  = np.nan
        proc["nearest_demand_structure"] = pd.array([None] * len(proc), dtype=object)
        proc["nearest_supply_bottom"]    = np.nan
        proc["nearest_supply_top"]       = np.nan
        proc["nearest_supply_dist_pct"]  = np.nan
        proc["nearest_supply_strength"]  = np.nan
        proc["nearest_supply_structure"] = pd.array([None] * len(proc), dtype=object)

        for i, row in proc.iterrows():
            date  = row["Date"]
            close = row["Close"]

            # Active zones: formed on or before today, not yet invalidated
            active = zones[
                (zones["formation_date"] <= date) &
                (
                    zones["invalidation_date"].isna() |
                    (zones["invalidation_date"] > date)
                )
            ]

            demand = active[active["type"] == "demand"]
            supply = active[active["type"] == "supply"]

            proc.at[i, "active_demand_count"] = len(demand)
            proc.at[i, "active_supply_count"] = len(supply)

            # Nearest demand zone: top closest to (but below) close
            if len(demand):
                # Distance from close to zone top (positive = zone below close)
                demand = demand.copy()
                demand["_dist"] = close - demand["top"]
                # Prefer zones whose top is below close (demand below price)
                below = demand[demand["_dist"] >= 0]
                if len(below):
                    nearest_d = below.loc[below["_dist"].idxmin()]
                else:
                    # Price is already inside or below demand — use the closest
                    nearest_d = demand.loc[demand["_dist"].abs().idxmin()]

                proc.at[i, "nearest_demand_top"]       = float(nearest_d["top"])
                proc.at[i, "nearest_demand_bottom"]    = float(nearest_d["bottom"])
                proc.at[i, "nearest_demand_strength"]  = float(nearest_d["_strength_ml"])
                proc.at[i, "nearest_demand_structure"] = nearest_d["structure"]
                if close > 0:
                    proc.at[i, "nearest_demand_dist_pct"] = (
                        (close - nearest_d["top"]) / close * 100
                    )

            # Nearest supply zone: bottom closest to (but above) close
            if len(supply):
                supply = supply.copy()
                supply["_dist"] = supply["bottom"] - close
                above = supply[supply["_dist"] >= 0]
                if len(above):
                    nearest_s = above.loc[above["_dist"].idxmin()]
                else:
                    nearest_s = supply.loc[supply["_dist"].abs().idxmin()]

                proc.at[i, "nearest_supply_bottom"]    = float(nearest_s["bottom"])
                proc.at[i, "nearest_supply_top"]       = float(nearest_s["top"])
                proc.at[i, "nearest_supply_strength"]  = float(nearest_s["_strength_ml"])
                proc.at[i, "nearest_supply_structure"] = nearest_s["structure"]
                if close > 0:
                    proc.at[i, "nearest_supply_dist_pct"] = (
                        (nearest_s["bottom"] - close) / close * 100
                    )

        self.logger.info("Active zone context columns stamped.")
        return proc

    # ── Step 3: Price position inside zones ────────────────────────────────

    def _stamp_price_position(
        self,
        proc: pd.DataFrame,
        zones: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Flag whether the closing price sits inside any active demand or
        supply zone.  Also flag confluence (both types active).
        """
        proc["price_in_demand_zone"] = 0
        proc["price_in_supply_zone"] = 0
        proc["zone_confluence"]      = 0

        for i, row in proc.iterrows():
            date  = row["Date"]
            close = row["Close"]

            active = zones[
                (zones["formation_date"] <= date) &
                (
                    zones["invalidation_date"].isna() |
                    (zones["invalidation_date"] > date)
                )
            ]

            in_demand = int(
                ((active["type"] == "demand") &
                 (close >= active["bottom"]) &
                 (close <= active["top"])).any()
            )
            in_supply = int(
                ((active["type"] == "supply") &
                 (close >= active["bottom"]) &
                 (close <= active["top"])).any()
            )

            proc.at[i, "price_in_demand_zone"] = in_demand
            proc.at[i, "price_in_supply_zone"] = in_supply
            proc.at[i, "zone_confluence"] = int(
                proc.at[i, "active_demand_count"] > 0 and
                proc.at[i, "active_supply_count"] > 0
            )

        in_demand_pct = proc["price_in_demand_zone"].mean() * 100
        in_supply_pct = proc["price_in_supply_zone"].mean() * 100
        self.logger.info(
            f"Price position: {in_demand_pct:.1f}% candles inside demand zone, "
            f"{in_supply_pct:.1f}% inside supply zone"
        )
        return proc

    # ── Step 4: One-hot encoding ───────────────────────────────────────────

    def _one_hot_structure(self, proc: pd.DataFrame) -> pd.DataFrame:
        """
        Produce four binary columns from zone_formed_structure.

        is_DBR = 1 when a Drop-Base-Rally (demand/reversal) zone formed today
        is_RBD = 1 when a Rally-Base-Drop (supply/reversal) zone formed today
        is_RBR = 1 when a Rally-Base-Rally (demand/continuation) zone formed today
        is_DBD = 1 when a Drop-Base-Drop (supply/continuation) zone formed today

        These are the direct ML feature inputs.  The label (y) — whether
        price reverses or breaks from this zone — comes from
        outcome_labeler.py (next step).
        """
        for label in STRUCTURE_LABELS:
            col = f"is_{label}"
            proc[col] = (proc["zone_formed_structure"] == label).astype(int)

        self.logger.info(
            "One-hot structure columns created: "
            + ", ".join(f"is_{s}" for s in STRUCTURE_LABELS)
        )
        return proc

    # ── Helpers ────────────────────────────────────────────────────────────

    def _new_columns(self) -> list[str]:
        return [
            "zone_formed_structure", "zone_formed_type",
            "zone_formed_strength", "zone_formed_count",
            "is_DBR", "is_RBD", "is_RBR", "is_DBD",
            "active_demand_count", "active_supply_count",
            "nearest_demand_top", "nearest_demand_bottom",
            "nearest_demand_dist_pct", "nearest_demand_strength",
            "nearest_demand_structure",
            "nearest_supply_bottom", "nearest_supply_top",
            "nearest_supply_dist_pct", "nearest_supply_strength",
            "nearest_supply_structure",
            "price_in_demand_zone", "price_in_supply_zone",
            "zone_confluence",
        ]


# ── Zone summary (for inspection / debugging) ───────────────────────────────

def build_zone_summary(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a concise summary table of zone-formation events from the
    labeled DataFrame.  Useful for visual inspection and sanity-checking
    before feeding data into a model.

    Returns
    -------
    pd.DataFrame  — one row per candle where a zone was formed, with
                    the candle's Close, the zone structure, type,
                    strength, and active zone counts.
    """
    formed = labeled_df[labeled_df["zone_formed_count"] > 0].copy()
    cols   = [
        "Date", "Close", "zone_formed_structure", "zone_formed_type",
        "zone_formed_strength", "zone_formed_count",
        "active_demand_count", "active_supply_count",
        "nearest_demand_dist_pct", "nearest_supply_dist_pct",
    ]
    available = [c for c in cols if c in formed.columns]
    return formed[available].reset_index(drop=True)


# ── I/O helpers ────────────────────────────────────────────────────────────

def load_processed(symbol: str, cfg: dict) -> pd.DataFrame:
    path = _processed_path(symbol, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\n"
            "Run src/data/preprocessor.py first."
        )
    return pd.read_csv(path)


def load_zones(symbol: str, cfg: dict) -> pd.DataFrame:
    path = _zones_path(symbol, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Zones data not found: {path}\n"
            "Run src/zones/zone_detector.py first."
        )
    return pd.read_csv(path)


def save_labeled(
    df: pd.DataFrame,
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
) -> Path:
    labeled_dir = _labeled_dir(cfg)
    path        = labeled_dir / (_symbol_to_stem(symbol) + "_labeled.csv")
    df.to_csv(path, index=False)
    logger.info(f"Labeled data saved → {path.relative_to(PROJECT_ROOT)}")
    return path


def save_zone_summary(
    summary: pd.DataFrame,
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
) -> Path:
    labeled_dir = _labeled_dir(cfg)
    path        = labeled_dir / (_symbol_to_stem(symbol) + "_zone_summary.csv")
    summary.to_csv(path, index=False)
    logger.info(f"Zone summary saved → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge processed OHLCV data with zone detections and "
                    "stamp zone-context columns for ML."
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Override the symbol in config.yaml (e.g. RELIANCE.NS)"
    )
    parser.add_argument(
        "--conflict", type=str, default="strongest",
        choices=["strongest", "first"],
        help="How to handle multiple zones forming on the same date "
             "(default: strongest)"
    )
    args   = parser.parse_args()
    cfg    = load_config()
    logger = setup_logging(cfg)
    symbol = args.symbol or cfg["data"]["symbol"]

    logger.info(f"=== Zone Labeler — {symbol} ===")

    processed_df = load_processed(symbol, cfg)
    zones_df     = load_zones(symbol, cfg)

    logger.info(
        f"Loaded {len(processed_df)} processed candles, "
        f"{len(zones_df)} zones"
    )

    labeler    = ZoneLabeler(cfg, conflict_strategy=args.conflict, logger=logger)
    labeled_df = labeler.run(processed_df, zones_df)

    save_labeled(labeled_df, symbol, cfg, logger)

    summary = build_zone_summary(labeled_df)
    save_zone_summary(summary, symbol, cfg, logger)

    # Quick diagnostic print
    print("\n=== Zone Formation Summary ===")
    print(f"Total candles      : {len(labeled_df)}")
    print(f"Zones formed       : {labeled_df['zone_formed_count'].gt(0).sum()}")
    for s in STRUCTURE_LABELS:
        count = labeled_df[f"is_{s}"].sum()
        print(f"  {s} formations   : {count}")
    print(f"Candles in demand  : {labeled_df['price_in_demand_zone'].sum()}")
    print(f"Candles in supply  : {labeled_df['price_in_supply_zone'].sum()}")
    print(f"Confluence candles : {labeled_df['zone_confluence'].sum()}")
    print()
    print("Zone summary (first 10 formation events):")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
