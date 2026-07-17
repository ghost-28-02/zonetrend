"""
zone_rf_model.py  —  Option B
==============================
Random Forest classifier trained on the FULL labeled candle dataset.

Approach (Option B)
-------------------
Instead of a small zone-window dataset (400 rows), we use every candle in
the processed history (2828 rows) as a training example.

  Label   : zone_formed_structure — what zone type (if any) formed TODAY
            DBR | RBD | RBR | DBD | no_zone
  Features: candle geometry + volume + trend position + zone context
            (all computed at close of that candle — no look-ahead)

Why this is valid
-----------------
zone_formed_structure is stamped on the DEPARTURE candle close by
zone_detector.py. The model's features are also computed at that candle's
close. No future information is used.

Class imbalance
---------------
2728 'no_zone' vs ~25 of each zone type  →  27:1 imbalance.
We handle this with class_weight='balanced', which auto-scales each
class's contribution to the loss inversely proportional to its frequency.
We NEVER report raw accuracy — it is meaningless when 96.5% of rows are
'no_zone'. We evaluate with per-class Precision / Recall / F1 only.

NaN imputation
--------------
nearest_demand_dist_pct  → 999.0  (no demand zone exists yet below price)
nearest_supply_dist_pct  → 999.0  (no supply zone exists yet above price)
All other features        → median of that column (warm-up period NaNs)

Feature set (19 features, all pre-departure / causal)
------------------------------------------------------
  Candle geometry (normalised, no price-level dependency):
    BodyRatio, UpperWickRatio, LowerWickRatio, BodyToATR, RangeToATR, IsBullish

  Returns & volatility:
    LogReturn, RollingVolatility

  Volume:
    VolumeRatio

  Trend position ((Close − EMA) / ATR — tells model where we are in trend):
    close_vs_ema20, close_vs_ema50, close_vs_ema200

  Zone context (from zone_labeler.py — causal, no look-ahead):
    active_demand_count, active_supply_count,
    nearest_demand_dist_pct, nearest_supply_dist_pct,
    price_in_demand_zone, price_in_supply_zone, zone_confluence

Walk-forward cross-validation
------------------------------
TimeSeriesSplit(n_splits=4) on 2828 rows sorted by Date.
Each fold trains on all past rows and tests on future rows.
We report per-class P/R/F1 for every fold — never macro accuracy.

Output
------
  data/models/<SYMBOL>_zone_rf.pkl     trained model (pickle)
  data/models/<SYMBOL>_rf_report.txt   CV metrics + feature importance
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

# ── Feature / class constants ───────────────────────────────────────────────
CLASSES = ["no_zone", "DBR", "RBD", "RBR", "DBD"]

ML_FEATURES = [
    # Candle geometry
    "BodyRatio",
    "UpperWickRatio",
    "LowerWickRatio",
    "BodyToATR",
    "RangeToATR",
    "IsBullish",
    # Returns & volatility
    "LogReturn",
    "RollingVolatility",
    # Volume
    "VolumeRatio",
    # Trend position
    "close_vs_ema20",
    "close_vs_ema50",
    "close_vs_ema200",
    # Zone context (causal — from zone_labeler)
    "active_demand_count",
    "active_supply_count",
    "nearest_demand_dist_pct",
    "nearest_supply_dist_pct",
    "price_in_demand_zone",
    "price_in_supply_zone",
    "zone_confluence",
]


# ── Config & logging ────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> logging.Logger:
    log_dir = PROJECT_ROOT / cfg["logging"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg["logging"]["log_level"].upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if cfg["logging"].get("log_to_file", True):
        handlers.append(logging.FileHandler(log_dir / "zone_rf_model.log"))
    if cfg["logging"].get("log_to_console", True):
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level, handlers=handlers,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("zone_rf_model")


def _symbol_to_stem(symbol: str) -> str:
    return "IDX_" + symbol[1:] if symbol.startswith("^") else symbol.replace(".", "_")


def _model_dir(cfg: dict) -> Path:
    d = PROJECT_ROOT / cfg.get("models", {}).get("model_dir", "data/models")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Data loading & engineering ──────────────────────────────────────────────

def load_labeled(symbol: str, cfg: dict) -> pd.DataFrame:
    labeled_dir = PROJECT_ROOT / cfg.get("features", {}).get("labeled_dir", "data/labeled")
    path = labeled_dir / (_symbol_to_stem(symbol) + "_labeled.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled data not found: {path}\n"
            "Run zone_pipeline.py (steps 1-2) first."
        )
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def engineer_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute derived features and impute NaNs.
    Returns a copy with all ML_FEATURES columns ready.
    """
    df = df.copy()

    # Trend position: (Close − EMA) / ATR  — dimensionless, scale-free
    for period, ema_col in [(20, "EMA20"), (50, "EMA50"), (200, "EMA200")]:
        feat = f"close_vs_ema{period}"
        df[feat] = (df["Close"] - df[ema_col]) / df["ATR"].replace(0, np.nan)

    # Zone distance NaN → 999 (means "no zone present on that side yet")
    df["nearest_demand_dist_pct"] = df["nearest_demand_dist_pct"].fillna(999.0)
    df["nearest_supply_dist_pct"] = df["nearest_supply_dist_pct"].fillna(999.0)

    # Remaining NaNs (warm-up period) → column median
    for col in ML_FEATURES:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' not found in labeled data — filling with 0")
            df[col] = 0.0
        n_nan = df[col].isna().sum()
        if n_nan:
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            logger.debug(f"  {col}: {n_nan} NaN → filled with median {fill:.4f}")

    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """Fill NaN structure with 'no_zone' to produce the 5-class label."""
    return df["zone_formed_structure"].fillna("no_zone")


def get_X_y(df: pd.DataFrame):
    X = df[ML_FEATURES].values.astype(float)
    y = make_target(df).values
    return X, y


# ── Model ───────────────────────────────────────────────────────────────────

def build_model(n_estimators: int = 300) -> RandomForestClassifier:
    """
    Random Forest tuned for imbalanced 5-class data.

    class_weight='balanced': weight of each class = n_samples / (n_classes × n_class_i)
    This makes the model treat each zone type as equally important as no_zone,
    even though no_zone is 27× more common.

    min_samples_leaf=5: prevents the model from memorising the rare zone candles.
    A leaf needs at least 5 samples — keeps generalisation.

    max_features='sqrt': each split considers sqrt(19) ≈ 4 features.
    Creates tree diversity and prevents dominant features from taking over.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ── Walk-forward CV ──────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 4,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Temporal walk-forward CV on the full labeled dataset.

    Reports per-class Precision, Recall, F1 for each fold.
    NEVER reports raw accuracy — it is misleading on imbalanced data.

    Returns
    -------
    dict: fold_results, avg metrics per class, feature importances
    """
    if logger is None:
        logger = logging.getLogger("zone_rf_model")

    X, y = get_X_y(df)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results  = []
    importances   = np.zeros(len(ML_FEATURES))
    n_valid_folds = 0

    class_dist = pd.Series(y).value_counts().to_dict()
    logger.info(f"Dataset: {len(df)} rows | class distribution: {class_dist}")
    logger.info(f"Walk-forward CV: {n_splits} folds")

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Count positive (zone) examples in training split
        n_zones_train = int((pd.Series(y_train) != "no_zone").sum())
        n_zones_test  = int((pd.Series(y_test)  != "no_zone").sum())

        if n_zones_train < 10:
            logger.warning(
                f"Fold {fold_idx}: only {n_zones_train} zone examples in "
                "training set — results unreliable, skipping."
            )
            continue

        model = build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Per-class metrics — the ONLY valid evaluation on imbalanced data
        present_classes = sorted(set(y_test) | set(y_pred))
        report = classification_report(
            y_test, y_pred,
            labels=present_classes,
            zero_division=0,
            output_dict=True,
        )

        # Log readable summary (no accuracy ever printed)
        logger.info(f"\nFold {fold_idx} | train={len(train_idx)} (zones={n_zones_train}) | "
                    f"test={len(test_idx)} (zones={n_zones_test})")
        logger.info(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        logger.info("─" * 52)
        for cls in CLASSES:
            if cls in report:
                m = report[cls]
                logger.info(
                    f"  {cls:<10} {m['precision']:>10.3f} {m['recall']:>8.3f} "
                    f"{m['f1-score']:>8.3f} {int(m['support']):>9}"
                )

        fold_results.append({
            "fold":          fold_idx,
            "n_train":       len(train_idx),
            "n_test":        len(test_idx),
            "zones_train":   n_zones_train,
            "zones_test":    n_zones_test,
            "per_class":     {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall":    round(report[cls]["recall"], 4),
                    "f1":        round(report[cls]["f1-score"], 4),
                    "support":   int(report[cls]["support"]),
                }
                for cls in CLASSES if cls in report
            },
            "macro_f1":       round(report["macro avg"]["f1-score"], 4),
            "weighted_f1":    round(report["weighted avg"]["f1-score"], 4),
        })

        importances   += model.feature_importances_
        n_valid_folds += 1

    if not fold_results:
        logger.error("No valid CV folds — dataset too small or no zone examples.")
        return {}

    importances /= n_valid_folds

    # Average per-class metrics across folds
    avg_per_class = {}
    for cls in CLASSES:
        metrics_list = [
            f["per_class"][cls] for f in fold_results if cls in f["per_class"]
        ]
        if metrics_list:
            avg_per_class[cls] = {
                "precision": round(np.mean([m["precision"] for m in metrics_list]), 4),
                "recall":    round(np.mean([m["recall"]    for m in metrics_list]), 4),
                "f1":        round(np.mean([m["f1"]        for m in metrics_list]), 4),
            }

    avg_macro_f1 = round(np.mean([f["macro_f1"] for f in fold_results]), 4)

    logger.info(f"\n{'═'*52}")
    logger.info(f"CV AVERAGE ACROSS {n_valid_folds} FOLDS")
    logger.info(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    logger.info("─" * 42)
    for cls in CLASSES:
        if cls in avg_per_class:
            m = avg_per_class[cls]
            logger.info(
                f"  {cls:<10} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}"
            )
    logger.info(f"  {'macro avg':<10} {'':>10} {'':>8} {avg_macro_f1:>8.3f}")

    return {
        "fold_results":        fold_results,
        "avg_per_class":       avg_per_class,
        "avg_macro_f1":        avg_macro_f1,
        "feature_importances": dict(zip(ML_FEATURES, importances.round(4))),
    }


# ── Final model ──────────────────────────────────────────────────────────────

def train_final_model(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> tuple[RandomForestClassifier, dict]:
    """Train on the full dataset for deployment in the backtester."""
    if logger is None:
        logger = logging.getLogger("zone_rf_model")

    X, y = get_X_y(df)
    model = build_model()
    model.fit(X, y)

    y_pred = model.predict(X)

    # Per-class report on train set (for reference — not evaluation)
    logger.info("\nFinal model — train-set per-class report:")
    logger.info(classification_report(y, y_pred, zero_division=0))

    fi_sorted = sorted(
        zip(ML_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("Feature importances (top 10):")
    for name, imp in fi_sorted[:10]:
        logger.info(f"  {name:<30} {imp:.4f}")

    return model, {
        "feature_importances": dict(fi_sorted),
        "train_report": classification_report(y, y_pred, zero_division=0),
    }


# ── Persistence ──────────────────────────────────────────────────────────────

def save_model(model, symbol, cfg, logger) -> Path:
    path = _model_dir(cfg) / (_symbol_to_stem(symbol) + "_zone_rf.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path.relative_to(PROJECT_ROOT)}")
    return path


def load_model(symbol, cfg) -> RandomForestClassifier:
    path = _model_dir(cfg) / (_symbol_to_stem(symbol) + "_zone_rf.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_report(cv_summary, eval_dict, symbol, cfg, logger) -> Path:
    path = _model_dir(cfg) / (_symbol_to_stem(symbol) + "_rf_report.txt")
    lines = [
        "=" * 60,
        f"Zone RF Model Report (Option B) — {symbol}",
        "=" * 60,
        "",
        "APPROACH: Full labeled candle dataset (Option B)",
        f"  Dataset : {len(ML_FEATURES)} features × candle-level rows",
        "  Target  : zone_formed_structure (5-class)",
        "  Classes : " + " | ".join(CLASSES),
        "  Metric  : Per-class Precision / Recall / F1 (NOT accuracy)",
        "",
        "── Walk-Forward CV Results ──",
        f"  avg macro F1 : {cv_summary.get('avg_macro_f1', 'N/A')}",
        "",
        "  Average per-class metrics:",
        f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}",
        "  " + "─" * 42,
    ]
    for cls in CLASSES:
        m = cv_summary.get("avg_per_class", {}).get(cls)
        if m:
            lines.append(
                f"  {cls:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}"
            )

    lines += [
        "",
        "  Per-fold macro F1:",
    ]
    for fold in cv_summary.get("fold_results", []):
        lines.append(
            f"    Fold {fold['fold']}: train={fold['n_train']} "
            f"(zones={fold['zones_train']})  "
            f"test={fold['n_test']} (zones={fold['zones_test']})  "
            f"macro_F1={fold['macro_f1']}"
        )

    lines += [
        "",
        "── Feature Importances (final model) ──",
    ]
    for feat, imp in eval_dict["feature_importances"].items():
        bar = "█" * int(imp * 60)
        lines.append(f"  {feat:<30} {imp:.4f}  {bar}")

    lines += [
        "",
        "── Train-set Classification Report (final model) ──",
        eval_dict["train_report"],
    ]

    path.write_text("\n".join(lines))
    logger.info(f"Report saved → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── Score zones ───────────────────────────────────────────────────────────────

def score_zones(zones_df, labeled_df, model, logger) -> pd.DataFrame:
    """
    For each zone, find the matching candle in labeled_df (by formation_date),
    run the model on that candle's features, and attach P(zone) as ml_zone_prob.
    """
    labeled_df = labeled_df.copy()
    labeled_df["Date"] = pd.to_datetime(labeled_df["Date"])

    # Build feature matrix for labeled_df
    labeled_eng = engineer_features(labeled_df, logger)
    X_all = labeled_eng[ML_FEATURES].values.astype(float)

    # Predict probabilities for every candle
    probs     = model.predict_proba(X_all)
    no_zone_i = list(model.classes_).index("no_zone") if "no_zone" in model.classes_ else 0
    # P(zone) = 1 - P(no_zone)
    p_zone = 1.0 - probs[:, no_zone_i]
    labeled_eng["_ml_p_zone"] = p_zone

    date_to_pzone = labeled_eng.set_index("Date")["_ml_p_zone"].to_dict()

    zones_df = zones_df.copy()
    zones_df["formation_date"] = pd.to_datetime(zones_df["formation_date"])
    zones_df["ml_zone_prob"] = zones_df["formation_date"].map(date_to_pzone)
    zones_df["ml_rank"] = zones_df["ml_zone_prob"].rank(ascending=False, method="first").astype("Int64")

    scored = zones_df["ml_zone_prob"].notna().sum()
    logger.info(
        f"ML scoring: {scored}/{len(zones_df)} zones scored  |  "
        f"P(zone) range [{zones_df['ml_zone_prob'].min():.3f}, "
        f"{zones_df['ml_zone_prob'].max():.3f}]"
    )
    return zones_df


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_model_training(symbol, cfg, logger, n_splits=4) -> RandomForestClassifier:
    """Called from zone_pipeline.py. Returns the trained final model."""
    raw_df      = load_labeled(symbol, cfg)
    df          = engineer_features(raw_df, logger)
    cv_summary  = walk_forward_cv(df, n_splits=n_splits, logger=logger)
    model, eval_dict = train_final_model(df, logger=logger)
    save_model(model, symbol, cfg, logger)
    save_report(cv_summary, eval_dict, symbol, cfg, logger)
    return model


# ── Expose for notebook / backtester ─────────────────────────────────────────

def load_windows(symbol: str, cfg: dict) -> pd.DataFrame:
    """Alias kept for pipeline compatibility — returns labeled dataset."""
    return load_labeled(symbol, cfg)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train zone RF (Option B — full candle dataset).")
    parser.add_argument("--symbol",   type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=4)
    args   = parser.parse_args()
    cfg    = load_config()
    logger = setup_logging(cfg)
    symbol = args.symbol or cfg["data"]["symbol"]
    logger.info(f"=== Zone RF Model (Option B) — {symbol} ===")
    run_model_training(symbol, cfg, logger, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
