"""
zone_detection_model.py
=======================
XGBoost classifier that learns to detect supply/demand zones from
raw candlestick windows — replacing the hand-coded rule-based detector.

Research framing
----------------
Traditional zone detectors use rigid, hand-crafted rules:
  - fixed ATR multipliers for base candle thresholds
  - fixed lookback windows for leg-in measurement
  - hard-coded departure displacement gates

These rules cannot adapt to changing volatility regimes, market
microstructure, or cross-asset differences. A learned model can
generalise across these conditions by discovering the underlying
price-action pattern from data.

Approach: supervised learning with rule-based labels
------------------------------------------------------
1. Run rule-based zone_detector once → generates "ground truth" labels
2. Train XGBoost on raw candle-window features to reproduce those labels
3. The trained model is then used as the zone detector going forward

This is valid research because:
  (a) The model learns to identify patterns without knowing the rules
  (b) It generalises to regimes where rule parameters would need tuning
  (c) Performance is measured independently via walk-forward CV

Model: XGBoost Binary Classifier
---------------------------------
Why XGBoost over Random Forest for this task:
  - Gradient boosting iteratively corrects errors → better recall on
    the minority class (zones) than RF at the same depth
  - `scale_pos_weight` natively handles the 27:1 imbalance by up-weighting
    positive examples in the gradient updates
  - Faster training with `tree_method='hist'`
  - L1/L2 regularisation (`reg_alpha`, `reg_lambda`) prevents memorising
    rare zone patterns

Imbalance handling
------------------
scale_pos_weight = n_negatives / n_positives
This is XGBoost's built-in equivalent of class_weight='balanced' in sklearn.
It multiplies the gradient weight of positive examples so that 100 zone
candles carry the same total influence as 2700 no-zone candles.

Walk-forward cross-validation
------------------------------
TimeSeriesSplit(n_splits=4) on windows sorted by date.
Train on past → test on future. Never random shuffle.
We report per-class Precision, Recall, F1 — never accuracy.

Output
------
data/models/<SYMBOL>_zone_xgb.pkl        trained model
data/models/<SYMBOL>_zone_xgb_report.txt CV metrics + feature importance
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    # Fallback to Random Forest if xgboost is not installed
    from sklearn.ensemble import RandomForestClassifier

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_COL     = "label"
DATE_COL      = "date"
ZONE_TYPE_COL = "zone_type"
SKIP_COLS     = {LABEL_COL, DATE_COL, ZONE_TYPE_COL}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _symbol_to_stem(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "IDX_")


def _model_dir(cfg: dict) -> Path:
    d = PROJECT_ROOT / cfg.get("models", {}).get("model_dir", "data/models")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Data loading ──────────────────────────────────────────────────────────────

def load_window_dataset(symbol: str, cfg: dict) -> pd.DataFrame:
    labeled_dir = PROJECT_ROOT / cfg.get("features", {}).get("labeled_dir", "data/labeled")
    path = labeled_dir / (_symbol_to_stem(symbol) + "_zone_windows_v2.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Window dataset not found: {path}\n"
            "Run zone_pipeline.py --skip-detection first."
        )
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL).reset_index(drop=True)


def get_X_y(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in SKIP_COLS]
    X = df[feat_cols].values.astype(float)
    y = df[LABEL_COL].values.astype(int)
    return X, y, feat_cols


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(scale_pos_weight: float = 1.0):
    """
    Build an XGBoost binary classifier configured for imbalanced zone detection.

    Parameters
    ----------
    scale_pos_weight : n_negatives / n_positives
        Up-weights zone examples so they contribute as much as no-zone examples
        to the loss gradient. Equivalent to class_weight='balanced' in sklearn.

    Hyperparameter choices
    ----------------------
    n_estimators=500       More trees help recall on rare positives
    max_depth=4            Shallow trees → less overfitting on small positive set
    min_child_weight=10    Leaf must cover 10 samples → prevents rare-positive
                           memorisation
    learning_rate=0.05     Slow learning with many trees → better generalisation
    subsample=0.8          Row subsampling → diversity, reduces variance
    colsample_bytree=0.6   Feature subsampling per tree → reduces correlation
    reg_alpha=0.1          L1 regularisation → sparse feature weights
    reg_lambda=1.0         L2 regularisation → smooth weights
    eval_metric='aucpr'    Area under Precision-Recall curve — correct metric
                           for imbalanced binary classification
    """
    if XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            min_child_weight=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        # Fallback: balanced Random Forest
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )


# ── Walk-forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 4,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Temporal walk-forward CV.

    Reports per-class Precision, Recall, F1 for each fold.
    NEVER reports accuracy — meaningless on 27:1 imbalanced data.

    Returns dict with fold_results, avg_per_class, feature_importances.
    """
    if logger is None:
        logger = logging.getLogger("zone_detection_model")

    X, y, feat_cols = get_X_y(df)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw   = n_neg / max(n_pos, 1)

    logger.info(f"Dataset: {len(df)} rows | positives={n_pos} | negatives={n_neg} "
                f"| scale_pos_weight={spw:.1f}")
    logger.info(f"Model: {'XGBoost' if XGB_AVAILABLE else 'RandomForest (fallback)'}")
    logger.info(f"Walk-forward CV: {n_splits} folds")

    fold_results  = []
    importances   = np.zeros(len(feat_cols))
    n_valid_folds = 0

    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        n_pos_tr = (y_tr == 1).sum()
        n_pos_te = (y_te == 1).sum()

        if n_pos_tr < 5:
            logger.warning(f"Fold {fold_idx}: only {n_pos_tr} positives in train — skipping.")
            continue

        # Recompute scale_pos_weight per fold
        spw_fold = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model = build_model(scale_pos_weight=spw_fold)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # Per-class metrics only — no accuracy ever printed
        report = classification_report(
            y_te, y_pred,
            labels=[0, 1],
            target_names=["no_zone", "zone"],
            zero_division=0,
            output_dict=True,
        )

        logger.info(f"\nFold {fold_idx}  |  train={len(tr_idx)} (zones={n_pos_tr})  |  "
                    f"test={len(te_idx)} (zones={n_pos_te})")
        logger.info(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
        logger.info("  " + "─" * 48)
        for cls in ["no_zone", "zone"]:
            m = report[cls]
            logger.info(f"  {cls:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} "
                        f"{m['f1-score']:>8.3f} {int(m['support']):>9}")
        logger.info(f"  macro F1: {report['macro avg']['f1-score']:.3f}")

        fold_results.append({
            "fold":        fold_idx,
            "n_train":     len(tr_idx),
            "n_test":      len(te_idx),
            "zones_train": int(n_pos_tr),
            "zones_test":  int(n_pos_te),
            "no_zone":     {k: round(report["no_zone"][k], 4)
                           for k in ("precision","recall","f1-score")},
            "zone":        {k: round(report["zone"][k], 4)
                           for k in ("precision","recall","f1-score")},
            "macro_f1":    round(report["macro avg"]["f1-score"], 4),
        })

        if hasattr(model, "feature_importances_"):
            importances   += model.feature_importances_
            n_valid_folds += 1

    if not fold_results:
        logger.error("No valid CV folds produced.")
        return {}

    if n_valid_folds > 0:
        importances /= n_valid_folds

    # Average across folds
    avg = {}
    for cls in ["no_zone", "zone"]:
        for metric in ["precision", "recall", "f1-score"]:
            key = f"{cls}_{metric.replace('-','_')}"
            avg[key] = round(np.mean([f[cls][metric] for f in fold_results]), 4)

    avg_macro = round(np.mean([f["macro_f1"] for f in fold_results]), 4)

    logger.info(f"\n{'═'*50}")
    logger.info(f"CV AVERAGE ACROSS {len(fold_results)} FOLDS")
    logger.info(f"  zone   Precision={avg['zone_precision']:.3f}  "
                f"Recall={avg['zone_recall']:.3f}  F1={avg['zone_f1_score']:.3f}")
    logger.info(f"  macro F1 = {avg_macro:.3f}")

    fi_dict = dict(sorted(
        zip(feat_cols, importances),
        key=lambda x: x[1], reverse=True,
    ))

    return {
        "fold_results":        fold_results,
        "avg":                 avg,
        "avg_macro_f1":        avg_macro,
        "feature_importances": fi_dict,
        "feature_names":       feat_cols,
    }


# ── Final model ───────────────────────────────────────────────────────────────

def train_final_model(df: pd.DataFrame, logger: logging.Logger | None = None):
    """Train on full dataset for deployment."""
    if logger is None:
        logger = logging.getLogger("zone_detection_model")

    X, y, feat_cols = get_X_y(df)
    spw   = (y == 0).sum() / max((y == 1).sum(), 1)
    model = build_model(scale_pos_weight=spw)
    model.fit(X, y)

    y_pred = model.predict(X)
    logger.info("\nFinal model — train-set report:")
    logger.info(classification_report(
        y, y_pred,
        target_names=["no_zone", "zone"],
        zero_division=0,
    ))

    fi = dict(sorted(
        zip(feat_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    ))
    logger.info("Top 10 features:")
    for name, imp in list(fi.items())[:10]:
        logger.info(f"  {name:<35} {imp:.4f}")

    return model, fi


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, symbol: str, cfg: dict, logger: logging.Logger) -> Path:
    path = _model_dir(cfg) / (_symbol_to_stem(symbol) + "_zone_xgb.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path.relative_to(PROJECT_ROOT)}")
    return path


def load_model(symbol: str, cfg: dict):
    path = _model_dir(cfg) / (_symbol_to_stem(symbol) + "_zone_xgb.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Zone detection model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_report(cv: dict, symbol: str, cfg: dict, logger: logging.Logger) -> Path:
    path = _model_dir(cfg) / (_symbol_to_stem(symbol) + "_zone_xgb_report.txt")
    lines = [
        "=" * 60,
        f"Zone Detection Model Report — {symbol}",
        f"Model: {'XGBoost' if XGB_AVAILABLE else 'RandomForest (fallback)'}",
        "=" * 60,
        "",
        "APPROACH:",
        "  Labels  : rule-based zone_detector output (ground truth)",
        "  Features: sliding 20-candle ATR-normalised windows",
        "  Target  : binary — zone (1) vs no_zone (0)",
        "  Metric  : Precision / Recall / F1 per class (NOT accuracy)",
        "",
        "── Walk-Forward CV Results ──",
        f"  avg macro F1 : {cv.get('avg_macro_f1', 'N/A')}",
        f"  zone Precision : {cv.get('avg', {}).get('zone_precision', 'N/A')}",
        f"  zone Recall    : {cv.get('avg', {}).get('zone_recall', 'N/A')}",
        f"  zone F1        : {cv.get('avg', {}).get('zone_f1_score', 'N/A')}",
        "",
        "── Per-Fold ──",
    ]
    for fold in cv.get("fold_results", []):
        lines.append(
            f"  Fold {fold['fold']}: zones_train={fold['zones_train']} "
            f"zones_test={fold['zones_test']}  macro_F1={fold['macro_f1']}"
        )
    lines += ["", "── Feature Importances (top 20) ──"]
    for name, imp in list(cv.get("feature_importances", {}).items())[:20]:
        bar = "█" * int(imp * 80)
        lines.append(f"  {name:<35} {imp:.4f}  {bar}")
    path.write_text("\n".join(lines))
    logger.info(f"Report saved → {path.relative_to(PROJECT_ROOT)}")
    return path


# ── Pipeline entry ────────────────────────────────────────────────────────────

def run_model_training(
    symbol: str,
    cfg: dict,
    logger: logging.Logger,
    n_splits: int = 4,
):
    """Called from zone_pipeline.py."""
    df = load_window_dataset(symbol, cfg)
    cv = walk_forward_cv(df, n_splits=n_splits, logger=logger)
    model, _ = train_final_model(df, logger=logger)
    save_model(model, symbol, cfg, logger)
    save_report(cv, symbol, cfg, logger)
    return model, cv


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train zone detection XGBoost model.")
    parser.add_argument("--symbol",   type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=4)
    args   = parser.parse_args()
    cfg    = load_config()
    import logging as _log
    logging.basicConfig(level=_log.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger = _log.getLogger("zone_detection_model")
    symbol = args.symbol or cfg["data"]["symbol"]
    run_model_training(symbol, cfg, logger, n_splits=args.n_splits)
