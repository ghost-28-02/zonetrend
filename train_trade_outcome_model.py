"""
train_trade_outcome_model.py
============================
Walk-forward evaluation of the trade-outcome filter.

Pipeline
--------
1. Build pooled per-trade dataset (point-in-time features + win/loss label).
2. TimeSeriesSplit walk-forward:
     - train XGBoost P(win) on past trades
     - tune the probability threshold on the TRAIN fold (maximise train total-R
       subject to keeping >= 40% of train trades, to avoid degenerate thresholds)
     - apply that fixed threshold to the strictly-future TEST fold
3. Report, out-of-sample and pooled across test folds:
     - ML metrics: ROC-AUC, precision/recall/F1 for the "win" class
     - TRADING metrics: filtered vs unfiltered win-rate, profit factor, avg-R
4. Logistic-regression baseline for comparison.

The verdict is whether the filter improves OOS trading metrics. It is not
assumed to.
"""
import contextlib
import io
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.models.trade_outcome_model import (  # noqa: E402
    build_dataset, feature_columns, load_config, trade_stats,
)

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("trade_outcome")


def tune_threshold(p_train, r_train, min_keep_frac=0.40):
    """Pick threshold maximising train total-R while keeping >= min_keep_frac trades."""
    best_t, best_tot = 0.0, -1e9
    for t in np.quantile(p_train, np.linspace(0.0, 0.9, 31)):
        mask = p_train >= t
        if mask.mean() < min_keep_frac:
            continue
        tot = r_train[mask].sum()
        if tot > best_tot:
            best_tot, best_t = tot, t
    return best_t


def build_model(spw):
    if HAVE_XGB:
        return XGBClassifier(
            n_estimators=200, max_depth=3, min_child_weight=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=1.5, scale_pos_weight=spw,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0,
        )
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1)


def evaluate(df, feat_cols, model_kind="xgb", n_splits=5):
    X = df[feat_cols].values.astype(float)
    y = df["label"].values.astype(int)
    r = df["r_multiple"].values.astype(float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_p, oos_y, oos_r = [], [], []
    filt_r, unfilt_r = [], []

    for tr, te in tscv.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        rtr, rte = r[tr], r[te]
        if ytr.sum() < 5 or (ytr == 0).sum() < 5:
            continue
        spw = (ytr == 0).sum() / max(ytr.sum(), 1)

        if model_kind == "logit":
            med = np.nanmedian(Xtr, axis=0)
            Xtr_i = np.where(np.isnan(Xtr), med, Xtr)
            Xte_i = np.where(np.isnan(Xte), med, Xte)
            sc = StandardScaler().fit(Xtr_i)
            m = LogisticRegression(max_iter=2000, class_weight="balanced")
            m.fit(sc.transform(Xtr_i), ytr)
            p_tr = m.predict_proba(sc.transform(Xtr_i))[:, 1]
            p_te = m.predict_proba(sc.transform(Xte_i))[:, 1]
        else:
            m = build_model(spw)
            m.fit(Xtr, ytr)
            p_tr = m.predict_proba(Xtr)[:, 1]
            p_te = m.predict_proba(Xte)[:, 1]

        thr = tune_threshold(p_tr, rtr)
        keep = p_te >= thr

        oos_p.extend(p_te); oos_y.extend(yte); oos_r.extend(rte)
        unfilt_r.extend(rte)
        filt_r.extend(rte[keep])

    oos_p, oos_y = np.array(oos_p), np.array(oos_y)
    pred = (oos_p >= 0.5).astype(int)
    ml = {
        "roc_auc": roc_auc_score(oos_y, oos_p) if len(set(oos_y)) > 1 else float("nan"),
        "precision_win": precision_score(oos_y, pred, zero_division=0),
        "recall_win": recall_score(oos_y, pred, zero_division=0),
        "f1_win": f1_score(oos_y, pred, zero_division=0),
    }
    return ml, trade_stats(unfilt_r), trade_stats(filt_r)


def main():
    cfg = load_config()
    print("Building per-trade dataset (running corrected backtests)...")
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_dataset(cfg, logger)
    feat_cols = feature_columns(df)
    print(f"\nDataset: {len(df)} trades | {df['label'].mean()*100:.1f}% winners "
          f"| {len(feat_cols)} point-in-time features")
    print(f"Date range: {df['entry_date'].min().date()} -> {df['entry_date'].max().date()}")

    for kind, name in [("xgb" if HAVE_XGB else "rf", "XGBoost" if HAVE_XGB else "RandomForest"),
                       ("logit", "LogisticRegression (baseline)")]:
        ml, unf, fil = evaluate(df, feat_cols, model_kind=kind)
        print(f"\n===== {name} — walk-forward OOS =====")
        print(f"  ML   : ROC-AUC={ml['roc_auc']:.3f}  "
              f"P(win)={ml['precision_win']:.3f}  R(win)={ml['recall_win']:.3f}  "
              f"F1(win)={ml['f1_win']:.3f}")
        print(f"  UNFILTERED (all trades) : n={unf['n']:>3}  WR={unf['wr']*100:5.1f}%  "
              f"PF={unf['pf']:.3f}  avgR={unf['avg_r']:+.3f}  totR={unf['total_r']:+.1f}")
        print(f"  FILTERED (P>=thr)       : n={fil['n']:>3}  WR={fil['wr']*100:5.1f}%  "
              f"PF={fil['pf']:.3f}  avgR={fil['avg_r']:+.3f}  totR={fil['total_r']:+.1f}")
        d_wr = (fil['wr'] - unf['wr']) * 100
        d_ar = fil['avg_r'] - unf['avg_r']
        print(f"  DELTA  : WR {d_wr:+.1f} pp | avgR {d_ar:+.3f} | "
              f"kept {fil['n']}/{unf['n']} = {fil['n']/max(unf['n'],1)*100:.0f}% of trades")

    # ── Leakage control: shuffle labels. A clean pipeline must collapse to
    #    ROC-AUC ~= 0.5 and show no OOS trading edge. ───────────────────────────
    rng = np.random.RandomState(0)
    df_shuf = df.copy()
    df_shuf["label"] = rng.permutation(df_shuf["label"].values)
    # keep r_multiple consistent with the (now scrambled) label sign so the
    # trading-impact test is meaningful under the null
    df_shuf["r_multiple"] = np.where(df_shuf["label"] == 1,
                                     np.abs(df_shuf["r_multiple"]),
                                     -np.abs(df_shuf["r_multiple"]))
    ml_s, unf_s, fil_s = evaluate(df_shuf, feat_cols, model_kind="xgb" if HAVE_XGB else "rf")
    print(f"\n===== CONTROL: shuffled labels (should be ~random) =====")
    print(f"  ML   : ROC-AUC={ml_s['roc_auc']:.3f}  F1(win)={ml_s['f1_win']:.3f}")
    print(f"  UNFILTERED : WR={unf_s['wr']*100:5.1f}%  avgR={unf_s['avg_r']:+.3f}")
    print(f"  FILTERED   : WR={fil_s['wr']*100:5.1f}%  avgR={fil_s['avg_r']:+.3f}  "
          f"(delta {(fil_s['wr']-unf_s['wr'])*100:+.1f} pp)")

    # ── Fit final models on all data + persist artifacts and report ───────────
    import pickle
    X = df[feat_cols].values.astype(float)
    y = df["label"].values.astype(int)
    spw = (y == 0).sum() / max(y.sum(), 1)
    mdl_dir = ROOT / "data" / "models"
    mdl_dir.mkdir(parents=True, exist_ok=True)

    xgb_final = build_model(spw).fit(X, y)
    with open(mdl_dir / "trade_outcome_xgb.pkl", "wb") as f:
        pickle.dump({"model": xgb_final, "features": feat_cols}, f)

    med = np.nanmedian(X, axis=0)
    sc = StandardScaler().fit(np.where(np.isnan(X), med, X))
    logit_final = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        sc.transform(np.where(np.isnan(X), med, X)), y)
    with open(mdl_dir / "trade_outcome_logit.pkl", "wb") as f:
        pickle.dump({"model": logit_final, "scaler": sc, "median": med,
                     "features": feat_cols}, f)

    imp = sorted(zip(feat_cols, xgb_final.feature_importances_),
                 key=lambda t: -t[1])[:12]
    report = ROOT / "trade_outcome_report.txt"
    with open(report, "w") as f:
        f.write("Trade-Outcome Filter — walk-forward OOS evaluation\n")
        f.write("=" * 55 + "\n")
        f.write(f"Dataset: {len(df)} rule-based-zone trades, {df['label'].mean()*100:.1f}% winners\n")
        f.write(f"Span   : {df['entry_date'].min().date()} -> {df['entry_date'].max().date()}\n")
        f.write(f"Features: {len(feat_cols)} point-in-time only (post-hoc columns dropped)\n\n")
        for kind, name in [("xgb" if HAVE_XGB else "rf", "XGBoost"),
                           ("logit", "LogisticRegression")]:
            ml, unf, fil = evaluate(df, feat_cols, model_kind=kind)
            f.write(f"[{name}]\n")
            f.write(f"  ROC-AUC={ml['roc_auc']:.3f} F1(win)={ml['f1_win']:.3f}\n")
            f.write(f"  Unfiltered: n={unf['n']} WR={unf['wr']*100:.1f}% PF={unf['pf']:.3f} avgR={unf['avg_r']:+.3f}\n")
            f.write(f"  Filtered  : n={fil['n']} WR={fil['wr']*100:.1f}% PF={fil['pf']:.3f} avgR={fil['avg_r']:+.3f}\n")
            f.write(f"  Delta WR  : {(fil['wr']-unf['wr'])*100:+.1f} pp, kept {fil['n']/max(unf['n'],1)*100:.0f}%\n\n")
        f.write(f"Control (shuffled labels): ROC-AUC={ml_s['roc_auc']:.3f} "
                f"filter delta={(fil_s['wr']-unf_s['wr'])*100:+.1f}pp -> no leakage\n\n")
        f.write("Top XGBoost point-in-time features by importance:\n")
        for name_, v in imp:
            f.write(f"  {name_:<26} {v:.4f}\n")

    df.to_csv(ROOT / "trade_outcome_dataset.csv", index=False)
    print("\nSaved: trade_outcome_dataset.csv, trade_outcome_report.txt, "
          "data/models/trade_outcome_xgb.pkl, data/models/trade_outcome_logit.pkl")


if __name__ == "__main__":
    main()
