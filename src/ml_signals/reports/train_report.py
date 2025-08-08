import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

def _to_float_series(s: pd.Series, name="value") -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out.name = name
    return out

def _to_utc_ts(s: pd.Series) -> pd.Series:
    # Robustly coerce to timezone-aware UTC datetimes
    return pd.to_datetime(s, utc=True, errors="coerce")

def make_train_report(df_proc: pd.DataFrame,
                      oof_df: pd.DataFrame,
                      report_dir: str,
                      tag: str = "1min") -> dict:
    """
    Build training report from processed dataframe + OOF probabilities.
    Saves:
      - ROC plot -> reports/roc_<tag>.png
      - Reliability plot -> reports/reliability_<tag>.png
      - JSON metrics -> reports/train_report_<tag>.json
    Returns a dict of summary metrics.
    """
    os.makedirs(report_dir, exist_ok=True)

    # --- Merge and sanitize dtypes ---
    df = df_proc.copy()
    oof = oof_df.copy()

    # Force timestamps to tz-aware UTC to avoid dtype issues
    df["timestamp"] = _to_utc_ts(df["timestamp"])
    oof["timestamp"] = _to_utc_ts(oof["timestamp"])

    # Keep only directional labeled rows and essential cols
    keep_cols = ["timestamp", "y"]
    if "w" in df.columns:
        keep_cols.append("w")
    if "ev_ret" in df.columns:
        keep_cols.append("ev_ret")

    m = df["y"].isin([-1, 1])
    d = df.loc[m, keep_cols].merge(
        oof[["timestamp", "prob_long"]], on="timestamp", how="inner"
    )

    # Coerce numerics
    d["prob_long"] = _to_float_series(d["prob_long"], "prob_long")
    if "w" in d.columns:
        d["w"] = _to_float_series(d["w"], "w")
    else:
        d["w"] = 1.0

    # Drop rows with missing prob or label; sort chronologically (nice for sanity)
    d = d.dropna(subset=["prob_long", "y"]).sort_values("timestamp").reset_index(drop=True)

    if len(d) == 0:
        raise ValueError("No aligned OOF probabilities for directional labels after cleaning.")

    # Build y (0/1), p (float), w (float)
    y = d["y"].map({-1: 0, 1: 1}).astype(int).to_numpy()
    p = d["prob_long"].astype(float).to_numpy()
    w = d["w"].fillna(1.0).astype(float).to_numpy()

    # --- Metrics ---
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = float("nan")

    try:
        brier = float(brier_score_loss(y, p, sample_weight=w))
    except TypeError:
        # older sklearn versions may not support sample_weight here
        brier = float(brier_score_loss(y, p))

    # --- ROC plot ---
    roc_path = None
    try:
        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC (AUC={auc:.3f})")
        roc_path = os.path.join(report_dir, f"roc_{tag}.png")
        plt.tight_layout(); plt.savefig(roc_path, dpi=120); plt.close()
    except Exception:
        pass  # skip plot gracefully

    # --- Reliability (calibration) plot ---
    calib_path = None
    try:
        frac_pos, mean_pred = calibration_curve(y.astype(int), p.astype(float),
                                                n_bins=20, strategy="quantile")
        plt.figure(figsize=(5, 4))
        plt.plot(mean_pred, frac_pos, marker="o", lw=1)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Reliability (Calibration) Curve")
        calib_path = os.path.join(report_dir, f"reliability_{tag}.png")
        plt.tight_layout(); plt.savefig(calib_path, dpi=120); plt.close()
    except Exception:
        pass  # skip plot gracefully

    # --- Summary JSON ---
    summary = {
        "n_samples": int(len(d)),
        "pos": int(d["y"].eq(1).sum()),
        "neg": int(d["y"].eq(-1).sum()),
        "auc": auc,
        "brier": brier,
        "roc_plot": roc_path,
        "calibration_plot": calib_path,
    }
    with open(os.path.join(report_dir, f"train_report_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
