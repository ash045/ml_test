# scripts/train_models.py
# -*- coding: utf-8 -*-
"""Train models → OOF + extra report (per-fold metrics, ROC/Rel, decile lift)."""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger
from ml_signals.pipeline.train_oof import train_oof_models
from ml_signals.reports.train_report import make_train_report
from ml_signals.metrics.sharpe import weighted_sharpe_ratio

log = get_logger()


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _merge_oof(df_proc: pd.DataFrame, oof_df: pd.DataFrame) -> pd.DataFrame:
    m = df_proc["y"].isin([-1, 1])
    base = df_proc.loc[m, ["timestamp", "y", "w", "ev_ret", "ev_ret_short"]].copy()
    out = base.merge(oof_df, on="timestamp", how="inner")
    out["y_bin"] = out["y"].map({-1: 0, 1: 1}).astype(int)
    return out


def _plot_roc(y: np.ndarray, p: np.ndarray, path: str, title: str) -> None:
    # Drop NaNs and guard against degenerate labels
    mask = np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{title} (insufficient data)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{title} (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_reliability(y: np.ndarray, p: np.ndarray, path: str, title: str) -> None:
    mask = np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{title} (insufficient data)")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return
    prob_true, prob_pred = calibration_curve(y, p, n_bins=20, strategy="quantile")
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _decile_lift(dfm: pd.DataFrame, prob_col: str, fee_bps: float, spread_bps: float) -> pd.DataFrame:
    d = dfm.dropna(subset=[prob_col]).copy()
    d["decile"] = pd.qcut(d[prob_col], q=10, labels=False, duplicates="drop")
    rows = []
    cost = (fee_bps + spread_bps) / 1e4
    for dec in sorted(d["decile"].dropna().unique()):
        dd = d[d["decile"] == dec]
        rets_long = dd["ev_ret"].to_numpy() - cost
        w = dd["w"].to_numpy()
        rows.append(
            {
                "decile": int(dec),
                "count": int(len(dd)),
                "mean_long_after_cost": float(np.average(rets_long, weights=w)) if len(dd) else np.nan,
                "sharpe_long_after_cost": float(weighted_sharpe_ratio(rets_long, w)) if len(dd) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("decile")


def _top_bottom_spread_sharpe(
    dfm: pd.DataFrame, prob_col: str, fee_bps: float, spread_bps: float
) -> float:
    d = dfm.dropna(subset=[prob_col]).copy()
    d["decile"] = pd.qcut(d[prob_col], q=10, labels=False, duplicates="drop")
    if d["decile"].isna().all():
        return float("nan")
    top = d[d["decile"] == d["decile"].max()]
    bot = d[d["decile"] == d["decile"].min()]
    cost = (fee_bps + spread_bps) / 1e4
    rets = np.zeros(len(d))
    w = d["w"].to_numpy()
    rets[np.isin(d.index, top.index)] = top["ev_ret"].to_numpy() - cost
    rets[np.isin(d.index, bot.index)] = bot["ev_ret_short"].to_numpy() - cost
    return float(weighted_sharpe_ratio(rets, w))


def make_extra_report(
    df_proc: pd.DataFrame,
    oof_df: pd.DataFrame,
    report_dir: str,
    timeframe: str,
    fee_bps: float,
    spread_bps: float,
) -> None:
    _ensure_dir(report_dir)
    merged = _merge_oof(df_proc, oof_df)
    # Drop warmup rows without OOF predictions (expanding splitter leaves NaNs at the start)
    merged = merged.dropna(subset=["prob_long"]).copy()
    y = merged["y_bin"].to_numpy()

    # Per-fold metrics (guard against degenerate folds)
    per_fold = []
    for f, g in merged.groupby("fold"):
        gg = g.dropna(subset=["prob_long"])
        yy = gg["y_bin"].to_numpy()
        pp = gg["prob_long"].to_numpy()
        if len(yy) == 0 or len(np.unique(yy)) < 2:
            auc = pr = brier = np.nan
        else:
            auc = roc_auc_score(yy, pp)
            pr = average_precision_score(yy, pp)
            brier = brier_score_loss(yy, pp)
        per_fold.append({"fold": int(f), "auc": auc, "pr_auc": pr, "brier": brier, "count": int(len(gg))})
    pd.DataFrame(per_fold).to_csv(os.path.join(report_dir, f"per_fold_metrics_{timeframe}.csv"), index=False)

    # ROC + Reliability (calibrated)
    p_cal = merged["prob_long"].to_numpy()
    _plot_roc(y, p_cal, os.path.join(report_dir, f"roc_{timeframe}.png"), "ROC")
    _plot_reliability(y, p_cal, os.path.join(report_dir, f"reliability_{timeframe}.png"), "Reliability (Calibration) Curve")

    # Raw variants if available
    if "prob_long_raw" in merged.columns:
        p_raw = merged["prob_long_raw"].to_numpy()
        _plot_roc(y, p_raw, os.path.join(report_dir, f"roc_raw_{timeframe}.png"), "ROC (raw)")
        _plot_reliability(y, p_raw, os.path.join(report_dir, f"reliability_raw_{timeframe}.png"), "Reliability (raw)")

    # Decile lift
    dec = _decile_lift(merged, "prob_long", fee_bps, spread_bps)
    dec.to_csv(os.path.join(report_dir, f"decile_lift_{timeframe}.csv"), index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(dec["decile"].astype(str), dec["sharpe_long_after_cost"])
    plt.title("After-cost Sharpe by decile (long)")
    plt.xlabel("Decile (low→high prob_long)")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"decile_sharpe_{timeframe}.png"))
    plt.close()

    spread_sharpe = _top_bottom_spread_sharpe(merged, "prob_long", fee_bps, spread_bps)
    log.info(f"Top-vs-bottom decile spread Sharpe (after-cost): {spread_sharpe:.3f}")


def main(cfg_path: str, timeframe: str = "1min") -> None:
    cfg = load_config(cfg_path)
    symbol = cfg["data"]["symbol"]
    processed = os.path.join(cfg["data"]["processed_dir"], f"{symbol}_{timeframe}_processed.csv")

    if not os.path.exists(processed):
        raise FileNotFoundError(f"Processed dataset missing: {processed}. Run scripts/train.py first.")

    df = pd.read_csv(processed, parse_dates=["timestamp"])

    # Grids & params
    grid_logit = cfg.get("models", {}).get("logit", {})
    grid_lgbm = cfg.get("models", {}).get("lgbm", {})
    grid_catb = cfg.get("models", {}).get("catb", {})

    fee_bps = cfg.get("costs", {}).get("fee_bps", 0)
    spread_bps = cfg.get("costs", {}).get("spread_bps", 0)
    embargo = cfg.get("validation", {}).get("embargo", 0)

    oof_df, feat_imps = train_oof_models(
        df=df,
        cfg=cfg,
        grid_logit=grid_logit,
        grid_lgbm=grid_lgbm,
        grid_catb=grid_catb,
        fee_bps=fee_bps,
        spread_bps=spread_bps,
        embargo=embargo,
    )

    # convenience
    oof_df["prob_short"] = 1.0 - oof_df["prob_long"]
    if "prob_long_raw" in oof_df.columns:
        oof_df["prob_short_raw"] = 1.0 - oof_df["prob_long_raw"]

    artifacts = cfg["artifact_dir"]
    _ensure_dir(artifacts)
    oof_tf_path = os.path.join(artifacts, f"oof_probs_{timeframe}.csv")
    oof_df.to_csv(oof_tf_path, index=False)

    if isinstance(feat_imps, pd.DataFrame) and not feat_imps.empty:
        feat_path = os.path.join(artifacts, "feature_importance.csv")
        feat_imps.to_csv(feat_path, index=False)

    _ensure_dir(cfg["report_dir"])
    make_train_report(df_proc=df, oof_df=oof_df, report_dir=cfg["report_dir"], tag=timeframe)
    make_extra_report(
        df_proc=df,
        oof_df=oof_df,
        report_dir=cfg["report_dir"],
        timeframe=timeframe,
        fee_bps=fee_bps,
        spread_bps=spread_bps,
    )
    log.info(f"Wrote OOF to {oof_tf_path} and extra report artifacts to {cfg['report_dir']}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeframe", default="1min")
    args = ap.parse_args()
    main(args.config, timeframe=args.timeframe)
