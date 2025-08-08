# scripts/train_models.py
# -*- coding: utf-8 -*-
"""Train models and write OOF + report (expects DataFrame from train_oof_models).

Fix:
- Handle the new return type from `train_oof_models`: (oof_df, feat_imps).
- Save OOF with `timestamp, prob_long`.
- Call `make_train_report(df_proc=df, oof_df=oof_df, ...)`.
"""
import argparse
import os
import pandas as pd
from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger
from ml_signals.pipeline.train_oof import train_oof_models
from ml_signals.reports.train_report import make_train_report

log = get_logger()


def main(cfg_path: str, timeframe: str = "1min") -> None:
    cfg = load_config(cfg_path)
    symbol = cfg["data"]["symbol"]
    processed = os.path.join(cfg["data"]["processed_dir"], f"{symbol}_{timeframe}_processed.csv")

    if not os.path.exists(processed):
        raise FileNotFoundError(f"Processed dataset missing: {processed}. Run scripts/train.py first.")

    df = pd.read_csv(processed, parse_dates=["timestamp"])  # required by report merge

    # Grids & params
    grid_logit = cfg.get("models", {}).get("logit", {})
    grid_lgbm = cfg.get("models", {}).get("lgbm", {})
    grid_catb = cfg.get("models", {}).get("catb", {})

    fee_bps = cfg.get("costs", {}).get("fee_bps", 0)
    spread_bps = cfg.get("costs", {}).get("spread_bps", 0)
    embargo = cfg.get("validation", {}).get("embargo", 0)

    # Train → returns OOF DataFrame + importances
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

    # Persist artifacts
    artifacts = cfg["artifact_dir"]
    os.makedirs(artifacts, exist_ok=True)

    oof_tf_path = os.path.join(artifacts, f"oof_probs_{timeframe}.csv")
    oof_df.to_csv(oof_tf_path, index=False)

    if isinstance(feat_imps, pd.DataFrame) and not feat_imps.empty:
        feat_path = os.path.join(artifacts, "feature_importance.csv")
        feat_imps.to_csv(feat_path, index=False)

    # Report
    os.makedirs(cfg["report_dir"], exist_ok=True)
    make_train_report(df_proc=df, oof_df=oof_df, report_dir=cfg["report_dir"], tag=timeframe)

    log.info(f"Wrote OOF ({timeframe}) to {oof_tf_path} and training report to {cfg['report_dir']}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeframe", default="1min")
    args = ap.parse_args()
    main(args.config, timeframe=args.timeframe)
