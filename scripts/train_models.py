import argparse, os
import pandas as pd
from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger
from ml_signals.pipeline.train_oof import train_oof_models
from ml_signals.reports.train_report import make_train_report

log = get_logger()

def main(cfg_path: str, timeframe: str = "1min"):
    cfg = load_config(cfg_path)
    symbol = cfg["data"]["symbol"]
    tf = timeframe

    processed = os.path.join(cfg["data"]["processed_dir"], f"{symbol}_{tf}_processed.csv")
    df = pd.read_csv(processed, parse_dates=["timestamp"])

    artifacts = cfg["artifact_dir"]
    os.makedirs(artifacts, exist_ok=True)

    # Train and get OOF probs
    oof = train_oof_models(df, cfg, artifacts)

    # Save TF-specific OOF (for ensembling later)
    oof_tf_path = os.path.join(artifacts, f"oof_probs_{tf}.csv")
    oof.to_csv(oof_tf_path, index=False)

    # Training report (AUC/Brier + plots)
    os.makedirs(cfg["report_dir"], exist_ok=True)
    make_train_report(df, oof, cfg["report_dir"], tag=tf)
    log.info(f"Wrote OOF ({tf}) to {oof_tf_path} and training report to {cfg['report_dir']}.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeframe", default="1min")
    args = ap.parse_args()
    main(args.config, timeframe=args.timeframe)
