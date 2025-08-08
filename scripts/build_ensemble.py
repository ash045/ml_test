import argparse, os
import pandas as pd
import numpy as np
from ml_signals.utils.config import load_config

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    artifacts = cfg["artifact_dir"]
    tf_list = [tf for tf in cfg["bars"]["timeframes"] if tf in ("1min","5min","15min","60min")]
    # gather available OOF files
    dfs = []
    for tf in tf_list:
        path = os.path.join(artifacts, f"oof_probs_{tf}.csv")
        if os.path.exists(path):
            d = pd.read_csv(path, parse_dates=["timestamp"]).rename(columns={"prob_long": f"prob_{tf}"})
            dfs.append(d)
    if not dfs:
        raise SystemExit("No per-timeframe OOF files found. Run train_models.py for each timeframe first.")

    # merge on timestamp
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on="timestamp", how="outer")

    # compute weighted average
    probs_cols = [c for c in base.columns if c.startswith("prob_")]
    weights = cfg["ensemble"].get("weights")
    if weights is None or len(weights) != len(probs_cols):
        weights = np.ones(len(probs_cols)) / len(probs_cols)
    else:
        weights = np.array(weights); weights = weights / weights.sum()

    P = base[probs_cols].fillna(0.5).values
    p_ens = P.dot(weights)

    out = base[["timestamp"]].copy()
    out["prob_long"] = p_ens
    os.makedirs(artifacts, exist_ok=True)
    out_path = os.path.join(artifacts, "oof_probs_ensemble.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote ensemble probabilities to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
