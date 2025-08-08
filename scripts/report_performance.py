import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_signals.utils.config import load_config
from ml_signals.backtest.reports import summary

def perf_by_bucket(df: pd.DataFrame, col: str, n_bins: int = 4):
    df = df.copy()
    df = df[df[col].notna()]
    df["bucket"] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
    out = []
    for b, grp in df.groupby("bucket"):
        s = summary(grp)
        s["bucket"] = int(b)
        out.append(s)
    return pd.DataFrame(out).sort_values("bucket")

def main(cfg_path: str, equity_csv: str = None):
    cfg = load_config(cfg_path)
    symbol = cfg["data"]["symbol"]
    report_dir = cfg["report_dir"]
    if equity_csv is None:
        # default to ensemble if present
        path_ens = os.path.join(report_dir, f"{symbol}_1min_equity_ensemble.csv")
        path_single = os.path.join(report_dir, f"{symbol}_1min_equity.csv")
        equity_csv = path_ens if os.path.exists(path_ens) else path_single

    df = pd.read_csv(equity_csv, parse_dates=["timestamp"])

    # Merge original features to get sigma and trades for bucketing
    proc_path = os.path.join(cfg["data"]["processed_dir"], f"{symbol}_1min_processed.csv")
    base = pd.read_csv(proc_path, parse_dates=["timestamp"])
    df = df.merge(base[["timestamp","sigma","trades"]], on="timestamp", how="left")

    # Buckets
    os.makedirs(report_dir, exist_ok=True)
    by_vol = perf_by_bucket(df, "sigma", 4)
    by_trd = perf_by_bucket(df, "trades", 4)

    by_vol.to_csv(os.path.join(report_dir, "perf_by_vol.csv"), index=False)
    by_trd.to_csv(os.path.join(report_dir, "perf_by_trades.csv"), index=False)
    print("Wrote perf_by_vol.csv and perf_by_trades.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--equity_csv", default=None)
    args = ap.parse_args()
    main(args.config, equity_csv=args.equity_csv)
