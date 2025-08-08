import argparse, os
import pandas as pd

from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger
from ml_signals.utils.seed import set_seed

from ml_signals.data.ingestion import read_ohlcv_csv, audit_ohlcv
from ml_signals.bars.time_bars import resample_time_bars

from ml_signals.features.core import add_core_features
from ml_signals.features.denoise import apply_denoise
from ml_signals.features.fdiff import add_fracdiff_feature

from ml_signals.labels.triple_barrier import (
    triple_barrier_labels,
    compute_concurrency_weights,
)
from ml_signals.labels.realized import (
    add_event_realized_returns,
    add_event_realized_returns_from,
)

log = get_logger()

def main(cfg_path: str, timeframe: str = "1min"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    # --- Ingest & audit ---
    raw_dir, symbol = cfg["data"]["raw_dir"], cfg["data"]["symbol"]
    src = os.path.join(raw_dir, f"{symbol}.csv")
    df = read_ohlcv_csv(src, cfg["data"]["timestamp_col"])
    audit_ohlcv(df)

    # --- Resample to requested timeframe (still time bars here) ---
    df = resample_time_bars(df, timeframe, cfg["data"]["timestamp_col"])

    # --- Core features / optional fractional diff / denoise ---
    df = add_core_features(df)
    if cfg["features"]["fractional_diff"]["enabled"]:
        d_val = cfg["features"]["fractional_diff"]["d_candidates"][0]
        df = add_fracdiff_feature(df, d_val)
    df = apply_denoise(df, cfg["features"]["denoise"]["ema_windows"])

    # --- Labels (triple barrier) for PRIMARY horizon/k ---
    horizons = cfg["labeling"]["horizon_minutes"]
    k_list   = cfg["labeling"]["k_sigma"]
    H0       = int(horizons[0])
    k0       = float(k_list[0])
    sigma_window = int(cfg["labeling"]["sigma_window"])

    lab0 = triple_barrier_labels(df, sigma_window, k0, H0)
    # primary y / t_end / sigma (+ tp/sl pct)
    df = pd.concat([df, lab0], axis=1)

    # --- Sample weights (Betas / concurrency) based on PRIMARY t_end ---
    if cfg["labeling"]["use_uniqueness_weights"]:
        df["w"] = compute_concurrency_weights(df["t_end"])
    else:
        df["w"] = 1.0

    # --- Realized returns for PRIMARY horizon ---
    df = add_event_realized_returns(df, price_col="close")  # ev_ret, ev_ret_short, ev_bps

    # --- EXTRA realized returns for ADDITIONAL horizons (suffix columns) ---
    # We keep training on primary y/t_end/sigma, but we also compute ev_ret_<H>
    # so you can choose execution.ret_col=ev_ret_<H> in backtests.
    if len(horizons) > 1:
        for H in map(int, horizons[1:]):
            # reuse same k0 for now to avoid changing the training target
            labH = triple_barrier_labels(df, sigma_window, k0, H)
            df[f"t_end_{H}"] = labH["t_end"]
            # realized returns with suffix _<H>
            df = add_event_realized_returns_from(df, t_end_col=f"t_end_{H}", price_col="close", suffix=f"_{H}")

    # --- Persist processed dataset ---
    out_dir = cfg["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{symbol}_{timeframe}_processed.csv")
    df.to_csv(out, index=False)
    log.info(f"Wrote processed dataset to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeframe", default="1min")
    args = ap.parse_args()
    main(args.config, timeframe=args.timeframe)
