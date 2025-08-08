import argparse
import os
import pandas as pd

from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger
from ml_signals.utils.seed import set_seed

# Updated imports
from ml_signals.data.ingestion import read_ohlcv_csv, audit_ohlcv
from ml_signals.bars.time_bars import resample_time_bars

from ml_signals.features.core import add_core_features
from ml_signals.features.denoise import apply_denoise
from ml_signals.features.fdiff import add_fracdiff_feature

# Use new multi-horizon labeling
from ml_signals.labels.triple_barrier_multi import triple_barrier_multi_labels
from ml_signals.labels.realized import (
    add_event_realized_returns,
    add_event_realized_returns_from,
)
# Keep concurrency weights function
from ml_signals.labels.triple_barrier import compute_concurrency_weights

log = get_logger()


def main(cfg_path: str, timeframe: str = "1min"):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))

    # --- Ingest & audit ---
    raw_dir, symbol = cfg["data"]["raw_dir"], cfg["data"]["symbol"]
    src = os.path.join(raw_dir, f"{symbol}.csv")
    df = read_ohlcv_csv(src, cfg["data"]["timestamp_col"])
    audit_ohlcv(df)

    # --- Resample to requested timeframe ---
    df = resample_time_bars(df, timeframe, cfg["data"]["timestamp_col"])

    # --- Core features / optional fractional diff / denoise ---
    df = add_core_features(df)
    if cfg["features"]["fractional_diff"]["enabled"]:
        d_val = cfg["features"]["fractional_diff"]["d_candidates"][0]
        df = add_fracdiff_feature(df, d_val)
    df = apply_denoise(df, cfg["features"]["denoise"]["ema_windows"])

    # --- Labels (triple barrier) for multiple horizons/k values ---
    horizons = list(map(int, cfg["labeling"]["horizon_minutes"]))
    k_list   = list(map(float, cfg["labeling"]["k_sigma"]))
    sigma_window = int(cfg["labeling"]["sigma_window"])

    lab_multi = triple_barrier_multi_labels(df, sigma_window, horizons, k_list, price_col=cfg["data"]["price_col"])

    # Merge sigma into df
    df["sigma"] = lab_multi["sigma"]

    # Choose primary H0 and k0 (first values) for y/t_end
    H0 = horizons[0]
    k0 = k_list[0]
    col_y = f"y_{H0}_{k0}"
    col_tend = f"t_end_{H0}_{k0}"
    df["y"] = lab_multi[col_y]
    df["t_end"] = lab_multi[col_tend]

    # --- Sample weights: incorporate uniqueness AND magnitude of realized returns ---
    # When enabled, compute concurrency-based uniqueness weights (1/overlap) and scale them
    # by the absolute realized return of the primary event.  This encourages the model
    # to pay more attention to larger moves while still downweighting overlapping events.
    if cfg["labeling"].get("use_uniqueness_weights", False):
        # First, compute uniqueness (inverse concurrency)
        uniq = compute_concurrency_weights(df["t_end"])
        # Next, take absolute realized returns; NaN returns are treated as zero
        abs_ret = df.get("ev_ret").abs().fillna(0.0)
        df["w"] = uniq * abs_ret
    else:
        df["w"] = 1.0

    # --- Realized returns for PRIMARY horizon ---
    df = add_event_realized_returns(df, price_col=cfg["data"]["price_col"])

    # --- EXTRA realized returns for ADDITIONAL horizons ---
    if len(horizons) > 1:
        for H in horizons[1:]:
            # use the same k0 when computing realized returns (training target remains the primary)
            col_tend_H = f"t_end_{H}_{k0}"
            if col_tend_H not in lab_multi.columns:
                # compute t_end for this horizon using primary k0
                # fallback: compute single horizon using triple_barrier_multi_labels
                continue
            df[col_tend_H] = lab_multi[col_tend_H]
            df = add_event_realized_returns_from(
                df,
                t_end_col=col_tend_H,
                price_col=cfg["data"]["price_col"],
                suffix=f"_{H}"
            )

    # --- Persist processed dataset ---
    out_dir = cfg["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_{timeframe}_processed.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Wrote processed dataset to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeframe", default="1min")
    args = ap.parse_args()
    main(args.config, timeframe=args.timeframe)
