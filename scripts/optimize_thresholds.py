"""Grid search threshold optimisation for trade signal probabilities.

This script reads out‑of‑fold (OOF) model probabilities and realised
returns, performs a grid search over long/short probability thresholds
and selects the thresholds that maximise total return on a training
slice.  A holdout slice is used to evaluate performance at the
selected thresholds and to search for a more robust pair among the
top‑ranking candidates.

The optimisation can be run for different bar timeframes and
return horizons by specifying ``--timeframe`` and ``--horizon``
arguments.  The timeframe determines which processed feature/return
file to load (e.g. ``240min``), while the horizon controls which
realised return column (e.g. ``ret_60``) is used.  If no horizon is
provided, the first entry in ``labeling.horizon_minutes`` from the
config is used.

Example usage:

    python scripts/optimize_thresholds.py --config configs/config.yaml \
        --timeframe 240min --horizon 60

Outputs are written into ``report_dir`` with filenames based on the
timeframe.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger

log = get_logger()


# -----------------------------------------------------------------------------
# Helper functions
#

def _to_utc_ts(s: pd.Series) -> pd.Series:
    """Parse a pandas Series into UTC timestamps."""
    return pd.to_datetime(s, utc=True, errors="coerce")


def _ensure_ret_col(df: pd.DataFrame, cfg: Dict[str, Any], horizon: Optional[int]) -> str:
    """Determine the realised return column to use for threshold evaluation.

    Priority is given to a ``ret_col`` specified under ``execution`` in the
    config.  Otherwise, the first horizon in ``labeling.horizon_minutes`` is
    selected unless the user provided a ``horizon`` argument.  If the chosen
    return column is missing from the DataFrame, it is computed on the fly
    using the ``close`` price.

    Parameters
    ----------
    df : pd.DataFrame
        The merged OOF/feature DataFrame.
    cfg : dict
        Parsed YAML configuration.
    horizon : Optional[int]
        Optional override for the return horizon in minutes.

    Returns
    -------
    str
        Name of the return column present or created in ``df``.
    """
    # explicit ret_col in config takes precedence
    ret_col_cfg = cfg.get("execution", {}).get("ret_col")
    if isinstance(ret_col_cfg, str) and ret_col_cfg in df.columns:
        return ret_col_cfg
    # determine horizon: user override or first horizon from config
    if horizon is not None:
        H = int(horizon)
    else:
        H_list = cfg.get("labeling", {}).get("horizon_minutes", [])
        if not H_list:
            raise KeyError("No horizon_minutes specified in config and no horizon argument provided")
        H = int(H_list[0])
    cand = f"ret_{H}"
    if cand not in df.columns:
        # compute on the fly using close prices
        if "close" not in df.columns:
            raise ValueError("No return column found and 'close' missing for on-the-fly computation.")
        df[cand] = df["close"].shift(-H) / df["close"] - 1.0
    return cand


def _equity_and_dd(returns: np.ndarray) -> tuple[np.ndarray, float]:
    eq = np.cumsum(returns)
    if len(eq):
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = float(dd.min())
    else:
        eq = np.array([])
        max_dd = 0.0
    return eq, max_dd


def _combine_actions(base: np.ndarray, mapped: np.ndarray, mode: str = "intersect") -> np.ndarray:
    base = base.astype(int)
    mapped = mapped.astype(int)
    if mode == "threshold_only":
        return base
    if mode == "mapper_only":
        return mapped
    if mode == "union":
        out = base.copy()
        mask = (out == 0)
        out[mask] = mapped[mask]
        return out
    # default: intersect — require same non-zero sign
    agree = (base != 0) & (mapped != 0) & (np.sign(base) == np.sign(mapped))
    out = np.zeros_like(base)
    out[agree] = base[agree]
    return out


def _build_hysteresis(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hyst_cfg = cfg.get("execution", {}).get("hysteresis", {})
    if not isinstance(hyst_cfg, dict):
        hyst_cfg = {
            "persistence_bars": 2,
            "cooldown_minutes": int(hyst_cfg) if hyst_cfg else 15,
        }
    return {
        "persistence_bars": int(hyst_cfg.get("persistence_bars", 2)),
        "cooldown_minutes": int(hyst_cfg.get("cooldown_minutes", 15)),
    }


def _maybe_apply_mapper(df: pd.DataFrame, cfg: Dict[str, Any], base_action: np.ndarray) -> tuple[np.ndarray, bool, Optional[Dict[str, Any]]]:
    combine_mode = cfg.get("execution", {}).get("mapper_combine", "intersect")
    if "sigma" not in df.columns:
        return base_action, False, None
    try:
        from ml_signals.execution.ev_mapping import map_probs_to_actions
    except Exception:
        return base_action, False, None
    try:
        hyst = _build_hysteresis(cfg)
        fee = float(cfg.get("execution", {}).get("fee_bps", 0.0))
        spread = float(cfg.get("execution", {}).get("spread_bps", 0.0))
        margin = float(cfg.get("execution", {}).get("cost_margin_mult", 1.0))
        k = float(cfg.get("labeling", {}).get("k_sigma", [2.0])[0])
        liq_gate = cfg.get("execution", {}).get("liquidity_gate", None)
        part_caps = cfg.get("execution", {}).get("participation_caps", None)

        df_map = map_probs_to_actions(
            df.copy(),
            "prob_long",
            "sigma",
            k,
            fee,
            spread,
            margin,
            hyst,
            liquidity_gate=liq_gate,
            participation_caps=part_caps,
        )  # returns columns: action_raw, action, ev_bps
        mapped = df_map["action"].values.astype(int)
        combined = _combine_actions(base_action, mapped, mode=combine_mode)

        # diagnostics
        diag = {
            "base": {
                "long": int((base_action == 1).sum()),
                "short": int((base_action == -1).sum()),
                "trades": int((base_action != 0).sum()),
            },
            "mapper": {
                "long": int((mapped == 1).sum()),
                "short": int((mapped == -1).sum()),
                "trades": int((mapped != 0).sum()),
            },
            "combo": {
                "long": int((combined == 1).sum()),
                "short": int((combined == -1).sum()),
                "trades": int((combined != 0).sum()),
            },
            "combine_mode": combine_mode,
        }
        return combined, True, diag
    except Exception:
        return base_action, False, None


def _evaluate_thresholds(
    data: pd.DataFrame,
    ret_col: str,
    fee_bps: float,
    spread_bps: float,
    t_long: float,
    t_short: float,
    apply_mapping: bool,
    cfg: Dict[str, Any],
    return_equity: bool = False,
) -> Dict[str, Any]:
    p = data["prob_long"].values.astype(float)
    r = data[ret_col].values.astype(float)

    base = np.zeros_like(p, dtype=int)
    base[p >= t_long] = 1
    base[p <= t_short] = -1

    diag = None
    if apply_mapping:
        a, used_mapper, diag = _maybe_apply_mapper(data, cfg, base)
    else:
        a, used_mapper = base, False

    trades = (a != 0).astype(float)
    cost_bps = float(fee_bps) + float(spread_bps)
    net = a * r - cost_bps / 1e4 * trades

    tot = float(np.nansum(net))
    n_trades = int(trades.sum())
    avg = float(np.nanmean(net[trades.astype(bool)])) if n_trades > 0 else 0.0
    eq, mdd = _equity_and_dd(net) if return_equity else (np.array([]), 0.0)

    out: Dict[str, Any] = {
        "t_long": float(t_long),
        "t_short": float(t_short),
        "total_return": tot,
        "trades": n_trades,
        "avg_per_trade": avg,
        "max_drawdown": float(mdd),
        "used_mapper": bool(used_mapper),
        "diag": diag,
    }
    if return_equity:
        out["equity"] = eq.tolist()
    return out


def _grid_search(
    train_df: pd.DataFrame,
    ret_col: str,
    fee_bps: float,
    spread_bps: float,
    apply_mapping: bool,
    cfg: Dict[str, Any],
    min_trades_train: int,
) -> tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    longs = np.linspace(0.60, 0.95, 15)
    shorts = np.linspace(0.05, 0.40, 15)

    rows: list[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for tL in longs:
        for tS in shorts:
            if not (tS < 0.5 < tL):
                continue
            res = _evaluate_thresholds(
                train_df,
                ret_col,
                fee_bps,
                spread_bps,
                float(tL),
                float(tS),
                apply_mapping,
                cfg,
                return_equity=False,
            )
            if res["trades"] < int(min_trades_train):
                continue
            rows.append(res)
            if best is None or res["total_return"] > best["total_return"]:
                best = res

    # If no rows were produced, return an empty DataFrame and None
    if not rows:
        return pd.DataFrame(), None
    # Otherwise construct a DataFrame and sort by total_return
    table = pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)
    return table, best


def _plot_heatmap(train_table: pd.DataFrame, path_png: str) -> None:
    if train_table.empty:
        return
    piv = train_table.pivot(index="t_short", columns="t_long", values="total_return")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="Total return (train)")
    xticks = np.linspace(0, piv.shape[1] - 1, num=min(10, piv.shape[1]), dtype=int)
    yticks = np.linspace(0, piv.shape[0] - 1, num=min(10, piv.shape[0]), dtype=int)
    plt.xticks(xticks, [f"{v:.2f}" for v in piv.columns.values[xticks]])
    plt.yticks(yticks, [f"{v:.2f}" for v in piv.index.values[yticks]])
    plt.xlabel("t_long")
    plt.ylabel("t_short")
    plt.title("OOF EV heatmap (TRAIN)")
    plt.tight_layout()
    plt.savefig(path_png, dpi=120)
    plt.close()


def _plot_equity(eq: np.ndarray, path_png: str, title: str) -> None:
    if eq.size == 0:
        return
    plt.figure(figsize=(8, 3.5))
    plt.plot(eq, lw=1.5)
    plt.xlabel("Events (chronological)")
    plt.ylabel("Cumulative return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# Main entry point
#

def main(cfg_path: str, timeframe: str = "1min", horizon: Optional[int] = None) -> None:
    cfg = load_config(cfg_path)
    sym = cfg["data"]["symbol"]
    # Determine the processed data file based on timeframe
    proc_path = os.path.join(cfg["data"]["processed_dir"], f"{sym}_{timeframe}_processed.csv")
    oof_path = os.path.join(cfg["artifact_dir"], "oof_probs.csv")

    # Load processed features and OOF probabilities
    df = pd.read_csv(proc_path)
    df["timestamp"] = _to_utc_ts(df["timestamp"])
    oof = pd.read_csv(oof_path)
    oof["timestamp"] = _to_utc_ts(oof["timestamp"])
    oof["prob_long"] = pd.to_numeric(oof["prob_long"], errors="coerce")

    # Merge on timestamp
    d = df.merge(oof[["timestamp", "prob_long"]], on="timestamp", how="inner")
    d = d.dropna(subset=["timestamp", "prob_long"]).sort_values("timestamp").reset_index(drop=True)

    # Choose realised return column based on horizon or config
    ret_col = _ensure_ret_col(d, cfg, horizon)
    d[ret_col] = pd.to_numeric(d[ret_col], errors="coerce")
    d = d.dropna(subset=[ret_col]).reset_index(drop=True)

    if len(d) < 200:
        log.warning(
            f"Only {len(d)} OOF events available after merge/clean. Results may be unstable."
        )

    fee_bps = float(cfg["execution"]["fee_bps"])
    spread_bps = float(cfg["execution"]["spread_bps"])

    # Holdout split / constraints / mapping toggle
    holdout_cfg = cfg.get("threshold_opt", {})
    frac = float(holdout_cfg.get("holdout_frac", 0.8))
    frac = min(max(frac, 0.5), 0.95)
    min_tr_train = int(holdout_cfg.get("min_trades_train", 100))
    min_tr_holdout = int(holdout_cfg.get("min_trades_holdout", 20))
    apply_mapping = bool(holdout_cfg.get("apply_execution_mapping", True))

    n = len(d)
    cut = max(int(n * frac), n - 200)
    cut = min(max(cut, 100), n - 50)
    train_df = d.iloc[:cut].copy()
    hold_df = d.iloc[cut:].copy()

    log.info(
        f"Threshold search on TRAIN slice: n={len(train_df)} | HOLDOUT: n={len(hold_df)} | apply_mapping={apply_mapping}"
    )

    # --- Search on TRAIN ---
    train_table, best_train = _grid_search(
        train_df,
        ret_col,
        fee_bps,
        spread_bps,
        apply_mapping,
        cfg,
        min_tr_train,
    )

    os.makedirs(cfg["report_dir"], exist_ok=True)
    base = os.path.join(cfg["report_dir"], f"threshold_{timeframe}")

    if best_train is None or train_table.empty:
        train_table.to_csv(f"{base}_grid_train.csv", index=False)
        with open(f"{base}_best.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "error": "no_thresholds_passed_min_trade_constraint",
                    "min_trades_train": min_tr_train,
                },
                f,
                indent=2,
            )
        log.warning(f"No thresholds met min_trades_train={min_tr_train}.")
        return

    # Evaluate best TRAIN thresholds on HOLDOUT
    best_on_hold = _evaluate_thresholds(
        hold_df,
        ret_col,
        fee_bps,
        spread_bps,
        best_train["t_long"],
        best_train["t_short"],
        apply_mapping,
        cfg,
        return_equity=True,
    )

    # Also find best-on-holdout among top-K train candidates
    topK = min(50, len(train_table))
    best_holdout: Optional[Dict[str, Any]] = None
    for i in range(topK):
        row = train_table.iloc[i]
        res = _evaluate_thresholds(
            hold_df,
            ret_col,
            fee_bps,
            spread_bps,
            row["t_long"],
            row["t_short"],
            apply_mapping,
            cfg,
            return_equity=False,
        )
        if res["trades"] < min_tr_holdout:
            continue
        if (best_holdout is None) or (res["total_return"] > best_holdout["total_return"]):
            best_holdout = res

    # --- Outputs ---
    # Save TRAIN grid + heatmap + top10
    grid_csv = f"{base}_grid_train.csv"
    train_table.to_csv(grid_csv, index=False)
    heatmap_png = f"{base}_heatmap_train.png"
    _plot_heatmap(train_table, heatmap_png)
    top10_csv = f"{base}_top10_train.csv"
    train_table.head(10).to_csv(top10_csv, index=False)

    # HOLDOUT equity of train-best
    eq = np.array(best_on_hold.get("equity", []), dtype=float)
    eq_png = f"{base}_equity_holdout.png"
    _plot_equity(eq, eq_png, title=f"Holdout equity | tL={best_train['t_long']:.3f}, tS={best_train['t_short']:.3f}")

    # --- DIAGNOSTICS: print action counts ---
    def _fmt_diag(dg: Optional[Dict[str, Any]]) -> str:
        if not dg:
            return "n/a"
        b = dg["base"]
        m = dg["mapper"]
        c = dg["combo"]
        return (
            f"base L/S/T={b['long']}/{b['short']}/{b['trades']} | "
            f"mapper L/S/T={m['long']}/{m['short']}/{m['trades']} | "
            f"combo L/S/T={c['long']}/{c['short']}/{c['trades']} | "
            f"mode={dg['combine_mode']}"
        )

    log.info("TRAIN diag @ train-best → " + _fmt_diag(best_train.get("diag")))
    log.info("HOLDOUT diag @ train-best → " + _fmt_diag(best_on_hold.get("diag")))
    if best_holdout:
        log.info("HOLDOUT diag @ best-on-holdout → " + _fmt_diag(best_holdout.get("diag")))

    # Summary JSON (also include legacy-compatible keys)
    def _flt(v: Any) -> Any:
        return float(v) if isinstance(v, (np.floating, np.integer)) else v

    summary = {
        "ret_col": ret_col,
        "holdout_frac": frac,
        "apply_execution_mapping": apply_mapping,
        "combine_mode": cfg.get("execution", {}).get("mapper_combine", "intersect"),
        "constraints": {"min_trades_train": min_tr_train, "min_trades_holdout": min_tr_holdout},
        "train_best": {k: _flt(v) for k, v in best_train.items() if k != "equity"},
        "holdout_at_train_best": {
            k: _flt(v) for k, v in best_on_hold.items() if k != "equity"
        },
        "best_on_holdout_among_top_train": (
            {k: _flt(v) for k, v in best_holdout.items()} if best_holdout else None
        ),
        # --- compatibility for old loaders ---
        "train": {"t_long": float(best_train["t_long"]), "t_short": float(best_train["t_short"])}
        if best_train
        else None,
        "best": {"t_long": float(best_train["t_long"]), "t_short": float(best_train["t_short"])}
        if best_train
        else None,
        "artifacts": {
            "grid_train_csv": grid_csv,
            "heatmap_train_png": heatmap_png,
            "top10_train_csv": top10_csv,
            "equity_holdout_png": eq_png,
        },
    }
    with open(f"{base}_best.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"Best TRAIN thresholds: t_long={best_train['t_long']:.3f}, t_short={best_train['t_short']:.3f}, "
        f"trades={best_train['trades']}, total_return={best_train['total_return']:.6f}, "
        f"avg_per_trade={best_train['avg_per_trade']:.6f}, max_drawdown={best_train['max_drawdown']:.6f}, "
        f"used_mapper={best_train.get('used_mapper', False)}"
    )
    log.info(
        f"HOLDOUT @ train-best: trades={best_on_hold['trades']}, "
        f"total_return={best_on_hold['total_return']:.6f}, "
        f"avg_per_trade={best_on_hold['avg_per_trade']:.6f}, "
        f"max_drawdown={best_on_hold['max_drawdown']:.6f}"
    )
    if best_holdout:
        log.info(
            f"Best-on-HOLDOUT among top {min(50, len(train_table))}: "
            f"tL={best_holdout['t_long']:.3f}, tS={best_holdout['t_short']:.3f}, "
            f"trades={best_holdout['trades']}, total_return={best_holdout['total_return']:.6f}"
        )
    log.info(
        f"Artifacts → grid:{grid_csv} | heatmap:{heatmap_png} | top10:{top10_csv} | equity (holdout):{eq_png}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise long/short probability thresholds via grid search")
    parser.add_argument("--config", required=True, help="Path to training configuration YAML")
    parser.add_argument(
        "--timeframe",
        default="1min",
        help="Bar timeframe to use (e.g. 1min, 5min, 240min). Determines which processed data file is read.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Return horizon in minutes for realised returns (e.g. 60). Overrides the first entry in labeling.horizon_minutes.",
    )
    args = parser.parse_args()
    main(args.config, timeframe=args.timeframe, horizon=args.horizon)