"""
Execute an event- and bar-based backtest using trained model probabilities.

This script loads processed feature/return data and corresponding out-of-fold
probabilities, converts probability thresholds into trading actions, applies
optional execution mapping, and computes performance metrics on either
event-based (signal) or bar-based (position) returns.  It automatically
loads thresholds from the optimiser JSON for the specified timeframe, and
falls back to the default `signals` section in the config if no optimiser
results are available.

Enhancements over the original version:

* **Timeframe-specific OOF probabilities** – reads `oof_probs_{timeframe}.csv` if it exists,
  ensuring alignment between the processed data and the out-of-fold predictions.
  Logs a warning and falls back to `oof_probs.csv` if the file is missing.
* **Cleaner logging and diagnostics** – retains all functionality of the original script
  while improving robustness when the OOF file is missing or thresholds are absent.
"""

import argparse
import json
import os
import inspect
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger

log = get_logger()


def _to_utc_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def _plot_equity(eq: np.ndarray, path_png: str, title: str) -> None:
    if eq.size == 0:
        return
    plt.figure(figsize=(8, 3.5))
    plt.plot(eq, lw=1.5)
    plt.xlabel("Events/Bars (chronological)")
    plt.ylabel("Cumulative return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=120)
    plt.close()


def _read_optimizer_json(report_dir: str, timeframe: str) -> Optional[dict]:
    """Load thresholds + holdout_frac from optimiser JSON."""
    p_json = os.path.join(report_dir, f"threshold_{timeframe}_best.json")
    if not os.path.exists(p_json):
        return None
    try:
        with open(p_json, "r", encoding="utf-8") as f:
            js = json.load(f)
        return js
    except Exception as e:
        log.warning(f"Failed to read optimizer JSON at {p_json}: {e}")
        return None


def _choose_thresholds(js: Optional[dict], pref: str = "auto") -> tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Choose thresholds from optimiser JSON:
      pref='train_best' → use train_best
      pref='best_on_holdout' → use best_on_holdout_among_top_train (if present)
      pref='auto' → prefer train_best, else best_on_holdout, else legacy train/best
    Returns (tL, tS, source_str) or (None,None,None)
    """
    if js is None:
        return None, None, None

    def _has(obj, *keys) -> bool:
        return obj is not None and all(k in obj for k in keys)

    if pref == "train_best":
        tb = js.get("train_best")
        if _has(tb, "t_long", "t_short"):
            return float(tb["t_long"]), float(tb["t_short"]), "optimizer:train_best"
    elif pref == "best_on_holdout":
        boh = js.get("best_on_holdout_among_top_train")
        if _has(boh, "t_long", "t_short"):
            return float(boh["t_long"]), float(boh["t_short"]), "optimizer:best_on_holdout"

    if pref == "auto":
        tb = js.get("train_best")
        if _has(tb, "t_long", "t_short"):
            return float(tb["t_long"]), float(tb["t_short"]), "optimizer:train_best"
        boh = js.get("best_on_holdout_among_top_train")
        if _has(boh, "t_long", "t_short"):
            return float(boh["t_long"]), float(boh["t_short"]), "optimizer:best_on_holdout"

    # legacy schema
    if _has(js.get("train", {}), "t_long", "t_short"):
        tr = js["train"]
        return float(tr["t_long"]), float(tr["t_short"]), "optimizer:train"
    if _has(js.get("best", {}), "t_long", "t_short"):
        be = js["best"]
        return float(be["t_long"]), float(be["t_short"]), "optimizer:best"

    return None, None, None


def _compute_action(prob: pd.Series, t_long: float, t_short: float) -> pd.Series:
    p = prob.values.astype(float)
    a = np.zeros_like(p, dtype=int)
    a[p >= t_long] = 1
    a[p <= t_short] = -1
    return pd.Series(a, index=prob.index, name="action")


def _combine_actions(base: np.ndarray, mapped: np.ndarray, mode: str = "intersect") -> np.ndarray:
    base = base.astype(int)
    mapped = mapped.astype(int)
    if mode == "threshold_only":
        return base
    if mode == "mapper_only":
        return mapped
    if mode == "union":
        out = base.copy()
        mask = base == 0
        out[mask] = mapped[mask]
        return out
    # default: intersect — require same non-zero sign
    agree = (base != 0) & (mapped != 0) & (np.sign(base) == np.sign(mapped))
    out = np.zeros_like(base)
    out[agree] = base[agree]
    return out


def _apply_execution_mapping(df: pd.DataFrame, cfg: dict, base_action: pd.Series) -> pd.Series:
    combine_mode = cfg.get("execution", {}).get("mapper_combine", "intersect")
    if "sigma" not in df.columns:
        return base_action
    try:
        from ml_signals.execution.ev_mapping import map_probs_to_actions
        fee = float(cfg.get("execution", {}).get("fee_bps", 0.0))
        spread = float(cfg.get("execution", {}).get("spread_bps", 0.0))
        margin = float(cfg.get("execution", {}).get("cost_margin_mult", 1.0))
        k = float(cfg.get("labeling", {}).get("k_sigma", [2.0])[0])
        hyst_cfg = cfg.get("execution", {}).get("hysteresis", {})
        if not isinstance(hyst_cfg, dict):
            hyst_cfg = {"persistence_bars": 2, "cooldown_minutes": int(hyst_cfg) if hyst_cfg else 15}
        hyst = {
            "persistence_bars": int(hyst_cfg.get("persistence_bars", 2)),
            "cooldown_minutes": int(hyst_cfg.get("cooldown_minutes", 15)),
        }
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
        combined = _combine_actions(base_action.values.astype(int), mapped, mode=combine_mode)
        return pd.Series(combined, index=df.index, name="action")
    except Exception as e:
        log.warning(f"Execution mapping failed: {e}. Using thresholds only.")
        return base_action


def _event_backtest(df: pd.DataFrame, ret_col: str, fee_bps: float, spread_bps: float) -> dict:
    a = df["action"].astype(int).values
    r = df[ret_col].astype(float).values
    trades = (a != 0).astype(float)
    cost_bps = float(fee_bps) + float(spread_bps)
    net = a * r - cost_bps / 1e4 * trades
    eq = np.cumsum(net)
    return {
        "total_return": float(np.nansum(net)),
        "trades": int(trades.sum()),
        "avg_per_trade": float(np.nanmean(net[trades.astype(bool)])) if trades.sum() > 0 else 0.0,
        "max_drawdown": _max_drawdown(eq),
        "equity": eq.tolist(),
    }


def _bar_backtest(
    df: pd.DataFrame,
    ret_col: str,
    fee_bps: float,
    spread_bps: float,
    slippage_alpha: float,
    adt_window: int,
    cost_margin_mult: float,
) -> dict:
    try:
        from ml_signals.backtest.engine import backtest_market
        sig = inspect.signature(backtest_market)
        call_args = [df.copy(), "action", ret_col, float(fee_bps), float(spread_bps), float(slippage_alpha)]
        kwargs = {}
        if "adt_window" in sig.parameters:
            kwargs["adt_window"] = int(adt_window)
        if "cost_margin_mult" in sig.parameters:
            kwargs["cost_margin_mult"] = float(cost_margin_mult)
        res = backtest_market(*call_args, **kwargs)
        if isinstance(res, dict) and "equity" in res:
            return res
        log.warning("Engine returned unexpected structure; using fallback bar backtest.")
    except Exception as e:
        log.warning(f"Engine backtest unavailable/failed ({e}); using fallback bar backtest.")
    a = df["action"].astype(int).values
    r = df[ret_col].astype(float).values
    a_prev = np.roll(a, 1)
    a_prev[0] = 0
    # entry change: 1.0 where position flips from 0→±1 or from ±1→0 or sign change
    # Use boolean arithmetic to avoid type errors with bitwise & on floats
    entry_change = ((a != a_prev) & (a != 0)).astype(float)
    cost_bps = (float(fee_bps) + float(spread_bps)) * float(cost_margin_mult)
    net = a * r - cost_bps / 1e4 * entry_change
    eq = np.cumsum(net)
    return {
        "total_return": float(np.nansum(net)),
        "trades": int(entry_change.sum()),
        "avg_per_trade": float(np.nanmean(net[entry_change.astype(bool)])) if entry_change.sum() > 0 else 0.0,
        "max_drawdown": _max_drawdown(eq),
        "equity": eq.tolist(),
    }


def main(
    cfg_path: str,
    timeframe: str = "1min",
    thresholds_path: str = None,
    threshold_source: str = "auto",
    eval_slice: str = "all",
) -> None:
    """
    Run a backtest for the specified timeframe.  If ``thresholds_path`` is given,
    thresholds are loaded from that JSON; otherwise they are taken from the
    optimiser JSON, the config's ``signals`` section, or defaults in that order.

    :param cfg_path: path to training config YAML
    :param timeframe: bar timeframe (e.g. 1min, 5min, 240min)
    :param thresholds_path: optional explicit thresholds JSON
    :param threshold_source: train_best|best_on_holdout|auto (default auto)
    :param eval_slice: all|train|holdout (subset of data for evaluation)
    """
    cfg = load_config(cfg_path)
    sym = cfg["data"]["symbol"]
    proc_path = os.path.join(cfg["data"]["processed_dir"], f"{sym}_{timeframe}_processed.csv")
    # Use timeframe-specific OOF predictions when available
    oof_tf_path = os.path.join(cfg["artifact_dir"], f"oof_probs_{timeframe}.csv")
    if os.path.exists(oof_tf_path):
        oof_path = oof_tf_path
    else:
        oof_path = os.path.join(cfg["artifact_dir"], "oof_probs.csv")
        log.warning(
            f"Timeframe‑specific OOF file missing: {oof_tf_path}. Using fallback {oof_path}."
        )
    report_dir = cfg["report_dir"]

    # Load data
    df = pd.read_csv(proc_path)
    df["timestamp"] = _to_utc_ts(df["timestamp"])
    oof = pd.read_csv(oof_path)
    oof["timestamp"] = _to_utc_ts(oof["timestamp"])
    oof["prob_long"] = pd.to_numeric(oof["prob_long"], errors="coerce")

    # Merge
    d = df.merge(oof[["timestamp", "prob_long"]], on="timestamp", how="inner")
    d = d.dropna(subset=["timestamp", "prob_long"]).sort_values("timestamp").reset_index(drop=True)
    if len(d) == 0:
        raise ValueError("No rows after merging processed data with OOF probabilities.")

    # Load optimizer JSON (for thresholds and holdout_frac)
    js = _read_optimizer_json(report_dir, timeframe)

    # Choose thresholds: CLI overrides JSON; else JSON; else config; else defaults
    tL = tS = None
    src = None
    if thresholds_path is not None and os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                js_cli = json.load(f)
            tL, tS, src = _choose_thresholds(js_cli, pref=threshold_source)
            if tL is not None:
                src = f"cli:{src}"
        except Exception as e:
            log.warning(f"Failed to read thresholds from {thresholds_path}: {e}")
    if tL is None:
        tL, tS, src = _choose_thresholds(js, pref=threshold_source)
    if tL is None:
        sig_cfg = cfg.get("signals", {})
        if "long_prob" in sig_cfg and "short_prob" in sig_cfg:
            tL, tS, src = float(sig_cfg["long_prob"]), float(sig_cfg["short_prob"]), "config:signals"
        else:
            tL, tS, src = 0.60, 0.40, "default"

    log.info(
        f"Using thresholds: t_long={tL:.3f}, t_short={tS:.3f} (source={src}, eval_slice={eval_slice})"
    )

    # Subset to train/holdout/all according to optimizer's holdout_frac
    if eval_slice.lower() != "all":
        frac = None
        if js is not None and "holdout_frac" in js:
            try:
                frac = float(js["holdout_frac"])
            except Exception:
                pass
        if frac is None:
            frac = 0.8  # fallback if JSON missing
        n = len(d)
        cut = max(int(n * frac), n - 200)
        cut = min(max(cut, 100), n - 50)
        if eval_slice.lower() == "train":
            d = d.iloc[:cut].copy()
        elif eval_slice.lower() == "holdout":
            d = d.iloc[cut:].copy()
        else:
            raise ValueError("eval_slice must be one of: all|train|holdout")

    # Build actions via thresholding + optional mapping
    d["action"] = _compute_action(d["prob_long"], tL, tS)
    d["action"] = _apply_execution_mapping(d, cfg, d["action"])

    # Determine return column for backtest (prefer ev_ret)
    exec_cfg = cfg.get("execution", {})
    ret_col = exec_cfg.get("ret_col")
    if ret_col is None:
        ret_col = "ev_ret" if "ev_ret" in d.columns else "ret_1"
    if ret_col not in d.columns:
        if ret_col == "ret_1" and "close" in d.columns:
            d[ret_col] = d["close"].shift(-1) / d["close"] - 1.0
        else:
            raise ValueError(
                f"Return column '{ret_col}' not found and cannot be synthesized."
            )

    fee_bps = exec_cfg.get("fee_bps", 0.0)
    spread_bps = exec_cfg.get("spread_bps", 0.0)
    slippage_alpha = exec_cfg.get("slippage_alpha", 0.0)
    adt_window = exec_cfg.get("adt_window", 60)
    cost_margin_mult = exec_cfg.get("cost_margin_mult", 1.0)

    # Choose backtest type based on presence of sigma & mapper
    use_bar = False
    try:
        from ml_signals.backtest.engine import backtest_market  # noqa: F401
        use_bar = True
    except Exception:
        use_bar = False

    if use_bar:
        res = _bar_backtest(
            d,
            ret_col,
            fee_bps,
            spread_bps,
            slippage_alpha,
            adt_window,
            cost_margin_mult,
        )
    else:
        res = _event_backtest(d, ret_col, fee_bps, spread_bps)

    log.info(
        f"[event/all] Backtest done. Total={res['total_return']:+.6f} | "
        f"Trades={res['trades']} | Avg/Trade={res['avg_per_trade']:+.6f} | "
        f"MaxDD={res['max_drawdown']:+.6f}"
    )
    # Save actions and equity
    actions_path = os.path.join(
        cfg["report_dir"], f"backtest_{timeframe}_event_all_actions.csv"
    )
    equity_path = os.path.join(
        cfg["report_dir"], f"backtest_{timeframe}_event_all_equity.png"
    )
    summary_path = os.path.join(
        cfg["report_dir"], f"backtest_{timeframe}_event_all_summary.json"
    )
    d[["timestamp", "prob_long", "action"]].to_csv(actions_path, index=False)
    _plot_equity(np.array(res["equity"], dtype=float), equity_path, "Cumulative return")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log.info(
        f"Artifacts → {actions_path} | {equity_path} | {summary_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtest on trained model outputs")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--timeframe",
        default="1min",
        help="Bar timeframe to use (e.g. 1min, 5min, 240min). Determines which processed data and OOF probabilities are read.",
    )
    parser.add_argument(
        "--thresholds_path",
        default=None,
        help="Optional path to thresholds JSON. Overrides optimiser/config defaults.",
    )
    parser.add_argument(
        "--threshold_source",
        default="auto",
        choices=["auto", "train_best", "best_on_holdout"],
        help="Which thresholds to prefer from optimiser JSON if present.",
    )
    parser.add_argument(
        "--eval_slice",
        default="all",
        choices=["all", "train", "holdout"],
        help="Sub‑set of events to evaluate: all (default), train or holdout", 
    )
    args = parser.parse_args()
    main(
        args.config,
        timeframe=args.timeframe,
        thresholds_path=args.thresholds_path,
        threshold_source=args.threshold_source,
        eval_slice=args.eval_slice,
    )