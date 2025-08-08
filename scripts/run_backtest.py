import argparse, os, json, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_signals.utils.config import load_config
from ml_signals.utils.logging import get_logger

log = get_logger()

# -------------------- helpers --------------------

def _to_utc_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())

def _plot_equity(eq: np.ndarray, path_png: str, title: str):
    if eq.size == 0:
        return
    plt.figure(figsize=(8, 3.5))
    plt.plot(eq, lw=1.5)
    plt.xlabel("Events/Bars (chronological)")
    plt.ylabel("Cumulative return")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path_png, dpi=120); plt.close()

def _read_optimizer_json(report_dir: str, timeframe: str):
    """
    Load thresholds + holdout_frac from optimizer JSON.
    Supports both old (train/best) and new (train_best, best_on_holdout_among_top_train) schemas.
    Returns dict or None.
    """
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

def _choose_thresholds(js: dict, pref: str = "auto"):
    """
    Choose thresholds from optimizer JSON:
      pref='train_best' → use train_best
      pref='best_on_holdout' → use best_on_holdout_among_top_train (if present)
      pref='auto' → prefer train_best, else best_on_holdout, else legacy train/best
    Returns (tL, tS, source_str) or (None,None,None)
    """
    if js is None:
        return None, None, None

    def _has(obj, *keys):
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
        tr = js["train"]; return float(tr["t_long"]), float(tr["t_short"]), "optimizer:train"
    if _has(js.get("best", {}), "t_long", "t_short"):
        be = js["best"];  return float(be["t_long"]), float(be["t_short"]), "optimizer:best"

    return None, None, None

def _compute_action(prob: pd.Series, t_long: float, t_short: float) -> pd.Series:
    p = prob.values.astype(float)
    a = np.zeros_like(p, dtype=int)
    a[p >= t_long] =  1
    a[p <= t_short] = -1
    return pd.Series(a, index=prob.index, name="action")

def _combine_actions(base: np.ndarray, mapped: np.ndarray, mode: str = "intersect") -> np.ndarray:
    base = base.astype(int); mapped = mapped.astype(int)
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

def _bar_backtest(df: pd.DataFrame, ret_col: str, fee_bps: float, spread_bps: float,
                  slippage_alpha: float, adt_window: int, cost_margin_mult: float) -> dict:
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
    a_prev = np.roll(a, 1); a_prev[0] = 0
    entry_change = (a != a_prev).astype(float) & (a != 0)
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

def _build_hysteresis(cfg: dict) -> dict:
    hyst_cfg = cfg.get("execution", {}).get("hysteresis", {})
    if not isinstance(hyst_cfg, dict):
        hyst_cfg = {"persistence_bars": 2, "cooldown_minutes": int(hyst_cfg) if hyst_cfg else 15}
    return {
        "persistence_bars": int(hyst_cfg.get("persistence_bars", 2)),
        "cooldown_minutes": int(hyst_cfg.get("cooldown_minutes", 15)),
    }

def _apply_execution_mapping(df: pd.DataFrame, cfg: dict, base_action: pd.Series) -> pd.Series:
    combine_mode = cfg.get("execution", {}).get("mapper_combine", "intersect")
    if "sigma" not in df.columns:
        return base_action
    try:
        from ml_signals.execution.ev_mapping import map_probs_to_actions
        fee    = float(cfg.get("execution", {}).get("fee_bps", 0.0))
        spread = float(cfg.get("execution", {}).get("spread_bps", 0.0))
        margin = float(cfg.get("execution", {}).get("cost_margin_mult", 1.0))
        k      = float(cfg.get("labeling", {}).get("k_sigma", [2.0])[0])
        hyst   = _build_hysteresis(cfg)
        liq_gate  = cfg.get("execution", {}).get("liquidity_gate", None)
        part_caps = cfg.get("execution", {}).get("participation_caps", None)

        df_map = map_probs_to_actions(
            df.copy(), "prob_long", "sigma", k, fee, spread, margin, hyst,
            liquidity_gate=liq_gate, participation_caps=part_caps
        )  # returns columns: action_raw, action, ev_bps
        mapped = df_map["action"].values.astype(int)
        combined = _combine_actions(base_action.values.astype(int), mapped, mode=combine_mode)
        return pd.Series(combined, index=df.index, name="action")
    except Exception as e:
        log.warning(f"Execution mapping failed: {e}. Using thresholds only.")
        return base_action

# -------------------- main --------------------

def main(cfg_path: str, timeframe: str = "1min",
         thresholds_path: str = None,
         threshold_source: str = "auto",
         eval_slice: str = "all"):
    """
    threshold_source: auto|train_best|best_on_holdout
    eval_slice: all|train|holdout
    """
    cfg = load_config(cfg_path)

    sym = cfg["data"]["symbol"]
    proc_path  = os.path.join(cfg["data"]["processed_dir"], f"{sym}_{timeframe}_processed.csv")
    oof_path   = os.path.join(cfg["artifact_dir"], "oof_probs.csv")
    report_dir = cfg["report_dir"]

    # Load
    df  = pd.read_csv(proc_path); df["timestamp"] = _to_utc_ts(df["timestamp"])
    oof = pd.read_csv(oof_path);  oof["timestamp"] = _to_utc_ts(oof["timestamp"])
    oof["prob_long"] = pd.to_numeric(oof["prob_long"], errors="coerce")

    # Merge
    d = df.merge(oof[["timestamp", "prob_long"]], on="timestamp", how="inner")
    d = d.dropna(subset=["timestamp", "prob_long"]).sort_values("timestamp").reset_index(drop=True)
    if len(d) == 0:
        raise ValueError("No rows after merging processed data with OOF probabilities.")

    # Load optimizer JSON (for thresholds and holdout_frac)
    js = _read_optimizer_json(report_dir, timeframe)

    # Choose thresholds: CLI overrides JSON; else config; else defaults
    tL, tS, src = (None, None, None)
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
        sig = cfg.get("signals", {})
        if "long_prob" in sig and "short_prob" in sig:
            tL, tS, src = float(sig["long_prob"]), float(sig["short_prob"]), "config:signals"
        else:
            tL, tS, src = 0.60, 0.40, "default"

    log.info(f"Using thresholds: t_long={tL:.3f}, t_short={tS:.3f} (source={src}, eval_slice={eval_slice})")

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

    # Build actions
    d["action"] = _compute_action(d["prob_long"], tL, tS)
    d["action"] = _apply_execution_mapping(d, cfg, d["action"])

    # Return column (prefer ev_ret)
    exec_cfg = cfg.get("execution", {})
    ret_col = exec_cfg.get("ret_col", None)
    if ret_col is None:
        ret_col = "ev_ret" if "ev_ret" in d.columns else "ret_1"
    if ret_col not in d.columns:
        if ret_col == "ret_1" and "close" in d.columns:
            d[ret_col] = d["close"].shift(-1) / d["close"] - 1.0
        else:
            raise ValueError(f"Return column '{ret_col}' not found and cannot be synthesized.")

    # Costs / knobs
    fee_bps    = exec_cfg.get("fee_bps", 0.0)
    spread_bps = exec_cfg.get("spread_bps", 0.0)
    slippage_alpha   = exec_cfg.get("slippage_alpha", 0.5)
    adt_window       = exec_cfg.get("adt_window", 60)
    cost_margin_mult = exec_cfg.get("cost_margin_mult", 1.0)

    # Backtest mode
    if ret_col.startswith("ev_"):
        mode = "event"
        res = _event_backtest(d, ret_col, fee_bps, spread_bps)
    else:
        mode = "bar"
        res = _bar_backtest(d, ret_col, fee_bps, spread_bps, slippage_alpha, adt_window, cost_margin_mult)

    # Outputs
    os.makedirs(report_dir, exist_ok=True)
    tag = f"{timeframe}_{mode}_{eval_slice}"
    base = os.path.join(report_dir, f"backtest_{tag}")

    eq = np.array(res.get("equity", []), dtype=float)
    _plot_equity(eq, f"{base}_equity.png", title=f"Equity ({mode}/{eval_slice}) | tL={tL:.3f}, tS={tS:.3f}")

    out_csv = f"{base}_actions.csv"
    d_out = d[["timestamp", "prob_long", "action"]].copy()
    if ret_col in d.columns:
        d_out[ret_col] = d[ret_col]
    d_out.to_csv(out_csv, index=False)

    summary = {
        "mode": mode,
        "symbol": cfg["data"]["symbol"],
        "timeframe": timeframe,
        "eval_slice": eval_slice,
        "thresholds": {"t_long": tL, "t_short": tS, "source": src},
        "ret_col": ret_col,
        "fees": {"fee_bps": float(fee_bps), "spread_bps": float(spread_bps)},
        "metrics": {
            "total_return": float(res.get("total_return", 0.0)),
            "trades": int(res.get("trades", 0)),
            "avg_per_trade": float(res.get("avg_per_trade", 0.0)),
            "max_drawdown": float(res.get("max_drawdown", 0.0)),
        },
        "artifacts": {
            "equity_png": f"{base}_equity.png",
            "actions_csv": out_csv,
        }
    }
    with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info(f"[{mode}/{eval_slice}] Backtest done. Total={summary['metrics']['total_return']:.6f} | "
             f"Trades={summary['metrics']['trades']} | Avg/Trade={summary['metrics']['avg_per_trade']:.6f} | "
             f"MaxDD={summary['metrics']['max_drawdown']:.6f}")
    log.info(f"Artifacts → {out_csv} | {base}_equity.png | {base}_summary.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--timeframe", default="1min")
    ap.add_argument("--thresholds", default=None, help="Optional path to thresholds JSON.")
    ap.add_argument("--threshold-source", default="auto", choices=["auto","train_best","best_on_holdout"])
    ap.add_argument("--eval-slice", default="all", choices=["all","train","holdout"])
    args = ap.parse_args()
    main(args.config, timeframe=args.timeframe,
         thresholds_path=args.thresholds,
         threshold_source=args.threshold_source,
         eval_slice=args.eval_slice)
