# src/ml_signals/pipeline/train_oof.py
# -*- coding: utf-8 -*-
"""Nested purged walk‑forward OOF training with options:
- separate splitters (outer=fixed, inner=expanding by default)
- neighbor refinement around best HP
- per-fold calibration (isotonic/Platt)
- robust final-fit even when outer validation is tiny/single-class
"""
from __future__ import annotations
from typing import Dict, List, Optional
from collections import defaultdict
from numbers import Number

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from ..utils.logging import get_logger
from ..validation.purged_walk_forward import purged_walk_forward_splits
from ..metrics.sharpe import weighted_sharpe_ratio
from ..metrics.sortino import weighted_sortino_ratio
from ..metrics.drawdown import max_drawdown

log = get_logger()

META_KEYS = {"hp_max_evals","refine_best","refine_max_evals","early_stopping_rounds","num_boost_round","num_threads","thread_count"}


# ---------------------- grids ----------------------
def _iter_grid(grid_params: Dict, max_evals: Optional[int] = None, seed: int = 42):
    clean: Dict[str, List] = {}
    for k, v in (grid_params or {}).items():
        if k in META_KEYS:
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            clean[k] = list(v)
    # Allow single dict of scalars (final-fit / refine best_hp)
    if not clean:
        if isinstance(grid_params, dict) and any(
            not isinstance(v, (list, tuple, np.ndarray)) for v in (grid_params or {}).values()
        ):
            return [dict(grid_params)]
        return [dict()]
    combos = list(ParameterGrid(clean))
    if max_evals is not None and len(combos) > max_evals:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), size=max_evals, replace=False)]
    return combos


# ---------------------- features/labels ----------------------
def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    import re
    drop_cols_explicit = {"timestamp","y","t_end","tp_pct","sl_pct","sigma","action","pnl","equity","w","ev_ret","ev_ret_short","ev_bps"}
    leak_pat = re.compile(r"(?:^|_)(?:ev|fwd|future|lead|next|target|label|y(?:$|_))|ret_\+|return_\+", re.IGNORECASE)
    cols: List[str] = []
    for c in df.columns:
        if c in drop_cols_explicit:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if leak_pat.search(c):
            continue
        cols.append(c)
    log.info(f"Selected {len(cols)} feature columns (after leakage screen).")
    return cols


def _make_directional_labels(df: pd.DataFrame):
    mask = df["y"].isin([-1, 1])
    y = df.loc[mask, "y"].map({-1: 0, 1: 1}).to_numpy(dtype=int)
    w = df.loc[mask, "w"].fillna(1.0).to_numpy()
    return mask.to_numpy(), y, w


def _map_t_end_to_dir_positions(mask: np.ndarray, t_end_raw: np.ndarray) -> np.ndarray:
    dir_rows = np.flatnonzero(mask)
    t_end_dir = np.searchsorted(dir_rows, t_end_raw, side="right") - 1
    t_end_dir = np.clip(t_end_dir, 0, len(dir_rows) - 1)
    return t_end_dir


def _t_end_relative_to_subset(t_end_dir: np.ndarray, subset_idx: np.ndarray) -> np.ndarray:
    ends_global_for_subset = t_end_dir[subset_idx]
    rel = np.searchsorted(subset_idx, ends_global_for_subset, side="right") - 1
    rel = np.clip(rel, 0, len(subset_idx) - 1)
    return rel


# ---------------------- splitters ----------------------
def _expanding_purged_splits(n: int, t_end: np.ndarray, n_splits: int, min_train: int) -> List[tuple]:
    """Expanding-window CV with purge at val_start."""
    assert n > min_train and n_splits > 0
    val_total = n - min_train
    fold = max(1, val_total // n_splits)
    splits: List[tuple] = []
    for i in range(n_splits):
        val_start = min_train + i * fold
        val_end = n if i == n_splits - 1 else min(n, val_start + fold)
        if val_start >= n or (val_end - val_start) <= 0:
            continue
        tr_idx = np.arange(0, val_start)
        purge_mask = t_end[tr_idx] < val_start
        tr_idx = tr_idx[purge_mask]
        va_idx = np.arange(val_start, val_end)
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        splits.append((tr_idx, va_idx))
    return splits


def _choose_splits(n: int, t_end: np.ndarray, n_splits: int, min_train: int, embargo: int, mode: str) -> List[tuple]:
    mode = (mode or "fixed").lower()
    if mode == "expanding":
        return _expanding_purged_splits(n=n, t_end=t_end, n_splits=n_splits, min_train=min_train)
    else:
        splits, _ = purged_walk_forward_splits(
            n=n, t_end=t_end, n_splits=n_splits, embargo=embargo, final_test_fraction=0.0, min_train=min_train,
        )
        return splits


# ---------------------- neighbor refinement ----------------------
def _neighbors_from_best(best_hp: Dict, base_grid: Dict, max_evals: int = 20) -> Dict:
    rng = np.random.default_rng(42)
    grid: Dict[str, List] = {}
    for k, v in (best_hp or {}).items():
        if isinstance(v, bool) or v is None:
            grid[k] = [v]
            continue
        base_vals = base_grid.get(k)
        lo, hi = None, None
        if isinstance(base_vals, (list, tuple, np.ndarray)) and len(base_vals) > 0 and isinstance(base_vals[0], Number):
            lo, hi = (min(base_vals), max(base_vals))
        if isinstance(v, int):
            cand = sorted(set([v-2, v-1, v, v+1, v+2]))
            cand = [c for c in cand if c >= 1]
            if lo is not None: cand = [c for c in cand if c >= lo]
            if hi is not None: cand = [c for c in cand if c <= hi]
            grid[k] = cand or [v]
        elif isinstance(v, float):
            factors = [0.7, 0.85, 1.0, 1.15, 1.3]
            cand = [float(v * f) for f in factors]
            if k in {"l1_ratio"}:
                cand = [min(1.0, max(0.0, c)) for c in cand]
            if lo is not None: cand = [max(lo, c) for c in cand]
            if hi is not None: cand = [min(hi, c) for c in cand]
            grid[k] = sorted(set(round(c, 6) for c in cand)) or [v]
        else:
            grid[k] = [v]
    combos = list(ParameterGrid(grid)) if grid else [{}]
    if len(combos) > max_evals:
        idx = rng.choice(len(combos), size=max_evals, replace=False)
        sel = [combos[i] for i in idx]
        grid = {k: sorted(set(d.get(k, best_hp.get(k)) for d in sel)) for k in grid.keys()}
    return grid


# ---------------------- model trainers ----------------------
def _train_logit(Xtr, ytr, wtr, Xval, yval, wval, grid_params: Dict, seed: int):
    best_model, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid_params, grid_params.get("hp_max_evals"), seed):
        hp = dict(hp or {})
        hp.setdefault("C", 1.0)
        hp.setdefault("l1_ratio", 0.5)
        lr = LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=10000, random_state=seed, n_jobs=1, **hp
        )
        lr.fit(Xtr, ytr, sample_weight=wtr)
        pv = lr.predict_proba(Xval)[:, 1]
        try:
            score = roc_auc_score(yval, pv, sample_weight=wval)
        except Exception:
            score = 0.5
        if score > best_score:
            best_model, best_score, best_hp = lr, score, hp
    return best_model, best_score, (best_hp or {})


def _train_lgbm(Xtr, ytr, wtr, Xval, yval, wval, grid_params: Dict, seed: int, feature_names: List[str]):
    best_model, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid_params, grid_params.get("hp_max_evals"), seed):
        params = {"objective":"binary","metric":"auc","boosting_type":"gbdt","verbosity":-1,"random_state":seed}
        params.update(hp)
        lgtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feature_names)
        lgvalid = lgb.Dataset(Xval, label=yval, weight=wval, reference=lgtrain)
        try:
            model = lgb.train(
                params, lgtrain,
                valid_sets=[lgtrain, lgvalid], valid_names=["train","valid"],
                num_boost_round=grid_params.get("num_boost_round", 5000),
                early_stopping_rounds=grid_params.get("early_stopping_rounds", 200),
                verbose_eval=False,
            )
            pv = model.predict(Xval, num_iteration=model.best_iteration)
            score = roc_auc_score(yval, pv, sample_weight=wval)
        except Exception:
            model, score = None, 0.5
        if score > best_score:
            best_model, best_score, best_hp = model, score, hp

    imps_df = None
    if best_model is not None:
        try:
            gain = best_model.feature_importance(importance_type="gain")
            split = best_model.feature_importance(importance_type="split")
            imps_df = pd.DataFrame({"feature": feature_names, "gain": gain, "split": split})
        except Exception:
            imps_df = None
    return best_model, imps_df, (best_hp or {})


def _train_catboost(Xtr, ytr, wtr, Xval, yval, wval, grid_params: Dict, seed: int):
    best_model, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid_params, grid_params.get("hp_max_evals"), seed):
        params = {"loss_function":"Logloss","eval_metric":"AUC","random_seed":seed,"verbose":False}
        params.update(hp)
        try:
            train_pool = Pool(Xtr, label=ytr, weight=wtr)
            valid_pool = Pool(Xval, label=yval, weight=wval)
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, verbose=False,
                      early_stopping_rounds=grid_params.get("early_stopping_rounds", 200))
            pv = model.predict_proba(Xval)[:, 1]
            score = roc_auc_score(yval, pv, sample_weight=wval)
        except Exception:
            model, score = None, 0.5
        if score > best_score:
            best_model, best_score, best_hp = model, score, hp
    return best_model, None, (best_hp or {})


# Fallback final-fit (no eval) when outer val is tiny/single-class

def _fit_lgbm_no_eval(X, y, w, params: dict, seed: int, feature_names: list):
    params_ = {"objective": "binary", "metric": "None", "boosting_type": "gbdt", "verbosity": -1, "random_state": seed}
    params_.update(params or {})
    dtrain = lgb.Dataset(X, label=y, weight=w, feature_name=feature_names)
    model = lgb.train(params_, dtrain, num_boost_round=int(params.get("num_boost_round", 500)))
    imps_df = None
    try:
        gain = model.feature_importance(importance_type="gain")
        split = model.feature_importance(importance_type="split")
        imps_df = pd.DataFrame({"feature": feature_names, "gain": gain, "split": split})
    except Exception:
        pass
    return model, imps_df


def _fit_catboost_no_eval(X, y, w, params: dict, seed: int):
    params_ = {"loss_function": "Logloss", "random_seed": seed, "verbose": False}
    params_.update(params or {})
    params_.pop("early_stopping_rounds", None)
    model = CatBoostClassifier(**params_)
    model.fit(Pool(X, label=y, weight=w), verbose=False)
    return model, None


# ---------------------- feature selection utils ----------------------
def _correlation_prune(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in df.columns if c not in drop_cols]


def _stability_select(feature_importances: pd.DataFrame, top_n: int = 20) -> List[str]:
    if feature_importances.empty or "gain" not in feature_importances.columns:
        return []
    agg = feature_importances.groupby("feature")["gain"].mean().sort_values(ascending=False)
    return list(agg.head(top_n).index)


# ---------------------- main entry ----------------------
def train_oof_models(
    df: pd.DataFrame,
    cfg: Dict,
    grid_logit: Dict,
    grid_lgbm: Dict,
    grid_catb: Dict,
    fee_bps: float,
    spread_bps: float,
    embargo: int,
):
    val_cfg = cfg.get("validation", {})
    debug = bool(val_cfg.get("debug_splits", False))
    # Separate splitters; default to (outer=fixed, inner=expanding) if neither provided
    split_outer = val_cfg.get("splitter_outer")
    split_inner = val_cfg.get("splitter_inner")
    if split_outer is None and split_inner is None:
        split_outer, split_inner = "fixed", "expanding"
    else:
        split_outer = str(split_outer or val_cfg.get("splitter", "fixed")).lower()
        split_inner = str(split_inner or split_outer).lower()
    min_val_size = int(val_cfg.get("min_val_size", 50))

    # calibration config
    cal_cfg = cfg.get("calibration", {})
    cal_enable = bool(cal_cfg.get("enable", False))
    cal_method = str(cal_cfg.get("method", "isotonic")).lower()
    cal_min = int(cal_cfg.get("min_samples", 500))

    mask, y_dir, w = _make_directional_labels(df)
    t_end_raw = df.loc[mask, "t_end"].astype(int).to_numpy()
    t_end_dir = _map_t_end_to_dir_positions(mask, t_end_raw)

    feat_cols_all = _select_feature_cols(df)
    feat_cols = _correlation_prune(df[feat_cols_all], threshold=0.95)

    outer_splits = _choose_splits(
        n=int(mask.sum()), t_end=t_end_dir,
        n_splits=val_cfg.get("outer_folds", 5),
        min_train=val_cfg.get("min_train_size", 1000),
        embargo=embargo, mode=split_outer,
    )
    if debug:
        print(f"[Debug] Splitters: outer={split_outer}, inner={split_inner}; min_val_size={min_val_size}")
    if debug:
        print(f"[Debug] Outer splits: {len(outer_splits)} splits")
        for i, (tr, va) in enumerate(outer_splits):
            print(f"  [{i+1}/{len(outer_splits)}] train={len(tr)} val={len(va)}")

    n_dir = int(mask.sum())
    oof = np.full(n_dir, np.nan)
    oof_raw = np.full(n_dir, np.nan)
    fold_ids = np.full(n_dir, -1, dtype=int)
    all_imps: List[pd.DataFrame] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer_splits):
        print(f"[Outer Fold {fold_idx+1}/{len(outer_splits)}]")

        t_end_tr_rel = _t_end_relative_to_subset(t_end_dir, tr_idx)
        inner_splits = _choose_splits(
            n=len(tr_idx), t_end=t_end_tr_rel,
            n_splits=val_cfg.get("inner_folds", 3),
            min_train=max(10, int(len(tr_idx) * 0.3)),
            embargo=embargo, mode=split_inner,
        )
        if len(inner_splits) == 0:
            n = len(tr_idx)
            n_tr = max(10, int(n * 0.8))
            inner_splits = [(np.arange(0, n_tr), np.arange(n_tr, n))]
            if debug:
                print(f"[Outer {fold_idx}] Inner purged splits empty → fallback 80/20.")
        if debug:
            print(f"[Debug] Inner splits (outer={fold_idx}): {len(inner_splits)} splits")
            for j, (itr, iva) in enumerate(inner_splits):
                print(f"  [{j+1}/{len(inner_splits)}] train={len(itr)} val={len(iva)}")

        hp_results = []
        calib_pool = defaultdict(lambda: {"p": [], "y": [], "w": []})
        imps_accum: List[pd.DataFrame] = []

        for (itr, iva) in inner_splits:
            Xtr_i = df.loc[mask].iloc[tr_idx][feat_cols].to_numpy()[itr]
            Xva_i = df.loc[mask].iloc[tr_idx][feat_cols].to_numpy()[iva]
            Xtr_i = np.nan_to_num(Xtr_i, nan=0.0, posinf=0.0, neginf=0.0)
            Xva_i = np.nan_to_num(Xva_i, nan=0.0, posinf=0.0, neginf=0.0)
            ytr_i = y_dir[tr_idx][itr]
            yva_i = y_dir[tr_idx][iva]
            wtr_i = w[tr_idx][itr]
            wva_i = w[tr_idx][iva]

            scaler = StandardScaler().fit(Xtr_i)
            Xtr_is = scaler.transform(Xtr_i)
            Xva_is = scaler.transform(Xva_i)

            for name, fn, Xtr_use, Xva_use, grid in [
                ("logit", _train_logit, Xtr_is, Xva_is, grid_logit),
                ("lgbm", _train_lgbm, Xtr_i, Xva_i, grid_lgbm),
                ("catboost", _train_catboost, Xtr_i, Xva_i, grid_catb),
            ]:
                if name == "lgbm":
                    model, imps, hp = fn(Xtr_use, ytr_i, wtr_i, Xva_use, yva_i, wva_i, grid, cfg["seed"], feature_names=feat_cols)
                else:
                    model, imps, hp = fn(Xtr_use, ytr_i, wtr_i, Xva_use, yva_i, wva_i, grid, cfg["seed"])
                if model is None:
                    continue

                pv = model.predict_proba(Xva_use)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xva_use)
                calib_pool[name]["p"].append(pv)
                calib_pool[name]["y"].append(yva_i)
                calib_pool[name]["w"].append(wva_i)

                ev_long = df.loc[mask, "ev_ret"].to_numpy()[tr_idx][iva]
                ev_short = df.loc[mask, "ev_ret_short"].to_numpy()[tr_idx][iva]
                cost = (fee_bps + spread_bps) / 1e4
                rets = np.where(pv >= 0.5, ev_long, ev_short) - cost

                hp_results.append({
                    "model": name,
                    "sharpe": weighted_sharpe_ratio(rets, wva_i),
                    "sortino": weighted_sortino_ratio(rets, wva_i),
                    "max_dd": max_drawdown(rets),
                    "hp": hp,
                })
                if isinstance(imps, pd.DataFrame) and not imps.empty:
                    imps_accum.append(imps)

        if hp_results:
            filt = [r for r in hp_results if np.isfinite(r["sharpe"]) and r["sortino"] > 0]
            candidate = max((filt or hp_results), key=lambda r: r["sharpe"])
            best_name, best_hp = candidate["model"], candidate["hp"]
        else:
            best_name, best_hp = "logit", {}
        print(f"[Outer {fold_idx}] Best inner-CV model: {best_name} HP: {best_hp}")

        # optional neighbor refinement
        model_cfg = cfg.get("models", {}).get("lgbm" if best_name=="lgbm" else ("catb" if best_name=="catboost" else "logit"), {})
        if bool(model_cfg.get("refine_best", False)):
            refine_grid = _neighbors_from_best(best_hp, model_cfg, int(model_cfg.get("refine_max_evals", 20)))
            if refine_grid:
                refined = []
                for (itr, iva) in inner_splits:
                    Xtr_i = df.loc[mask].iloc[tr_idx][feat_cols].to_numpy()[itr]
                    Xva_i = df.loc[mask].iloc[tr_idx][feat_cols].to_numpy()[iva]
                    Xtr_i = np.nan_to_num(Xtr_i, nan=0.0, posinf=0.0, neginf=0.0)
                    Xva_i = np.nan_to_num(Xva_i, nan=0.0, posinf=0.0, neginf=0.0)
                    ytr_i = y_dir[tr_idx][itr]; yva_i = y_dir[tr_idx][iva]
                    wtr_i = w[tr_idx][itr]; wva_i = w[tr_idx][iva]
                    if best_name == "logit":
                        scaler = StandardScaler().fit(Xtr_i)
                        Xtr_use = scaler.transform(Xtr_i); Xva_use = scaler.transform(Xva_i)
                        for hp in _iter_grid(refine_grid, refine_grid.get("hp_max_evals")):
                            model, _, _ = _train_logit(Xtr_use, ytr_i, wtr_i, Xva_use, yva_i, wva_i, hp, cfg["seed"])
                            pv = model.predict_proba(Xva_use)[:,1]
                            ev_long = df.loc[mask, "ev_ret"].to_numpy()[tr_idx][iva]
                            ev_short = df.loc[mask, "ev_ret_short"].to_numpy()[tr_idx][iva]
                            cost = (fee_bps + spread_bps) / 1e4
                            rets = np.where(pv >= 0.5, ev_long, ev_short) - cost
                            refined.append((weighted_sharpe_ratio(rets, wva_i), hp))
                    elif best_name == "lgbm":
                        for hp in _iter_grid(refine_grid, refine_grid.get("hp_max_evals")):
                            model, _, _ = _train_lgbm(Xtr_i, ytr_i, wtr_i, Xva_i, yva_i, wva_i, hp, cfg["seed"], feature_names=feat_cols)
                            pv = model.predict(Xva_i)
                            ev_long = df.loc[mask, "ev_ret"].to_numpy()[tr_idx][iva]
                            ev_short = df.loc[mask, "ev_ret_short"].to_numpy()[tr_idx][iva]
                            cost = (fee_bps + spread_bps) / 1e4
                            rets = np.where(pv >= 0.5, ev_long, ev_short) - cost
                            refined.append((weighted_sharpe_ratio(rets, wva_i), hp))
                    else:  # catboost
                        for hp in _iter_grid(refine_grid, refine_grid.get("hp_max_evals")):
                            model, _, _ = _train_catboost(Xtr_i, ytr_i, wtr_i, Xva_i, yva_i, wva_i, hp, cfg["seed"]) 
                            pv = model.predict_proba(Xva_i)[:,1]
                            ev_long = df.loc[mask, "ev_ret"].to_numpy()[tr_idx][iva]
                            ev_short = df.loc[mask, "ev_ret_short"].to_numpy()[tr_idx][iva]
                            cost = (fee_bps + spread_bps) / 1e4
                            rets = np.where(pv >= 0.5, ev_long, ev_short) - cost
                            refined.append((weighted_sharpe_ratio(rets, wva_i), hp))
                if refined:
                    best_hp_ref = max(refined, key=lambda t: t[0])[1]
                    if best_hp_ref != best_hp:
                        print(f"[Outer {fold_idx}] Refine improved HP: {best_hp} → {best_hp_ref}")
                        best_hp = best_hp_ref

        # build calibrator from inner predictions of chosen model
        calibrator = None
        cal_bucket = calib_pool.get(best_name)
        if cal_enable and cal_bucket and len(cal_bucket["p"]) > 0:
            p_cal = np.concatenate(cal_bucket["p"]) ; y_cal = np.concatenate(cal_bucket["y"]) ; w_cal = np.concatenate(cal_bucket["w"])
            if np.isfinite(p_cal).all() and np.nanstd(p_cal) > 1e-6 and p_cal.size >= cal_min:
                if cal_method == "isotonic":
                    ir = IsotonicRegression(out_of_bounds="clip")
                    ir.fit(p_cal, y_cal, sample_weight=w_cal)
                    calibrator = lambda z: np.clip(ir.predict(z), 0.0, 1.0)
                elif cal_method == "platt":
                    from sklearn.linear_model import LogisticRegression as _LR
                    def _logit(a):
                        a = np.clip(a, 1e-6, 1-1e-6)
                        return np.log(a/(1-a))
                    X = _logit(p_cal).reshape(-1, 1)
                    lr = _LR(solver="lbfgs", C=1e6, max_iter=1000)
                    lr.fit(X, y_cal, sample_weight=w_cal)
                    calibrator = lambda z: lr.predict_proba(_logit(np.clip(z,1e-6,1-1e-6)).reshape(-1,1))[:,1]
            elif debug:
                print(f"[Outer {fold_idx}] Calibration skipped (insufficient/degenerate data)")

        imps_concat = pd.concat(imps_accum, ignore_index=True) if imps_accum else pd.DataFrame()
        selected = _stability_select(imps_concat, top_n=cfg.get("features", {}).get("stability_top_n", 20))
        feat_cols_fold = [f for f in feat_cols if f in selected] if selected else feat_cols

        # final fit per outer fold
        Xtr_f = df.loc[mask].iloc[tr_idx][feat_cols_fold].to_numpy()
        Xva_f = df.loc[mask].iloc[va_idx][feat_cols_fold].to_numpy()
        Xtr_f = np.nan_to_num(Xtr_f, nan=0.0, posinf=0.0, neginf=0.0)
        Xva_f = np.nan_to_num(Xva_f, nan=0.0, posinf=0.0, neginf=0.0)
        ytr_f = y_dir[tr_idx]; wtr_f = w[tr_idx]
        yva_f = y_dir[va_idx]; wva_f = w[va_idx]

        if best_name == "logit":
            scaler = StandardScaler().fit(Xtr_f)
            Xtr_f = scaler.transform(Xtr_f)
            Xva_f = scaler.transform(Xva_f)

        valid_ok = (len(np.unique(yva_f)) >= 2) and (len(yva_f) >= min_val_size)
        if not valid_ok:
            classes = np.unique(yva_f).tolist()
            log.warning(f"[Outer {fold_idx}] Using no-eval final fit: model={best_name}, n_val={len(yva_f)}, classes={classes}, min_val_size={min_val_size}")
        if best_name == "lgbm":
            if valid_ok:
                model_f, imps_f, _ = _train_lgbm(Xtr_f, ytr_f, wtr_f, Xva_f, yva_f, wva_f, best_hp, cfg["seed"], feature_names=feat_cols_fold)
            else:
                model_f, imps_f = _fit_lgbm_no_eval(Xtr_f, ytr_f, wtr_f, best_hp, cfg["seed"], feature_names=feat_cols_fold)
        elif best_name == "catboost":
            if valid_ok:
                model_f, imps_f, _ = _train_catboost(Xtr_f, ytr_f, wtr_f, Xva_f, yva_f, wva_f, best_hp, cfg["seed"])
            else:
                model_f, imps_f = _fit_catboost_no_eval(Xtr_f, ytr_f, wtr_f, best_hp, cfg["seed"])
        else:
            model_f, imps_f, _ = _train_logit(Xtr_f, ytr_f, wtr_f, Xva_f, yva_f, wva_f, best_hp, cfg["seed"])

        if isinstance(imps_f, pd.DataFrame) and not imps_f.empty:
            all_imps.append(imps_f)

        preds_raw = model_f.predict_proba(Xva_f)[:, 1] if hasattr(model_f, "predict_proba") else model_f.predict(Xva_f)
        preds = calibrator(preds_raw) if calibrator is not None else preds_raw
        oof_raw[va_idx] = preds_raw
        oof[va_idx] = preds
        fold_ids[va_idx] = fold_idx

    ts_dir = df.loc[mask, "timestamp"].reset_index(drop=True)
    oof_df = pd.DataFrame({
        "timestamp": ts_dir,
        "prob_long": oof,
        "prob_long_raw": oof_raw,
        "fold": fold_ids,
    })

    imps_df = pd.concat(all_imps, ignore_index=True) if all_imps else pd.DataFrame()
    return oof_df, imps_df
