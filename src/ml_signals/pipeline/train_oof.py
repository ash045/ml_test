# src/ml_signals/pipeline/train_oof.py
# -*- coding: utf-8 -*-
"""Nested purged walk‑forward OOF training (robust importances & proba).

Fixes in this patch:
- Prevent TypeError when concatenating feature importances (guard types).
- Make LightGBM trainer return a DataFrame of importances (gain/split).
- Use probabilities for all models: hasattr(..., 'predict_proba') → else predict().
- Stability selection now skips gracefully when no valid importances exist.
- Map `t_end` → directional indices and inner-subset indices to avoid empty inner folds.
- Optional `validation.debug_splits` for split-size prints.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from ..utils.logging import get_logger
from ..validation.purged_walk_forward import purged_walk_forward_splits
from ..metrics.sharpe import weighted_sharpe_ratio
from ..metrics.sortino import weighted_sortino_ratio
from ..metrics.drawdown import max_drawdown

log = get_logger()

META_KEYS = {
    "hp_max_evals",
    "refine_best",
    "refine_max_evals",
    "early_stopping_rounds",
    "num_boost_round",
    "num_threads",
    "thread_count",
}


def _iter_grid(grid_params: Dict, max_evals: Optional[int] = None, seed: int = 42):
    clean: Dict[str, List] = {}
    for k, v in (grid_params or {}).items():
        if k in META_KEYS:
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            clean[k] = list(v)
    combos = list(ParameterGrid(clean)) if clean else [dict()]
    if max_evals is not None and len(combos) > max_evals:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), size=max_evals, replace=False)]
    return combos


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    import re
    drop_cols_explicit = {
        "timestamp",
        "y",
        "t_end",
        "tp_pct",
        "sl_pct",
        "sigma",
        "action",
        "pnl",
        "equity",
        "w",
        "ev_ret",
        "ev_ret_short",
        "ev_bps",
    }
    leak_pat = re.compile(r"(?:^|_)(?:ev|fwd|future|lead|next|target|label|y(?:$|_))|ret_\+|return_\+",
                          re.IGNORECASE)
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


# --- Debug helpers -----------------------------------------------------------

def _dbg(msg: str, enable: bool):
    if enable:
        print(msg)


def _summarize_splits(splits, label: str, enable: bool):
    if not enable:
        return
    total = len(splits)
    _dbg(f"{label}: {total} splits", enable)
    for i, (tr, va) in enumerate(splits):
        _dbg(f"  [{i+1}/{total}] train={len(tr)} val={len(va)}", enable)


# --- Model trainers ----------------------------------------------------------

def _train_logit(Xtr, ytr, wtr, Xval, yval, wval, grid_params: Dict, seed: int):
    best_model, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid_params, grid_params.get("hp_max_evals"), seed):
        # Ensure elasticnet has valid hyperparameters; fallback keeps training robust.
        hp = dict(hp or {})
        hp.setdefault("C", 1.0)
        hp.setdefault("l1_ratio", 0.5)
        lr = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=seed,
            n_jobs=1,
            **hp,
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


def _train_lgbm(
    Xtr, ytr, wtr, Xval, yval, wval, grid_params: Dict, seed: int, feature_names: List[str]
):
    best_model, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid_params, grid_params.get("hp_max_evals"), seed):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": seed,
        }
        params.update(hp)
        lgtrain = lgb.Dataset(Xtr, label=ytr, weight=wtr, feature_name=feature_names)
        lgvalid = lgb.Dataset(Xval, label=yval, weight=wval, reference=lgtrain)
        try:
            model = lgb.train(
                params,
                lgtrain,
                valid_sets=[lgtrain, lgvalid],
                valid_names=["train", "valid"],
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
            imps_df = pd.DataFrame({
                "feature": feature_names,
                "gain": gain,
                "split": split,
            })
        except Exception:
            imps_df = None
    return best_model, imps_df, (best_hp or {})


def _train_catboost(Xtr, ytr, wtr, Xval, yval, wval, grid_params: Dict, seed: int):
    best_model, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid_params, grid_params.get("hp_max_evals"), seed):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": seed,
            "verbose": False,
        }
        params.update(hp)
        try:
            train_pool = Pool(Xtr, label=ytr, weight=wtr)
            valid_pool = Pool(Xval, label=yval, weight=wval)
            model = CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                verbose=False,
                early_stopping_rounds=grid_params.get("early_stopping_rounds", 200),
            )
            pv = model.predict_proba(Xval)[:, 1]
            score = roc_auc_score(yval, pv, sample_weight=wval)
        except Exception:
            model, score = None, 0.5
        if score > best_score:
            best_model, best_score, best_hp = model, score, hp
    return best_model, None, (best_hp or {})


def _correlation_prune(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in df.columns if c not in drop_cols]


def _stability_select(feature_importances: pd.DataFrame, top_n: int = 20) -> List[str]:
    if feature_importances.empty:
        return []
    if "gain" not in feature_importances.columns:
        return []
    agg = (
        feature_importances.groupby("feature")["gain"].mean().sort_values(ascending=False)
    )
    return agg.head(top_n).index.tolist()


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
    debug = bool(cfg.get("validation", {}).get("debug_splits", False))

    mask, y_dir, w = _make_directional_labels(df)

    # map t_end to directional index space (not raw df rows)
    t_end_raw = df.loc[mask, "t_end"].astype(int).to_numpy()
    t_end_dir = _map_t_end_to_dir_positions(mask, t_end_raw)

    feat_cols = _correlation_prune(df[_select_feature_cols(df)], threshold=0.95)

    outer_splits, _ = purged_walk_forward_splits(
        n=int(mask.sum()),
        t_end=t_end_dir,
        n_splits=cfg["validation"].get("outer_folds", 5),
        embargo=embargo,
        final_test_fraction=0.0,
        min_train=cfg["validation"].get("min_train_size", 1000),
    )
    _summarize_splits(outer_splits, label="[Debug] Outer splits", enable=debug)

    oof = np.full(int(mask.sum()), np.nan, dtype=float)
    all_imps: List[pd.DataFrame] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer_splits):
        print(f"[Outer Fold {fold_idx+1}/{len(outer_splits)}]")

        # inner splits on subset-relative indices
        t_end_tr_rel = _t_end_relative_to_subset(t_end_dir, tr_idx)
        inner_splits, _ = purged_walk_forward_splits(
            n=len(tr_idx),
            t_end=t_end_tr_rel,
            n_splits=cfg["validation"].get("inner_folds", 3),
            embargo=embargo,
            final_test_fraction=0.0,
            min_train=max(10, int(len(tr_idx) * 0.3)),
        )
        if len(inner_splits) == 0:
            n = len(tr_idx)
            n_tr = max(10, int(n * 0.8))
            inner_splits = [(np.arange(0, n_tr), np.arange(n_tr, n))]
            _dbg(f"[Outer {fold_idx}] Inner purged splits empty → fallback 80/20.", debug)
        _summarize_splits(inner_splits, label=f"[Debug] Inner splits (outer={fold_idx})", enable=debug)

        hp_results = []
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
                    model, imps, hp = fn(
                        Xtr_use, ytr_i, wtr_i, Xva_use, yva_i, wva_i, grid, cfg["seed"], feature_names=feat_cols
                    )
                else:
                    model, imps, hp = fn(
                        Xtr_use, ytr_i, wtr_i, Xva_use, yva_i, wva_i, grid, cfg["seed"]
                    )
                if model is None:
                    continue

                # probabilities consistently
                if hasattr(model, "predict_proba"):
                    pv = model.predict_proba(Xva_use)[:, 1]
                else:
                    pv = model.predict(Xva_use)

                ev_long = df.loc[mask, "ev_ret"].to_numpy()[tr_idx][iva]
                ev_short = df.loc[mask, "ev_ret_short"].to_numpy()[tr_idx][iva]
                cost = (fee_bps + spread_bps) / 1e-4 / 10000  # avoid float slip; equal to /1e4
                cost = (fee_bps + spread_bps) / 1e4
                rets = np.where(pv >= 0.5, ev_long, ev_short) - cost

                hp_results.append({
                    "model": name,
                    "sharpe": weighted_sharpe_ratio(rets, wva_i),
                    "sortino": weighted_sortino_ratio(rets, wva_i),
                    "max_dd": max_drawdown(rets),
                    "hp": hp,
                })
                # only keep DataFrame importances
                if isinstance(imps, pd.DataFrame) and not imps.empty:
                    imps_accum.append(imps)

        if hp_results:
            filt = [r for r in hp_results if np.isfinite(r["sharpe"]) and r["sortino"] > 0]
            candidate = max((filt or hp_results), key=lambda r: r["sharpe"])
            best_name, best_hp = candidate["model"], candidate["hp"]
        else:
            best_name, best_hp = "logit", {}
        print(f"[Outer {fold_idx}] Best inner-CV model: {best_name} HP: {best_hp}")

        # stability selection (optional)
        if len(imps_accum) > 0:
            try:
                imps_concat = pd.concat(imps_accum, ignore_index=True)
            except Exception:
                imps_concat = pd.DataFrame()
        else:
            imps_concat = pd.DataFrame()

        selected = _stability_select(imps_concat, top_n=cfg.get("features", {}).get("stability_top_n", 20))
        feat_cols_fold = [f for f in feat_cols if f in selected] if selected else feat_cols

        # final fit per outer fold
        Xtr_f = df.loc[mask].iloc[tr_idx][feat_cols_fold].to_numpy()
        Xva_f = df.loc[mask].iloc[va_idx][feat_cols_fold].to_numpy()
        Xtr_f = np.nan_to_num(Xtr_f, nan=0.0, posinf=0.0, neginf=0.0)
        Xva_f = np.nan_to_num(Xva_f, nan=0.0, posinf=0.0, neginf=0.0)
        ytr_f = y_dir[tr_idx]
        yva_f = y_dir[va_idx]
        wtr_f = w[tr_idx]
        wva_f = w[va_idx]

        if best_name == "logit":
            scaler = StandardScaler().fit(Xtr_f)
            Xtr_f = scaler.transform(Xtr_f)
            Xva_f = scaler.transform(Xva_f)

        if best_name == "lgbm":
            model_f, imps_f, _ = _train_lgbm(Xtr_f, ytr_f, wtr_f, Xva_f, yva_f, wva_f, best_hp, cfg["seed"], feature_names=feat_cols_fold)
        elif best_name == "catboost":
            model_f, imps_f, _ = _train_catboost(Xtr_f, ytr_f, wtr_f, Xva_f, yva_f, wva_f, best_hp, cfg["seed"])
        else:
            model_f, imps_f, _ = _train_logit(Xtr_f, ytr_f, wtr_f, Xva_f, yva_f, wva_f, best_hp, cfg["seed"])

        if isinstance(imps_f, pd.DataFrame) and not imps_f.empty:
            all_imps.append(imps_f)

        if hasattr(model_f, "predict_proba"):
            preds = model_f.predict_proba(Xva_f)[:, 1]
        else:
            preds = model_f.predict(Xva_f)
        oof[va_idx] = preds

    ts_dir = df.loc[mask, "timestamp"].reset_index(drop=True)
    oof_df = pd.DataFrame({"timestamp": ts_dir, "prob_long": oof})

    imps_df = pd.concat(all_imps, ignore_index=True) if all_imps else pd.DataFrame()
    return oof_df, imps_df
