import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ..calibration.calibrate import fit_isotonic, fit_platt

log = get_logger()

# -------- helper: sub-sample large grids deterministically per seed ----------
def _iter_grid(grid_params, max_evals=None, seed=42):
    """
    Build a ParameterGrid from real hyperparameters only (drop meta keys),
    then optionally sample at most `max_evals` combos deterministically.
    """
    import numpy as np
    from sklearn.model_selection import ParameterGrid

    META_KEYS = {
        "hp_max_evals",
        "refine_best",
        "refine_max_evals",
        "early_stopping_rounds",
        "num_boost_round",
        "num_threads",
        "thread_count",
    }
    # keep only params that look like true grid lists/tuples/arrays and aren't meta
    clean = {}
    for k, v in (grid_params or {}).items():
        if k in META_KEYS:
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            clean[k] = list(v)

    # if nothing to grid over (e.g., user gave only meta), return a single empty dict
    combos = list(ParameterGrid(clean)) if clean else [dict()]

    if max_evals is not None and len(combos) > max_evals:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), size=max_evals, replace=False)
        combos = [combos[i] for i in idx]
    return combos


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Keep only numeric, non-leaky columns. Ban any names that look forward-looking,
    label-derived, or event-realized (ev_*). This is a hard denylist to prevent leakage.
    """
    import re

    # Explicit non-features (labels, outcomes, book-keeping, realized returns)
    drop_cols_explicit = {
        "timestamp", "y", "t_end", "tp_pct", "sl_pct", "sigma",
        "action", "pnl", "equity", "w",
        "ev_ret", "ev_ret_short", "ev_bps"
    }

    # Anything with these patterns is suspicious / forward-looking
    # Examples blocked: fwd_ret_5, future_close, lead_*, next_*, target, label, y_, ret_+5
    # Also *any* ev_* (event-realized) column name is banned from features.
    LEAK_PAT = re.compile(
        r"(?:^|_)(?:ev|fwd|future|lead|next|target|label|y(?:$|_))|ret_\+|return_\+",
        re.IGNORECASE
    )

    cols: List[str] = []
    for c in df.columns:
        if c in drop_cols_explicit:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if LEAK_PAT.search(c):
            continue
        cols.append(c)

    # Helpful diagnostic
    # Uncomment if you want to see the resulting count in logs:
    log.info(f"Selected {len(cols)} feature columns (after leakage screen).")
    return cols



def _make_directional_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only y in {-1,+1}. Map -1->0, +1->1. Return mask (on full df), y, sample weights."""
    mask = df["y"].isin([-1, 1])
    y = df.loc[mask, "y"].map({-1: 0, 1: 1}).values.astype(int)
    w = df.loc[mask, "w"].fillna(1.0).values
    return mask.values, y, w

def _purged_walk_splits_expanding(
    n: int,
    t_end: np.ndarray,
    n_splits: int,
    embargo: int,
    min_train: int = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window purged CV:
    - First validation starts after a warmup of at least `min_train` samples.
    - Train indices are [0 : v_start - embargo], purged via t_end < v_start.
    - Ensures every fold has non-empty training set; will relax embargo for the first fold if needed.
    """
    if min_train is None:
        min_train = max(1000, n // 5)
    min_train = max(1, int(min_train))

    remaining = max(0, n - min_train)
    if remaining < n_splits:
        n_splits = max(1, remaining) or 1

    val_size = max(1, remaining // n_splits) if n_splits > 0 else n
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(n_splits):
        v_start = min_train + k * val_size
        v_end = n if k == n_splits - 1 else min(min_train + (k + 1) * val_size, n)
        val_idx = np.arange(v_start, v_end)
        if len(val_idx) == 0:
            continue

        tr_mask = np.arange(n) < max(0, v_start - embargo)
        purge_mask = (t_end < v_start)
        train_idx = np.arange(n)[tr_mask & purge_mask]

        if len(train_idx) == 0:
            # Relax embargo for the very first fold if it kills the train set
            tr_mask_relaxed = np.arange(n) < v_start
            train_idx = np.arange(n)[tr_mask_relaxed & purge_mask]

        if len(train_idx) == 0:
            # Still empty — skip this fold
            log.warning(f"Skipping fold {k+1}: empty train after purge/embargo.")
            continue

        splits.append((train_idx, val_idx))

    return splits

def _split_tail(X, y, w, val_frac: float = 0.2):
    """Tail split inside the training window (causal)."""
    n = len(y)
    cut = max(1, int((1.0 - val_frac) * n))
    idx_tr = np.arange(0, cut)
    idx_va = np.arange(cut, n)
    return (X[idx_tr], y[idx_tr], w[idx_tr]), (X[idx_va], y[idx_va], w[idx_va])

# ---------------------- neighborhood helpers (refinement) ----------------------
def _bounds_from_grid(key: str, grid_params: Dict, default_low=None, default_high=None):
    vals = grid_params.get(key)
    if isinstance(vals, (list, tuple)) and len(vals) > 0 and np.isscalar(vals[0]):
        return (min(vals), max(vals))
    return (default_low, default_high)

def _clamp(x, lo, hi):
    if lo is not None: x = max(x, lo)
    if hi is not None: x = min(x, hi)
    return x

def _neighbors_logit(best: Dict, grid_params: Dict):
    C_lo, C_hi = _bounds_from_grid("C", grid_params, 1e-4, 1e4)
    l1_lo, l1_hi = 0.0, 1.0
    c = best.get("C", 1.0)
    l1 = best.get("l1_ratio", 0.5)
    C_cands = sorted(set([_clamp(c*0.5, C_lo, C_hi), _clamp(c, C_lo, C_hi), _clamp(c*2.0, C_lo, C_hi)]))
    l1_cands = sorted(set([_clamp(l1-0.25, l1_lo, l1_hi), _clamp(l1, l1_lo, l1_hi), _clamp(l1+0.25, l1_lo, l1_hi)]))
    return {"C": C_cands, "l1_ratio": l1_cands}

def _neighbors_lgbm(best: Dict, grid_params: Dict):
    md_lo, md_hi = _bounds_from_grid("max_depth", grid_params, 1, 12)
    nl_lo, nl_hi = _bounds_from_grid("num_leaves", grid_params, 4, 512)
    mil_lo, mil_hi = _bounds_from_grid("min_data_in_leaf", grid_params, 20, 5000)
    ff_lo, ff_hi = 0.5, 1.0
    bf_lo, bf_hi = 0.5, 1.0
    l1_lo, l1_hi = _bounds_from_grid("lambda_l1", grid_params, 0.0, 10.0)
    l2_lo, l2_hi = _bounds_from_grid("lambda_l2", grid_params, 0.0, 10.0)
    lr_lo, lr_hi = _bounds_from_grid("learning_rate", grid_params, 0.005, 0.3)

    def _uniq_sorted(vals, is_int=False):
        vals = list(sorted(set(vals)))
        return [int(round(v)) for v in vals] if is_int else vals

    md = best.get("max_depth", 3)
    nl = best.get("num_leaves", 16)
    mil = best.get("min_data_in_leaf", 200)
    ff = best.get("feature_fraction", 0.8)
    bf = best.get("bagging_fraction", 0.8)
    l1 = best.get("lambda_l1", 0.0)
    l2 = best.get("lambda_l2", 0.0)
    lr = best.get("learning_rate", 0.05)

    md_c = _uniq_sorted([_clamp(md-1, md_lo, md_hi), _clamp(md, md_lo, md_hi), _clamp(md+1, md_lo, md_hi)], True)
    nl_c = _uniq_sorted([_clamp(nl/2, nl_lo, nl_hi), _clamp(nl, nl_lo, nl_hi), _clamp(nl*2, nl_lo, nl_hi)], True)
    mil_c = _uniq_sorted([_clamp(mil/2, mil_lo, mil_hi), _clamp(mil, mil_lo, mil_hi), _clamp(mil*2, mil_lo, mil_hi)], True)
    ff_c = _uniq_sorted([_clamp(ff-0.1, ff_lo, ff_hi), _clamp(ff, ff_lo, ff_hi), _clamp(ff+0.1, ff_lo, ff_hi)])
    bf_c = _uniq_sorted([_clamp(bf-0.1, bf_lo, bf_hi), _clamp(bf, bf_lo, bf_hi), _clamp(bf+0.1, bf_lo, bf_hi)])
    l1_c = _uniq_sorted([_clamp(l1*0.1, l1_lo, l1_hi), _clamp(l1, l1_lo, l1_hi), _clamp(l1*10, l1_lo, l1_hi)])
    l2_c = _uniq_sorted([_clamp(l2*0.1, l2_lo, l2_hi), _clamp(l2, l2_lo, l2_hi), _clamp(l2*10, l2_lo, l2_hi)])
    lr_c = _uniq_sorted([_clamp(lr*0.5, lr_lo, lr_hi), _clamp(lr, lr_lo, lr_hi), _clamp(lr*1.5, lr_lo, lr_hi)])

    return {
        "max_depth": md_c,
        "num_leaves": nl_c,
        "min_data_in_leaf": mil_c,
        "feature_fraction": ff_c,
        "bagging_fraction": bf_c,
        "lambda_l1": l1_c,
        "lambda_l2": l2_c,
        "learning_rate": lr_c,
    }

def _neighbors_catb(best: Dict, grid_params: Dict):
    d_lo, d_hi = _bounds_from_grid("depth", grid_params, 2, 10)
    lr_lo, lr_hi = _bounds_from_grid("learning_rate", grid_params, 0.005, 0.3)
    l2_lo, l2_hi = _bounds_from_grid("l2_leaf_reg", grid_params, 1.0, 100.0)
    ss_lo, ss_hi = 0.5, 1.0

    depth = best.get("depth", 3)
    lr = best.get("learning_rate", 0.05)
    l2 = best.get("l2_leaf_reg", 3.0)
    subs = best.get("subsample", 0.8)

    depth_c = sorted(set([_clamp(depth-1, d_lo, d_hi), _clamp(depth, d_lo, d_hi), _clamp(depth+1, d_lo, d_hi)]))
    lr_c = sorted(set([_clamp(lr*0.5, lr_lo, lr_hi), _clamp(lr, lr_lo, lr_hi), _clamp(lr*1.5, lr_lo, lr_hi)]))
    l2_c = sorted(set([_clamp(l2*0.5, l2_lo, l2_hi), _clamp(l2, l2_lo, l2_hi), _clamp(l2*2.0, l2_lo, l2_hi)]))
    ss_c = sorted(set([_clamp(subs-0.1, ss_lo, ss_hi), _clamp(subs, ss_lo, ss_hi), _clamp(subs+0.1, ss_lo, ss_hi)]))

    return {"depth": depth_c, "learning_rate": lr_c, "l2_leaf_reg": l2_c, "subsample": ss_c}

# --------------------------------- trainers -----------------------------------
def _train_logit(X, y, w, Xv, yv, wv, grid: Dict, seed: int):
    max_evals = grid.get("hp_max_evals")
    best, best_score, best_hp = None, -np.inf, None
    for hp in _iter_grid(grid, max_evals=max_evals, seed=seed):
        clf = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=10000,
            C=hp.get("C", 1.0), l1_ratio=hp.get("l1_ratio", 0.5), random_state=seed
        )
        clf.fit(X, y, sample_weight=w)
        pv = clf.predict_proba(Xv)[:, 1]
        score = roc_auc_score(yv, pv, sample_weight=wv)
        if score > best_score:
            best, best_score, best_hp = clf, score, hp

    # refine around best (optional)
    if grid.get("refine_best", False) and best_hp is not None:
        neigh = _neighbors_logit(best_hp, grid)
        max_evals_ref = grid.get("refine_max_evals", 12)
        for hp in _iter_grid(neigh, max_evals=max_evals_ref, seed=seed+1):
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=10000,
                C=hp.get("C", 1.0), l1_ratio=hp.get("l1_ratio", 0.5), random_state=seed
            )
            clf.fit(X, y, sample_weight=w)
            pv = clf.predict_proba(Xv)[:, 1]
            score = roc_auc_score(yv, pv, sample_weight=wv)
            if score > best_score:
                best, best_score, best_hp = clf, score, hp
    return best, best_score, best_hp

def _train_lgbm(X, y, w, Xv, yv, wv, grid: Dict, seed: int, feature_names=None):
    import copy
    # Keep names for importance reporting
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    dtrain = lgb.Dataset(X, label=y, weight=w, free_raw_data=False, feature_name=feature_names)
    dvalid = lgb.Dataset(Xv, label=yv, weight=wv, reference=dtrain, feature_name=feature_names)

    # Pull out non-grid knobs
    early = grid.get("early_stopping_rounds", 200)
    nrounds = grid.get("num_boost_round", 5000)
    max_evals = grid.get("hp_max_evals")

    # Only grid-search real model params (strip meta keys)
    grid_params = copy.deepcopy(grid)
    for k in ("early_stopping_rounds", "num_boost_round", "hp_max_evals", "refine_best", "refine_max_evals"):
        grid_params.pop(k, None)

    best_model, best_score, best_hp = None, -np.inf, None
    base = dict(
        objective="binary",
        metric="auc",
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        learning_rate=0.05,
        lambda_l1=0.0,
        lambda_l2=0.0,
        bagging_freq=1,
        verbose=-1,
        seed=seed,
        num_threads=-1,           # use all cores
        feature_pre_filter=False, # allow min_data_in_leaf to vary across trials
        force_col_wise=True
    )

    for hp in _iter_grid(grid_params, max_evals=max_evals, seed=seed):
        params = base.copy(); params.update(hp)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=nrounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early, verbose=False)]
        )
        score = model.best_score.get("valid_0", {}).get("auc", -np.inf)
        if score > best_score:
            best_model, best_score, best_hp = model, score, hp

    # refine around best (optional)
    if grid.get("refine_best", False) and best_hp is not None:
        neigh = _neighbors_lgbm(best_hp, grid_params)
        max_evals_ref = grid.get("refine_max_evals", 20)
        for hp in _iter_grid(neigh, max_evals=max_evals_ref, seed=seed+1):
            params = base.copy(); params.update(hp)
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=nrounds,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(early, verbose=False)]
            )
            score = model.best_score.get("valid_0", {}).get("auc", -np.inf)
            if score > best_score:
                best_model, best_score, best_hp = model, score, hp

    return best_model, best_score, best_hp




def _train_catboost(X, y, w, Xv, yv, wv, grid: Dict, seed: int):
    import copy
    early = grid.get("early_stopping_rounds", 200)
    max_evals = grid.get("hp_max_evals")

    grid_params = copy.deepcopy(grid)
    for k in ("early_stopping_rounds", "hp_max_evals", "refine_best", "refine_max_evals"):
        grid_params.pop(k, None)

    best_model, best_score, best_hp = None, -np.inf, None
    base = dict(
        loss_function="Logloss",
        depth=3,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=seed,
        od_type="Iter",
        od_wait=early,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False
    )

    for hp in _iter_grid(grid_params, max_evals=max_evals, seed=seed):
        params = base.copy(); params.update(hp)
        model = CatBoostClassifier(**params)
        model.fit(
            X, y, sample_weight=w,
            eval_set=Pool(Xv, label=yv, weight=wv),
            use_best_model=True,
            verbose=False
        )
        pv = model.predict_proba(Xv)[:, 1]
        score = roc_auc_score(yv, pv, sample_weight=wv)
        if score > best_score:
            best_model, best_score, best_hp = model, score, hp

    if grid.get("refine_best", False) and best_hp is not None:
        neigh = _neighbors_catb(best_hp, grid_params)
        max_evals_ref = grid.get("refine_max_evals", 10)
        for hp in _iter_grid(neigh, max_evals=max_evals_ref, seed=seed+1):
            params = base.copy(); params.update(hp)
            model = CatBoostClassifier(**params)
            model.fit(
                X, y, sample_weight=w,
                eval_set=Pool(Xv, label=yv, weight=wv),
                use_best_model=True,
                verbose=False
            )
            pv = model.predict_proba(Xv)[:, 1]
            sscore = roc_auc_score(yv, pv, sample_weight=wv)
            if sscore > best_score:
                best_model, best_score, best_hp = model, sscore, hp

    return best_model, best_score, best_hp




# --------------------------------- pipeline -----------------------------------
def train_oof_models(df: pd.DataFrame, cfg: Dict, artifact_dir: str) -> pd.DataFrame:
    set_seed(cfg.get("seed", 42))
    os.makedirs(artifact_dir, exist_ok=True)

    feat_cols = _select_feature_cols(df)
    mask, y_dir, w = _make_directional_labels(df)

    # Features and event ends for directional rows
    X_full = df.loc[mask, feat_cols].fillna(0).values
    t_end = df.loc[mask, "t_end"].values.astype(int)

    # Timestamps for directional rows and event end timestamps
    df_dir_ts = df.loc[mask, "timestamp"].to_numpy()
    t_end_idx = np.clip(t_end, 0, len(df_dir_ts) - 1)   # safety clamp
    t_end_ts  = df_dir_ts[t_end_idx]

    # Outer splits: purged + embargoed
    n_splits = cfg["validation"]["folds"]
    H = cfg["labeling"]["horizon_minutes"][0]
    embargo = H if str(cfg["validation"].get("embargo_minutes", "auto")) == "auto" else int(cfg["validation"]["embargo_minutes"])
    min_train = int(cfg.get("validation", {}).get("min_train_bars", max(1000, len(y_dir) // 5)))
    splits = _purged_walk_splits_expanding(len(y_dir), t_end, n_splits, embargo, min_train=min_train)

    grid_logit = cfg["models"]["logit"]
    grid_lgbm = cfg["models"]["lightgbm"]
    grid_catb = cfg["models"]["catboost"]
    calib_method = cfg["calibration"]["method"]

    oof = pd.DataFrame(index=df.index, columns=["prob_long"])

    # collect LGBM importances per fold
    imp_records = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        log.info(f"Fold {fold_idx+1}/{len(splits)}: train={len(tr_idx)} val={len(va_idx)}")

        Xtr, ytr, wtr = X_full[tr_idx], y_dir[tr_idx], w[tr_idx]
        Xva, yva, wva = X_full[va_idx], y_dir[va_idx], w[va_idx]

        # --- Single-feature AUC scan (diagnostics for leakage) ---
        try:
            sf_scores = []
            for j, name in enumerate(feat_cols):
                x = Xva[:, j]
                if np.std(x) == 0:
                    continue
                auc1 = roc_auc_score(yva, x, sample_weight=wva)
                auc1 = float(auc1 if auc1 >= 0.5 else 1.0 - auc1)
                sf_scores.append((name, auc1))
            sf_scores.sort(key=lambda t: t[1], reverse=True)
            top = ", ".join([f"{n}:{a:.3f}" for n, a in sf_scores[:5]])
            log.info(f"    Top single-feature AUCs (fold-val): {top}")
        except Exception as e:
            log.warning(f"    Single-feature AUC scan skipped: {e}")
        # ----------------------------------------------------------

        # Guards: enough data and both classes
        if len(ytr) < 100 or np.unique(ytr).size < 2:
            pv_va_cal = np.full(len(yva), 0.5, dtype=float)
            oof_idx = np.where(mask)[0][va_idx]
            oof.loc[oof_idx, "prob_long"] = pv_va_cal
            log.warning(f"Fold {fold_idx+1}: skipped training (len(ytr)={len(ytr)}, classes={np.unique(ytr) if len(ytr)>0 else []}). Filled 0.5.")
            continue

        # --- Purged inner tail split (timestamp-based) ---
        val_frac = 0.2
        ntr = len(ytr)
        cut = max(1, int((1.0 - val_frac) * ntr))

        va_start_abs = tr_idx[cut] if cut < ntr else tr_idx[-1] + 1
        va_start_abs = min(va_start_abs, len(df_dir_ts) - 1)  # clamp
        va_start_ts = df_dir_ts[va_start_abs]

        emb_mins = int(embargo)
        emb_delta = np.timedelta64(emb_mins, 'm')

        tr_cand_loc = np.arange(cut)
        keep_mask = t_end_ts[tr_idx[:cut]] < (va_start_ts - emb_delta)
        if keep_mask.sum() == 0:
            keep_mask = t_end_ts[tr_idx[:cut]] < va_start_ts

        tr_keep_loc = tr_cand_loc[keep_mask]
        log.info(f"    inner-train kept after purge: {len(tr_keep_loc)}/{cut}")

        # final inner-train / inner-val subsets
        Xtr_i, ytr_i, wtr_i = Xtr[tr_keep_loc], ytr[tr_keep_loc], wtr[tr_keep_loc]
        Xval_i, yval_i, wval_i = Xtr[cut:], ytr[cut:], wtr[cut:]
        log.info(f"    inner-val size={len(yval_i)}, pos={int(yval_i.sum())}, neg={len(yval_i)-int(yval_i.sum())}")
        # ----------------------------------------------------------------------

        # Standardize for logit using only inner-train
        scaler = StandardScaler().fit(Xtr_i)
        Xtr_is = scaler.transform(Xtr_i)
        Xval_is = scaler.transform(Xval_i)
        Xva_s = scaler.transform(Xva)

        # -------------------- fit on inner-train only --------------------
        logit_model, logit_score, _ = _train_logit(
            Xtr_is, ytr_i, wtr_i, Xval_is, yval_i, wval_i, grid_logit, cfg["seed"]
        )
        lgbm_model, lgbm_score, _ = _train_lgbm(
            Xtr_i, ytr_i, wtr_i, Xval_i, yval_i, wval_i, grid_lgbm, cfg["seed"], feature_names=feat_cols
        )
        catb_model, catb_score, _ = _train_catboost(
            Xtr_i, ytr_i, wtr_i, Xval_i, yval_i, wval_i, grid_catb, cfg["seed"]
        )
        # -----------------------------------------------------------------

        # collect LGBM importances (even if LGBM not the winner)
        try:
            fnames = lgbm_model.feature_name()
            gain = lgbm_model.feature_importance(importance_type="gain")
            split = lgbm_model.feature_importance(importance_type="split")
            for f, g, s in zip(fnames, gain, split):
                imp_records.append({"fold": fold_idx + 1, "feature": f, "gain": float(g), "split": int(s)})
        except Exception as e:
            log.warning(f"    Importance capture skipped: {e}")

        scores = {"logit": logit_score, "lgbm": lgbm_score, "catboost": catb_score}
        best_name = max(scores, key=scores.get)
        log.info(f"Fold {fold_idx+1} winner: {best_name} (AUC={scores[best_name]:.4f})")

        # Get predictions for calibrator (inner-val) + for fold OOF
        if best_name == "logit":
            pv_val_inner = logit_model.predict_proba(Xval_is)[:, 1]
            pv_va_raw = logit_model.predict_proba(Xva_s)[:, 1]
        elif best_name == "lgbm":
            pv_val_inner = lgbm_model.predict(Xval_i)
            pv_va_raw = lgbm_model.predict(Xva)
        else:
            pv_val_inner = catb_model.predict_proba(Xval_i)[:, 1]
            pv_va_raw = catb_model.predict_proba(Xva)[:, 1]

        # Calibrator trained ONLY on inner-validation (weighted)
        if calib_method == "isotonic":
            calibrator = fit_isotonic(pv_val_inner, yval_i, sample_weight=wval_i)
            pv_va_cal = calibrator.predict(pv_va_raw)
        else:
            calibrator = fit_platt(pv_val_inner, yval_i, sample_weight=wval_i)
            pv_va_cal = calibrator.predict_proba(pv_va_raw.reshape(-1, 1))[:, 1]

        # Store OOF predictions aligned to original index
        oof_idx = np.where(mask)[0][va_idx]
        oof.loc[oof_idx, "prob_long"] = pv_va_cal

    # Persist OOF probs (timestamp, prob_long) — hardened
    out = pd.concat([df[["timestamp"]], oof], axis=1)

    # Force clean dtypes and drop junk before saving
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["prob_long"] = pd.to_numeric(out["prob_long"], errors="coerce")

    out = out.dropna(subset=["timestamp", "prob_long"])\
             .sort_values("timestamp")\
             .reset_index(drop=True)

    oof_path = os.path.join(artifact_dir, "oof_probs.csv")
    out.to_csv(oof_path, index=False, float_format="%.10f")
    log.info(f"Saved OOF probabilities to {oof_path} (n={len(out)})")

    # Save LGBM importances by fold + stability report
    if imp_records:
        imp_df = pd.DataFrame(imp_records)
        imp_path = os.path.join(artifact_dir, "lgbm_feature_importance_by_fold.csv")
        imp_df.to_csv(imp_path, index=False)
        log.info(f"Saved LGBM importances to {imp_path}")

        try:
            from ..reports.feature_stability import make_lgbm_feature_stability
            os.makedirs(cfg["report_dir"], exist_ok=True)
            make_lgbm_feature_stability(imp_df, cfg["report_dir"], top_k=30)
        except Exception as e:
            log.warning(f"Feature stability report skipped: {e}")

    return out
