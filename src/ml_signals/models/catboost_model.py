from catboost import CatBoostClassifier, Pool
def train_catboost(X, y, params: dict, valid_set=None, sample_weight=None, es_rounds=200, seed=42):
    default = dict(loss_function="Logloss", depth=3, learning_rate=0.05, l2_leaf_reg=3.0,
                   random_seed=seed, od_type="Iter", verbose=False)
    default.update(params or {})
    model = CatBoostClassifier(**default)
    eval_set = None
    if valid_set is not None:
        Xv, yv, wv = valid_set
        eval_set = Pool(Xv, label=yv, weight=wv)
    model.fit(X, y, sample_weight=sample_weight, eval_set=eval_set, use_best_model=True, od_wait=es_rounds)
    return model
