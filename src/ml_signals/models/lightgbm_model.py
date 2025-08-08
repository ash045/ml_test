import lightgbm as lgb
def train_lgbm(X, y, params: dict, valid_set=None, sample_weight=None, es_rounds=200, seed=42):
    dtrain = lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)
    valid = None
    if valid_set is not None:
        Xv, yv, wv = valid_set
        valid = [lgb.Dataset(Xv, label=yv, weight=wv, reference=dtrain)]
    default = dict(objective="binary", metric="auc", max_depth=3, num_leaves=8,
                   min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                   learning_rate=0.05, lambda_l1=0.0, lambda_l2=0.0, bagging_freq=1,
                   verbose=-1, seed=seed)
    default.update(params or {})
    booster = lgb.train(default, dtrain, num_boost_round=5000,
                        valid_sets=valid, callbacks=[lgb.early_stopping(es_rounds)] if valid else None)
    return booster
