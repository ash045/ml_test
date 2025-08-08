import numpy as np
from ml_signals.validation.purged_cv import PurgedKFold
def test_purged_kfold_basic():
    n = 100
    y_end = (np.arange(n) + 5).astype(int)
    cv = PurgedKFold(n_splits=5, embargo=5)
    folds = cv.split(n, y_end)
    assert len(folds) == 5
    for train, val in folds:
        assert set(train).isdisjoint(set(val))
