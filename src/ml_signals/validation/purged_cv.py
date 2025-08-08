import numpy as np
from typing import List, Tuple
class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: int = 0):
        assert n_splits > 1
        self.n_splits = n_splits
        self.embargo = embargo
    def split(self, X_len: int, y_end_idx: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        fold_sizes = np.full(self.n_splits, X_len // self.n_splits, dtype=int)
        fold_sizes[: X_len % self.n_splits] += 1
        indices = np.arange(X_len)
        current = 0; folds = []
        for fs in fold_sizes:
            start, stop = current, current + fs
            val_idx = indices[start:stop]
            train_mask = (indices < max(0, start - self.embargo))
            purge_mask = (y_end_idx < start)
            train_idx = indices[train_mask & purge_mask]
            folds.append((train_idx, val_idx))
            current = stop
        return folds
