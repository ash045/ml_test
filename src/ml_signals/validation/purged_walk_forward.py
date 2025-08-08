"""
Purged walk-forward cross-validation splits with optional final test hold-out.

This module provides a function to generate expanding-window purged
cross-validation splits while optionally reserving a final fraction of the data
for out-of-sample testing.  The splitting ensures that training samples do
not leak information from future events by purging overlapping event horizons
and applying an embargo (gap) before the validation period.

The logic is adapted from the existing `_purged_walk_splits_expanding` in
`train_oof.py` but extended to support a final hold-out and configurable
minimum training size.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def purged_walk_forward_splits(
    n: int,
    t_end: np.ndarray,
    n_splits: int,
    embargo: int,
    final_test_fraction: float = 0.0,
    min_train: int | None = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Generate expanding-window purged cross-validation splits with optional final hold-out.

    Parameters
    ----------
    n : int
        Number of samples.
    t_end : np.ndarray
        Array of event end indices (length n) used for purging.  Entries should be
        integers; invalid or missing values should be -1.
    n_splits : int
        Number of CV splits in the training+validation portion (excluding final test).
    embargo : int
        Number of samples to embargo (gap) between the end of the training set and
        start of the validation set.
    final_test_fraction : float, optional
        Fraction of the tail of the dataset to reserve for a final test set.  Must be
        between 0 and 0.5.  Defaults to 0.0 (no hold-out).
    min_train : int, optional
        Minimum number of samples in the initial training window.  If None,
        defaults to max(1, n // 5).

    Returns
    -------
    splits : list of (train_idx, val_idx)
        List of tuples containing the indices for each training and validation fold.
    test_idx : np.ndarray
        Array of indices reserved for the final test set.  Empty if
        `final_test_fraction` == 0.
    """
    # Validate arguments
    n = int(n)
    if n <= 0:
        return [], np.array([], dtype=int)
    final_test_fraction = float(final_test_fraction)
    if final_test_fraction < 0.0 or final_test_fraction >= 0.5:
        raise ValueError("final_test_fraction must be between 0 and <0.5")
    if min_train is None:
        # by default, allocate 20% of available samples to initial train
        min_train = max(1, n // 5)
    min_train = max(1, int(min_train))

    # Determine number of samples for final test
    n_test = int(n * final_test_fraction)
    train_val_len = n - n_test
    if train_val_len <= 0:
        # all data reserved for testing
        return [], np.arange(n, dtype=int)

    # Outer CV splits within the train+validation window
    # Adjust number of splits if there isn't enough data
    remaining = max(0, train_val_len - min_train)
    if remaining < n_splits:
        n_splits = max(1, remaining) or 1

    val_size = max(1, remaining // n_splits) if n_splits > 0 else train_val_len
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(n_splits):
        v_start = min_train + k * val_size
        v_end = train_val_len if k == n_splits - 1 else min(min_train + (k + 1) * val_size, train_val_len)
        val_idx = np.arange(v_start, v_end)
        if len(val_idx) == 0:
            continue

        # Training indices are everything before v_start minus embargo
        tr_mask = np.arange(train_val_len) < max(0, v_start - embargo)
        # Purge any samples whose event horizon (t_end) falls within the validation period
        purge_mask = (t_end[:train_val_len] < v_start)
        train_idx = np.arange(train_val_len)[tr_mask & purge_mask]

        # Relax embargo for first fold if necessary
        if len(train_idx) == 0:
            tr_relaxed = np.arange(train_val_len) < v_start
            train_idx = np.arange(train_val_len)[tr_relaxed & purge_mask]

        if len(train_idx) == 0:
            continue
        splits.append((train_idx, val_idx))

    # Final test indices (empty if no hold-out)
    test_idx = np.arange(train_val_len, n) if n_test > 0 else np.array([], dtype=int)
    return splits, test_idx