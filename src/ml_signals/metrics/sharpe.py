"""
Utility functions for evaluating trading models based on economic metrics.

This module includes a weighted Sharpe ratio calculator that takes into
account sample weights.  The Sharpe ratio is defined as the mean excess
return divided by the standard deviation of returns.  In this context,
excess returns are simply the per-sample net returns (after costs) and
weights correspond to the importance of each observation (e.g., uniqueness).
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def weighted_sharpe_ratio(
    returns: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute a weighted Sharpe ratio of a return series.

    Parameters
    ----------
    returns : np.ndarray
        Array of per-sample net returns.
    weights : np.ndarray, optional
        Sample weights.  Must be the same length as `returns`.  If None,
        equal weights are assumed.

    Returns
    -------
    sharpe : float
        The weighted Sharpe ratio (mean divided by standard deviation).  If
        standard deviation is zero, returns 0 to avoid division by zero.
    """
    returns = np.asarray(returns, dtype=float)
    if weights is None:
        weights = np.ones_like(returns, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        # Ensure weights are non-negative
        weights = np.maximum(weights, 0.0)
    # Normalise weights to sum to 1 for numerical stability
    if weights.sum() > 0:
        weights = weights / weights.sum()
    mean = np.sum(weights * returns)
    # Compute weighted variance
    var = np.sum(weights * (returns - mean) ** 2)
    std = np.sqrt(var)
    if std == 0.0:
        return 0.0
    return mean / std