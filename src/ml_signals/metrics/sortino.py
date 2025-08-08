import numpy as np

def weighted_sortino_ratio(returns: np.ndarray, weights: np.ndarray, target: float = 0.0) -> float:
    """
    Weighted Sortino ratio: (mean(returns - target)) / downside_std(returns - target)
    Only downside deviation is used in denominator.
    """
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = (weights > 0) & np.isfinite(returns)

    if mask.sum() < 2:
        return np.nan

    r = returns[mask]
    w = weights[mask]
    excess = r - target

    downside = excess[excess < 0]
    downside_w = w[excess < 0]

    mean_excess = np.average(excess, weights=w)
    if downside.size == 0:
        return np.inf

    downside_std = np.sqrt(np.average(downside ** 2, weights=downside_w))
    if downside_std == 0:
        return np.inf

    return mean_excess / downside_std
