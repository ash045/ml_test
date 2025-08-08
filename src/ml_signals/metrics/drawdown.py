import numpy as np

def max_drawdown(returns: np.ndarray) -> float:
    """
    Compute max drawdown from a series of returns (not prices).
    Returns a positive float (e.g., 0.25 means -25% drawdown).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan

    # Convert returns to cumulative equity curve
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    return -np.min(drawdowns)  # positive number
