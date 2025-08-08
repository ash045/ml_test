import numpy as np
def sharpe(returns, bars_per_year=365*24*60):
    r = np.asarray(returns)
    mu = r.mean() * bars_per_year
    sd = r.std() * (bars_per_year**0.5)
    return float(mu / (sd + 1e-12))
def max_drawdown(equity_curve):
    eq = np.asarray(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return float(dd.min())
