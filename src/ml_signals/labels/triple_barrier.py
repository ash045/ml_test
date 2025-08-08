import pandas as pd, numpy as np
def triple_barrier_labels(df: pd.DataFrame, sigma_window: int, k: float, horizon: int) -> pd.DataFrame:
    out = df.copy()
    ret = np.log(out["close"].replace(0, np.nan)).diff()
    sigma = ret.rolling(sigma_window, min_periods=sigma_window).std()
    out["sigma"] = sigma
    out["tp_pct"] = k * sigma
    out["sl_pct"] = -k * sigma
    y = np.full(len(out), 0, dtype=int)
    t_end = np.arange(len(out))
    closes = out["close"].values
    for t in range(len(out) - horizon - 1):
        if not np.isfinite(out["tp_pct"].iloc[t]): continue
        p0 = closes[t]
        upper = p0 * np.exp(out["tp_pct"].iloc[t])
        lower = p0 * np.exp(out["sl_pct"].iloc[t])
        window = closes[t+1:t+1+horizon]
        up_idx = np.where(window >= upper)[0]
        dn_idx = np.where(window <= lower)[0]
        up = up_idx[0]+1 if len(up_idx)>0 else None
        dn = dn_idx[0]+1 if len(dn_idx)>0 else None
        if up is not None and (dn is None or up < dn):
            y[t] = 1; t_end[t] = t + up
        elif dn is not None and (up is None or dn < up):
            y[t] = -1; t_end[t] = t + dn
        else:
            y[t] = 0; t_end[t] = t + horizon
    return pd.DataFrame({"y": y, "t_end": t_end, "tp_pct": out["tp_pct"], "sl_pct": out["sl_pct"], "sigma": out["sigma"]})
def compute_concurrency_weights(t_end: pd.Series) -> pd.Series:
    n = len(t_end)
    conc = np.zeros(n, dtype=int)
    for t in range(n):
        end = int(t_end.iloc[t])
        conc[t:end+1] += 1
    w = 1.0 / np.maximum(conc, 1)
    w = w / np.nanmean(w)
    return pd.Series(w)
