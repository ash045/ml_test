import numpy as np
import pandas as pd


def triple_barrier_multi_labels(
    df: pd.DataFrame,
    sigma_window: int,
    horizons: list,
    k_list: list,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute triple-barrier event labels for multiple horizons and k-sigma thresholds.

    For each combination of horizon H and multiplier k, the function sets a take-profit (TP) and
    stop-loss (SL) barrier at k * sigma and -k * sigma, respectively, where sigma is a rolling
    standard deviation of log returns computed over `sigma_window`.  The function then scans
    forward H bars to determine whether the TP or SL barrier is hit first.  If neither barrier
    is hit, the event ends at the vertical barrier (H bars ahead) and is labeled 0.

    The output contains, for each (H,k) pair, two columns: y_{H}_{k} and t_end_{H}_{k}, where
    y takes values {+1, -1, 0} and t_end is the index of the bar where the event concludes.
    Additionally, the output includes a common "sigma" column for reference.

    Args:
        df: Input DataFrame with at least a price column `price_col`.
        sigma_window: Rolling window length for estimating volatility.
        horizons: List of integer horizons (in number of bars) to apply.
        k_list: List of multipliers (floats) applied to sigma for TP/SL.
        price_col: Name of the price column.

    Returns:
        DataFrame containing sigma and, for each (H,k) pair, y_{H}_{k} and t_end_{H}_{k}.
    """
    out = pd.DataFrame(index=df.index)
    closes = pd.to_numeric(df[price_col], errors="coerce").values

    # Compute rolling volatility (sigma) of log returns
    log_ret = np.log(closes).astype(float)
    ret = np.diff(log_ret, prepend=np.nan)
    sigma = pd.Series(ret).rolling(sigma_window, min_periods=sigma_window).std().to_numpy()
    out["sigma"] = sigma

    n = len(df)
    for H in horizons:
        for k in k_list:
            y = np.zeros(n, dtype=int)
            t_end = np.arange(n, dtype=int)
            for t in range(n - H):
                if not np.isfinite(sigma[t]):
                    continue
                p0 = closes[t]
                tp_pct = k * sigma[t]
                sl_pct = -k * sigma[t]
                # Convert pct barriers into price barriers in log-space to handle multiplicative returns
                upper = p0 * np.exp(tp_pct)
                lower = p0 * np.exp(sl_pct)
                window = closes[t + 1 : t + 1 + H]
                # find first index where price >= upper or <= lower
                up_hit = np.argmax(window >= upper)
                dn_hit = np.argmax(window <= lower)
                # np.argmax returns 0 even if no condition is met; handle separately
                up = None
                dn = None
                if (window >= upper).any():
                    up = up_hit + 1  # +1 because event end is relative to t+1
                if (window <= lower).any():
                    dn = dn_hit + 1
                if up is not None and (dn is None or up < dn):
                    y[t] = 1
                    t_end[t] = t + up
                elif dn is not None and (up is None or dn < up):
                    y[t] = -1
                    t_end[t] = t + dn
                else:
                    y[t] = 0
                    t_end[t] = t + H
            out[f"y_{H}_{k}"] = y
            out[f"t_end_{H}_{k}"] = t_end
    return out
