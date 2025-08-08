import pandas as pd
import numpy as np


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a richer set of causal features for financial time series.

    This extended version computes returns over multiple horizons (1,5,15,30,60,90 minutes),
    volatility over multiple windows, range features, volume/trade intensity z-scores,
    higher moments of returns, and trades-based features.  All rolling statistics are
    computed using a minimum period equal to the window to avoid look-ahead bias.

    Args:
        df: Input DataFrame containing at least 'close', 'high', 'low', 'open', 'volume'
             and optionally 'trades'.  Must be time-indexed.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()

    # Log returns over various horizons (causal)
    logp = np.log(df["close"].replace(0, np.nan))
    for h in [1, 5, 15, 30, 60, 90]:
        df[f"ret_{h}"] = logp.diff(h)

    # Volatility (rolling standard deviation of 1-bar returns)
    for w in [15, 30, 60, 90, 240]:
        df[f"vol_{w}"] = df["ret_1"].rolling(w, min_periods=w).std()

    # Range features
    prev_close = df["close"].shift(1).replace(0, np.nan)
    df["range_hl"] = (df["high"] - df["low"]) / prev_close
    df["range_co"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    # Volume intensity z-scores
    for w in [15, 30, 60, 90, 240]:
        m = df["volume"].rolling(w, min_periods=w).mean()
        s = df["volume"].rolling(w, min_periods=w).std()
        df[f"vol_z_{w}"] = (df["volume"] - m) / s.replace(0, np.nan)

    # Higher moments of returns (skewness and kurtosis)
    for w in [20, 60]:
        r = df["ret_1"].rolling(w, min_periods=w)
        df[f"ret_skew_{w}"] = r.skew()
        df[f"ret_kurt_{w}"] = r.kurt()

    # Trades-based features (if trades column exists)
    if "trades" in df.columns:
        trades = df["trades"]
        vol = df["volume"]
        df["avg_trade_size"] = vol / trades.replace(0, np.nan)
        df["tickiness"] = trades / vol.replace(0, np.nan)
        for w in [15, 30, 60, 90, 240]:
            m_t = trades.rolling(w, min_periods=w).mean()
            s_t = trades.rolling(w, min_periods=w).std()
            df[f"trades_z_{w}"] = (trades - m_t) / s_t.replace(0, np.nan)
            df[f"trades_cv_{w}"] = s_t / m_t.replace(0, np.nan)
            m_a = df["avg_trade_size"].rolling(w, min_periods=w).mean()
            s_a = df["avg_trade_size"].rolling(w, min_periods=w).std()
            df[f"avg_trade_size_z_{w}"] = (df["avg_trade_size"] - m_a) / s_a.replace(0, np.nan)
            df[f"avg_trade_size_cv_{w}"] = s_a / m_a.replace(0, np.nan)

    return df
