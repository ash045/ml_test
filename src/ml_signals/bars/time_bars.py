import pandas as pd


def resample_time_bars(df: pd.DataFrame, timeframe: str, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Resample raw OHLCV data into fixed time bars and fill any missing bars.

    This function differs from the original implementation in two key ways:

    1. It constructs a complete date range at the requested `timeframe` between the first
       and last timestamps, ensuring that no minute (or other timeframe) is missing.
    2. It forward-fills price columns (open, high, low, close) for missing bars and
       sets volume/trades to zero.  Filling missing bars prevents the inadvertent
       removal of data and avoids artificial gaps in downstream features.

    Args:
        df: DataFrame containing at least a timestamp column and OHLCV columns.
        timeframe: A pandas offset alias such as '1min', '5min', '15min', etc.
        ts_col: Name of the timestamp column (will be converted to DateTimeIndex).

    Returns:
        DataFrame indexed by the complete date range with all bars filled.
    """
    # Ensure we operate on a copy and have a datetime index
    df = df.copy()
    idx = pd.DatetimeIndex(df[ts_col])
    df = df.set_index(idx)

    # Define aggregation for OHLCV/trades when resampling
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "trades" in df.columns:
        agg["trades"] = "sum"

    # Resample to the desired timeframe without dropping missing periods
    resampled = df.resample(timeframe).agg(agg)

    # Build a complete index from start to end at the given frequency
    full_index = pd.date_range(start=resampled.index.min(), end=resampled.index.max(), freq=timeframe)

    # Reindex to the full index (this introduces NaNs for missing periods)
    out = resampled.reindex(full_index)

    # Forward-fill price columns from the previous known bar
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])

    # Volume and trades are zero where there was no activity
    out["volume"] = out["volume"].fillna(0)
    if "trades" in out.columns:
        out["trades"] = out["trades"].fillna(0)

    # Drop any rows where price columns are still NaN (e.g., leading NaNs)
    out = out.dropna(subset=["close", "open", "high", "low"])

    # Restore the timestamp column for downstream code
    out[ts_col] = out.index

    return out.reset_index(drop=True)
