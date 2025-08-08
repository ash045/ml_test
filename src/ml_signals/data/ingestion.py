import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.datetime import ensure_utc


def read_ohlcv_csv(path: str, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Read OHLCV CSV, ensure UTC, and return DataFrame sorted by timestamp.
    """
    df = pd.read_csv(path)
    df = ensure_utc(df, ts_col)
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df


def merge_and_reconcile(feeds: List[pd.DataFrame], ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Merge multiple OHLCV feeds on timestamp, preferring non-NaN values.
    This is optional and only used when combining multi-exchange data.
    """
    base = feeds[0].copy()
    for feed in feeds[1:]:
        feed = feed.sort_values(ts_col)
        base = base.combine_first(feed).sort_values(ts_col)
    return base.reset_index(drop=True)


def log_gaps(df: pd.DataFrame, ts_col: str = "timestamp", freq: str = "T") -> List[pd.Timestamp]:
    """
    Identify gaps in the time series and return list of missing timestamps.
    """
    full_index = pd.date_range(df[ts_col].min(), df[ts_col].max(), freq=freq, tz="UTC")
    missing = full_index.difference(df[ts_col])
    if not missing.empty:
        print(f"[Data Integrity] Found {len(missing)} missing bars between "
              f"{missing.min()} and {missing.max()}")
    return list(missing)


def remove_duplicates(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Remove duplicate timestamps, keeping the first occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[Data Integrity] Removed {before - after} duplicate rows.")
    return df


def fill_missing_minutes(df: pd.DataFrame,
                         ts_col: str = "timestamp",
                         freq: str = "T",
                         fill_value: Optional[dict] = None) -> pd.DataFrame:
    """
    Fill missing bars explicitly with NaNs, then forward-fill OHLC,
    and set volume/trades to 0.
    """
    df = df.set_index(ts_col)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    df = df.reindex(full_index)

    # Forward-fill OHLC, zero-fill volume/trades
    ohlc_cols = ["open", "high", "low", "close"]
    vol_cols = ["volume", "trades"]

    df[ohlc_cols] = df[ohlc_cols].ffill()
    df[vol_cols] = df[vol_cols].fillna(0)

    df = df.reset_index().rename(columns={"index": ts_col})
    return df


def audit_ohlcv(df: pd.DataFrame, ts_col: str = "timestamp") -> None:
    """
    Validate data integrity: monotonic timestamps, non-negative values,
    no duplicates. Prints gap info but does not stop execution unless
    critical integrity issues are found.
    """
    assert df[ts_col].is_monotonic_increasing, "Timestamps must be increasing"
    for c in ["open", "high", "low", "close", "volume", "trades"]:
        assert (df[c] >= 0).all(), f"Negative values in {c}"
    dup = df[ts_col].duplicated().sum()
    assert dup == 0, f"Duplicate timestamp rows: {dup}"
    log_gaps(df, ts_col)
