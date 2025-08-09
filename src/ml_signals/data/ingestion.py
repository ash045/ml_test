# src/ml_signals/data/ingestion.py
# -*- coding: utf-8 -*-
"""Ingestion helpers with robust gap logging.

- `log_gaps(df, ts_col, freq=None)`: if `freq` omitted, infer from timestamps; otherwise honor provided
  timeframe (e.g., "15min"). Uses the index timezone when building the full range.
- `audit_ohlcv(df, ts_col="timestamp", freq=None)`: calls `log_gaps` with the inferred/provided
  frequency and performs basic assertions.
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd

from ..utils.datetime import ensure_utc


def read_ohlcv_csv(path: str, ts_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_utc(df, ts_col)
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df


def merge_and_reconcile(feeds: List[pd.DataFrame], ts_col: str = "timestamp") -> pd.DataFrame:
    base = feeds[0].copy()
    for feed in feeds[1:]:
        feed = feed.sort_values(ts_col)
        base = base.combine_first(feed).sort_values(ts_col)
    return base.reset_index(drop=True)


def _infer_freq(idx: pd.DatetimeIndex) -> Optional[str]:
    idx = pd.DatetimeIndex(idx).sort_values()
    # Try pandas native inference first
    f = pd.infer_freq(idx)
    if f:
        return f.replace("T", "min")  # avoid deprecated alias
    # Fallback: use the mode of diffs
    diffs = pd.Series(idx).diff().dropna()
    if diffs.empty:
        return None
    step = diffs.mode().iloc[0]
    try:
        return pd.tseries.frequencies.to_offset(step).freqstr
    except Exception:
        return None


def log_gaps(df: pd.DataFrame, ts_col: str = "timestamp", freq: Optional[str] = None) -> list[pd.Timestamp]:
    """Identify missing timestamps at the requested or inferred frequency.

    If `freq` is None, infer from the series. If provided (e.g., "15min"), use it directly.
    The generated full_index uses the same timezone as the input timestamps.
    """
    ts = pd.DatetimeIndex(df[ts_col])
    tz = ts.tz
    use_freq = (freq or _infer_freq(ts) or "min").replace("T", "min")
    full_index = pd.date_range(ts.min(), ts.max(), freq=use_freq, tz=tz)
    missing = full_index.difference(ts)
    if not missing.empty:
        print(
            f"[Data Integrity] Found {len(missing)} missing bars at '{use_freq}' between "
            f"{missing.min()} and {missing.max()}"
        )
    return list(missing)


def remove_duplicates(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[ts_col], keep="first").reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[Data Integrity] Removed {before - after} duplicate rows.")
    return df


def fill_missing_minutes(
    df: pd.DataFrame, ts_col: str = "timestamp", freq: str = "min", fill_value: Optional[dict] = None
) -> pd.DataFrame:
    # Kept for backward compatibility; not used in the new path.
    df = df.set_index(ts_col)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
    df = df.reindex(full_index)
    ohlc_cols = ["open", "high", "low", "close"]
    vol_cols = ["volume", "trades"]
    df[ohlc_cols] = df[ohlc_cols].ffill()
    df[vol_cols] = df[vol_cols].fillna(0)
    df = df.reset_index().rename(columns={"index": ts_col})
    return df


def audit_ohlcv(df: pd.DataFrame, ts_col: str = "timestamp", freq: Optional[str] = None) -> None:
    """Validate basic integrity and print gap info at the given or inferred frequency."""
    assert df[ts_col].is_monotonic_increasing, "Timestamps must be increasing"
    for c in ["open", "high", "low", "close", "volume", "trades"]:
        assert (df[c] >= 0).all(), f"Negative values in {c}"
    dup = df[ts_col].duplicated().sum()
    assert dup == 0, f"Duplicate timestamp rows: {dup}"
    log_gaps(df, ts_col=ts_col, freq=freq)
