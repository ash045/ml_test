import pandas as pd
from ..utils.datetime import ensure_utc
def read_ohlcv_csv(path: str, ts_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_utc(df, ts_col)
    return df
def audit_ohlcv(df: pd.DataFrame, ts_col: str = "timestamp") -> None:
    assert df[ts_col].is_monotonic_increasing, "Timestamps must be increasing"
    for c in ["open","high","low","close","volume","trades"]:
        assert (df[c] >= 0).all(), f"Negative values in {c}"
    dup = df[ts_col].duplicated().sum()
    assert dup == 0, f"Duplicate timestamp rows: {dup}"
