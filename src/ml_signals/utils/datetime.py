import pandas as pd
def ensure_utc(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col)
    return df
