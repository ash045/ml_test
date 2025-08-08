import pandas as pd
def resample_time_bars(df: pd.DataFrame, timeframe: str, ts_col: str = "timestamp") -> pd.DataFrame:
    # timeframe like '1min', '5min', '15min', '60min'
    df = df.set_index(pd.DatetimeIndex(df[ts_col]))
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    if "trades" in df.columns:
        agg["trades"] = "sum"
    out = df.resample(timeframe).agg(agg).dropna()
    out[ts_col] = out.index
    return out.reset_index(drop=True)
