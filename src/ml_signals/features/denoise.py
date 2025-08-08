import pandas as pd
def ema_causal(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()
def apply_denoise(df: pd.DataFrame, ema_windows=(3,5)) -> pd.DataFrame:
    df = df.copy()
    for w in ema_windows:
        df[f"close_ema_{w}"] = ema_causal(df["close"], w)
    return df
