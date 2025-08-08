import pandas as pd, numpy as np
def fracdiff(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres: break
        w.append(w_); k += 1
    w = np.array(w[::-1])
    out = pd.Series(index=series.index, dtype=float)
    for i in range(len(w), len(series)):
        out.iloc[i] = np.dot(w, series.iloc[i-len(w):i].values)
    return out
def add_fracdiff_feature(df: pd.DataFrame, d: float = 0.4) -> pd.DataFrame:
    df = df.copy()
    df[f"fdiff_{d:.2f}"] = fracdiff(np.log(df["close"].replace(0, np.nan)), d)
    return df
