import pandas as pd
from .metrics import sharpe, max_drawdown
def summary(df: pd.DataFrame):
    s = sharpe(df["pnl"].fillna(0))
    mdd = max_drawdown(df["equity"].fillna(1.0))
    return {"sharpe": s, "max_drawdown": mdd, "turnover_proxy": float(df.get("action", pd.Series(0)).diff().abs().sum())}
