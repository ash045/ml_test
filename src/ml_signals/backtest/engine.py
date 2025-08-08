import pandas as pd
from .costs import estimate_costs_bps
def backtest_market(
    df: pd.DataFrame,
    action_col: str,
    ret_col: str,
    fee_bps: float,
    spread_bps: float,
    slippage_alpha: float,
    beta_trades: float = 0.5,
    adt_window: int = 240,
):
    trades = df.get("trades", pd.Series(0, index=df.index, dtype=float)).fillna(0)
    adt = trades.rolling(adt_window, min_periods=adt_window).mean().bfill()
    actions = df[action_col].astype(float).fillna(0).values
    rets = df[ret_col].shift(-1).fillna(0).values
    sigmas = df.get("sigma", pd.Series(0, index=df.index, dtype=float)).fillna(0).values
    pnl = []
    for i in range(len(df)):
        size = actions[i]; r = rets[i]; s = sigmas[i]
        cost_bps = estimate_costs_bps(fee_bps, spread_bps, slippage_alpha, s, size,
                                      trades=float(trades.iloc[i]), adt=float(adt.iloc[i]), beta_trades=beta_trades)
        net_ret = size * r - (cost_bps / 1e4)
        pnl.append(net_ret)
    out = df.copy()
    out["pnl"] = pnl
    out["equity"] = (1 + out["pnl"]).cumprod()
    return out
