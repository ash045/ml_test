def estimate_costs_bps(fee_bps: float, spread_bps: float, slippage_alpha: float, sigma_1m: float, size: float,
                       trades: float = None, adt: float = None, beta_trades: float = 0.5) -> float:
    """Dynamic costs:
    - Base = fee + spread
    - Slippage ~ alpha * sigma * sqrt(|size|) * liquidity_penalty
    - liquidity_penalty increases when trades < ADT
    """
    base = (fee_bps or 0.0) + (spread_bps or 0.0)
    liq_pen = 1.0
    if trades is not None and adt is not None and adt > 0:
        ratio = trades / adt if adt > 0 else 0.0
        if ratio <= 0:
            liq_pen = 1.0 + beta_trades
        else:
            # penalty rises when ratio < 1
            liq_pen = 1.0 + beta_trades * max((1.0 / ratio)**0.5 - 1.0, 0.0)
    impact_bps = (slippage_alpha or 0.0) * (sigma_1m or 0.0) * (abs(size) ** 0.5) * 1e4 * liq_pen
    return base + impact_bps
