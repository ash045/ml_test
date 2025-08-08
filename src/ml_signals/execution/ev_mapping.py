import numpy as np, pandas as pd
def expected_value(prob: float, theta_bps: float, cost_bps: float) -> float:
    return (2*prob - 1.0) * theta_bps - cost_bps
def map_probs_to_actions(df: pd.DataFrame, prob_col: str, sigma_col: str, k: float,
                         fee_bps: float, spread_bps: float, margin_mult: float, hysteresis: dict,
                         liquidity_gate: dict = None, participation_caps: dict = None):
    persistence = hysteresis.get("persistence_bars", 2)
    cooldown = hysteresis.get("cooldown_minutes", 15)
    cost_bps = float(fee_bps) + float(spread_bps)
    ev = (2*df[prob_col] - 1.0) * (k * df[sigma_col] * 1e4) - cost_bps
    margin = margin_mult * cost_bps

    # Liquidity gate via trades vs ADT
    liquid_mask = pd.Series(True, index=df.index)
    if liquidity_gate and liquidity_gate.get("enabled", False):
        w = int(liquidity_gate.get("window", 60))
        trades = df.get("trades", pd.Series(0, index=df.index)).fillna(0)
        adt = trades.rolling(w, min_periods=w).mean().bfill()
        min_pct = float(liquidity_gate.get("min_trades_pct_of_adt", 0.5))
        liquid_mask = trades >= (min_pct * adt)

    actions, last_action, persist_counter, cooldown_counter = [], 0, 0, 0
    for i, e in enumerate(ev.fillna(-1e9).values):
        if cooldown_counter > 0 or not bool(liquid_mask.iloc[i]):
            actions.append(0); cooldown_counter = max(0, cooldown_counter-1); continue
        if e > max(0.0, margin):
            persist_counter += 1
            action = 1 if persist_counter >= persistence else 0
        elif e < -max(0.0, margin):
            persist_counter += 1
            action = -1 if persist_counter >= persistence else 0
        else:
            action = 0; persist_counter = 0
        if last_action != 0 and action == 0:
            cooldown_counter = cooldown
        actions.append(action); last_action = action

    df = df.copy()
    df["action_raw"] = actions

    # Participation scaling using trades ratio
    if participation_caps:
        trades = df.get("trades", pd.Series(0, index=df.index)).fillna(0)
        adt = trades.rolling(liquidity_gate.get("window", 60) if liquidity_gate else 60, min_periods=1).mean()
        liq_factor = (trades / adt.replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
        df["action"] = (df["action_raw"] * liq_factor).round()
    else:
        df["action"] = df["action_raw"]

    df["ev_bps"] = ev
    return df
