import pandas as pd
from typing import Literal
def build_information_bars(df: pd.DataFrame, mode: Literal["dollar","volume","tick"], threshold: float) -> pd.DataFrame:
    """Emit a bar when cumulative notional/volume/trades exceed threshold."""
    rows, cum, start_idx = [], 0.0, 0
    for i, row in df.iterrows():
        if mode == "dollar":
            inc = float(row["close"]) * float(row.get("volume", 0.0))
        elif mode == "volume":
            inc = float(row.get("volume", 0.0))
        else:
            inc = float(row.get("trades", 0.0))
        cum += inc
        if cum >= threshold:
            chunk = df.iloc[start_idx:i+1]
            rows.append({
                "timestamp": chunk["timestamp"].iloc[-1],
                "open": chunk["open"].iloc[0],
                "high": chunk["high"].max(),
                "low": chunk["low"].min(),
                "close": chunk["close"].iloc[-1],
                "volume": chunk["volume"].sum(),
                "trades": chunk["trades"].sum() if "trades" in chunk.columns else None,
            })
            start_idx, cum = i+1, 0.0
    return pd.DataFrame(rows)
