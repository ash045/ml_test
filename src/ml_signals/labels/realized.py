import numpy as np
import pandas as pd

def add_event_realized_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute realized returns using the PRIMARY event end column 't_end'.
    Produces: ev_ret, ev_ret_short, ev_bps
    """
    return add_event_realized_returns_from(df, t_end_col="t_end", price_col=price_col, suffix=None)

def add_event_realized_returns_from(
    df: pd.DataFrame,
    t_end_col: str,
    price_col: str = "close",
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    Compute realized returns from entry (row index) to event end at `t_end_col`.

    Writes columns:
      - ev_ret{suffix}
      - ev_ret_short{suffix}
      - ev_bps{suffix}

    Notes:
      - Leaves NaN where inputs are missing or t_end is invalid/out of range.
      - Does NOT leak: uses only forward price at the specified event end index.
    """
    out = df.copy()
    if t_end_col not in out.columns or price_col not in out.columns:
        return out

    n = len(out)

    # Convert to numeric safely (no .fillna on numpy)
    te_raw = pd.to_numeric(out[t_end_col], errors="coerce").to_numpy()
    # te = -1 for invalid; keep as -1 so we can mask those rows out
    te = np.where(np.isfinite(te_raw), te_raw.astype(np.int64), -1)

    # We'll use a clipped version only for indexing, but require te_ok in the mask
    te_clip = np.clip(te, 0, max(0, n - 1))

    # Price array
    p = pd.to_numeric(out[price_col], errors="coerce").to_numpy(dtype="float64")

    # Valid rows: current price finite AND t_end is a valid index AND future price finite
    te_ok = (te >= 0) & (te < n)
    valid = np.isfinite(p) & te_ok & np.isfinite(p[te_clip])

    # Realized long return
    r_long = np.full(n, np.nan, dtype="float64")
    r_long[valid] = (p[te_clip[valid]] / p[valid]) - 1.0

    suf = "" if not suffix else str(suffix)
    out[f"ev_ret{suf}"] = r_long
    out[f"ev_ret_short{suf}"] = -r_long
    out[f"ev_bps{suf}"] = r_long * 1e4

    return out
