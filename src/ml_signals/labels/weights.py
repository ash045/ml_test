import pandas as pd
from .triple_barrier import compute_concurrency_weights
def sample_weights(t_end: pd.Series) -> pd.Series:
    return compute_concurrency_weights(t_end)
