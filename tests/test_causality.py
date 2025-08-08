import pandas as pd, numpy as np
from ml_signals.features.core import add_core_features
def test_causal_rolling_does_not_use_future():
    ts = pd.date_range('2020-01-01', periods=300, freq='T', tz='UTC')
    df = pd.DataFrame({'timestamp': ts, 'open':1.0, 'high':1.0, 'low':1.0, 'close': range(300), 'volume':1.0, 'trades':1})
    out = add_core_features(df)
    assert out['vol_240'].isna().sum() >= 239
