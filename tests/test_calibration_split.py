import numpy as np
from sklearn.isotonic import IsotonicRegression
def test_isotonic_regression_shapes():
    rng = np.random.RandomState(0)
    probs = rng.rand(100)
    y = (probs > 0.5).astype(int)
    iso = IsotonicRegression(out_of_bounds='clip').fit(probs, y)
    v = iso.predict([0.1])[0]
    assert 0.0 <= v <= 1.0
