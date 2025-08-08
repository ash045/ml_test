from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np

def fit_isotonic(probs, y, sample_weight=None):
    """Weighted isotonic calibration (monotonic)."""
    probs = np.asarray(probs).ravel()
    y = np.asarray(y).ravel()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, y, sample_weight=sample_weight)
    return iso

def fit_platt(probs, y, sample_weight=None):
    """Weighted Platt scaling via logistic regression."""
    probs = np.asarray(probs).ravel()
    y = np.asarray(y).ravel()
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(probs.reshape(-1, 1), y, sample_weight=sample_weight)
    return lr
