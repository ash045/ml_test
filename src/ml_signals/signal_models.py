"""
Model and calibrator interfaces for realtime inference.

This module defines a small set of protocols and concrete classes
implementing lightweight trading signal models used by the realtime
inference engine. A ``SignalModel`` exposes a ``predict_score``
method that maps a row of features into an uncalibrated score. A
``Calibrator`` transforms this score into a calibrated value in the
range [‑1,1].

Two builtin models are provided:

* ``EMACrossoverModel`` computes fast and slow exponential moving
  averages and returns a normalised distance between them.
* ``SklearnModel`` wraps a pre‑trained scikit‑learn estimator loaded
  from disk via joblib. It supports both probabilistic and decision
  function APIs and falls back to a tanh of the raw prediction.

Calibrators include ``IdentityCalibrator`` (no change) and
``PlattCalibrator`` implementing Platt scaling. Additional
calibrators can be added by implementing the protocol.
"""
from __future__ import annotations

from typing import Any, List, Optional, Protocol
import math
import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


class SignalModel(Protocol):
    """Protocol for models that produce a raw trading score."""

    def predict_score(self, row: pd.Series) -> float:
        """Compute an uncalibrated score from a feature row."""
        ...


class Calibrator(Protocol):
    """Protocol for score calibrators."""

    def calibrate(self, score: float) -> float:
        """Map a raw score into a calibrated range."""
        ...


class IdentityCalibrator:
    """Return the score unchanged."""

    def calibrate(self, score: float) -> float:
        return float(score)


class PlattCalibrator:
    """Platt scaling calibrator.

    Applies a logistic transform ``1/(1+exp(A*score+B))`` and then
    rescales to [‑1,1] by the transformation ``2p‑1``. Parameters A and B
    can be learned during training and passed through the registry.
    """

    def __init__(self, A: float = -1.0, B: float = 0.0) -> None:
        self.A = float(A)
        self.B = float(B)

    def calibrate(self, score: float) -> float:
        z = self.A * float(score) + self.B
        p = 1.0 / (1.0 + math.exp(z))
        return float(2 * p - 1.0)


class EMACrossoverModel:
    """Compute the normalised distance between fast and slow EMAs.

    If ``normalize`` is true the difference is divided by the current
    price and passed through a hyperbolic tangent to bound the output.
    This model updates its internal state incrementally, making it
    efficient for streaming applications.
    """

    def __init__(self, fast: int = 12, slow: int = 48, normalize: bool = True) -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("EMA params must satisfy: 0 < fast < slow")
        self.fast = int(fast)
        self.slow = int(slow)
        self.normalize = bool(normalize)
        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None

    def _update(self, price: float) -> None:
        # exponential weighting factors
        af = 2 / (self.fast + 1)
        as_ = 2 / (self.slow + 1)
        self.ema_fast = price if self.ema_fast is None else (1 - af) * self.ema_fast + af * price
        self.ema_slow = price if self.ema_slow is None else (1 - as_) * self.ema_slow + as_ * price

    def predict_score(self, row: pd.Series) -> float:
        price = float(row["close"])  # requires 'close' to be present
        self._update(price)
        if self.ema_fast is None or self.ema_slow is None:
            return 0.0
        dist = float(self.ema_fast - self.ema_slow)
        if not self.normalize:
            return dist
        rel = dist / max(1e-8, price)
        return float(np.tanh(5.0 * rel))


class SklearnModel:
    """Wrap a pre‑trained scikit‑learn model for score prediction."""

    def __init__(self, path: str, features: List[str]) -> None:
        if joblib is None:
            raise RuntimeError("joblib not available; cannot load sklearn model")
        self.model = joblib.load(path)
        self.features = features

    def predict_score(self, row: pd.Series) -> float:
        # reorder features into the expected order and convert to a 2D array
        x = row.reindex(self.features).to_numpy(dtype=float)[None, :]
        # prefer probability outputs if available (binary classification)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            if len(proba) == 2:
                return float(proba[1] - proba[0])
        # fall back to decision_function if present
        if hasattr(self.model, "decision_function"):
            return float(np.ravel(self.model.decision_function(x))[0])
        # else use raw prediction and squash into [‑1,1]
        pred = float(self.model.predict(x)[0])
        return float(np.tanh(pred))