"""
Weighted ensembling of trading signal models.

An ensemble combines multiple ``SignalModel`` instances into a single
score via a weighted average. Each model may have its own
``Calibrator`` applied to its raw score before aggregation. A final
calibrator can then be applied to the ensemble output. The weights
control the contribution of each member and are normalised by the sum
of absolute weights.

The ``EVMapper`` converts an ensemble score into an expected value
expressed in basis points. It subtracts estimated costs and caps the
absolute EV to avoid excessive leverage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .signal_models import (
    SignalModel,
    Calibrator,
    IdentityCalibrator,
    PlattCalibrator,
    EMACrossoverModel,
    SklearnModel,
)
from .registry import ArtifactRegistry
from .config import EVMappingConfig, CostsConfig


class WeightedEnsemble:
    """Combine multiple models into a single score via weighted averaging."""

    def __init__(self, registry: ArtifactRegistry, members: List[Dict[str, Any]], final_calibrator: Optional[Dict[str, Any]]) -> None:
        self.members: List[Tuple[SignalModel, Calibrator, float]] = []
        for m in members:
            weight = float(m.get("weight", 1.0))
            model = self._build_model(registry, m)
            calib = self._build_calibrator(registry, m.get("calibrator"))
            self.members.append((model, calib, weight))
        self.final_calibrator = self._build_calibrator(registry, final_calibrator) if final_calibrator else IdentityCalibrator()

    @staticmethod
    def _build_model(registry: ArtifactRegistry, spec: Dict[str, Any]) -> SignalModel:
        t = str(spec.get("type", "builtin:ema_crossover"))
        # handle builtin models
        if t == "builtin:ema_crossover":
            return EMACrossoverModel(fast=int(spec.get("fast", 12)), slow=int(spec.get("slow", 48)), normalize=bool(spec.get("normalize", True)))
        elif t.startswith("builtin:"):
            raise ValueError(f"Unknown builtin model: {t}")
        # treat as registry lookup otherwise
        resolved = registry.resolve(t)
        features = list(spec.get("features", []))
        return SklearnModel(path=resolved.get("path") or t, features=features)

    @staticmethod
    def _build_calibrator(registry: ArtifactRegistry, spec: Optional[Dict[str, Any]]) -> Calibrator:
        if spec is None:
            return IdentityCalibrator()
        t = str(spec.get("type", "builtin:identity"))
        if t == "builtin:identity":
            return IdentityCalibrator()
        if t == "builtin:platt":
            return PlattCalibrator(A=float(spec.get("A", -1.0)), B=float(spec.get("B", 0.0)))
        if t.startswith("builtin:"):
            raise ValueError(f"Unknown builtin calibrator: {t}")
        # external calibrators could be loaded here in the future
        return IdentityCalibrator()

    def predict(self, feature_row: pd.Series) -> float:
        """Compute the weighted ensemble score for a single feature row."""
        num = 0.0
        den = 0.0
        for model, calib, w in self.members:
            raw = model.predict_score(feature_row)
            cal = calib.calibrate(raw)
            num += float(cal) * w
            den += abs(w)
        ensemble_score = (num / den) if den > 0 else 0.0
        return float(self.final_calibrator.calibrate(ensemble_score))


class EVMapper:
    """Map ensemble scores into expected value (EV) in basis points."""

    def __init__(self, cfg: EVMappingConfig, costs: CostsConfig) -> None:
        self.cfg = cfg
        self.costs = costs

    def map_to_ev_bps(self, score: float) -> float:
        ev = self.cfg.ev_per_unit_score_bps * float(score)
        ev -= self.costs.roundtrip_bps()
        # cap the EV to avoid extremely large positions
        return float(np.clip(ev, -self.cfg.cap_ev_bps, self.cfg.cap_ev_bps))