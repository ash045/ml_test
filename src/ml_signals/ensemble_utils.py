"""Weighted ensemble and EV mapping for realtime inference.

This module defines two classes used during the realtime inference
process:

* ``WeightedEnsemble`` aggregates the predictions from a collection of
  signal models, each optionally followed by a calibrator, into a
  single ensemble score using a weighted average.
* ``EVMapper`` converts the ensemble score into an expected value (EV)
  expressed in basis points by scaling, capping and subtracting
  estimated trading costs.

These utilities live in a separate module rather than ``ensemble`` to
avoid clashing with the existing ``ml_signals.ensemble`` package in the
original code base.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .registry import ArtifactRegistry
from .signal_models import Calibrator, SignalModel, IdentityCalibrator, PlattCalibrator, EMACrossoverModel, SklearnModel
from .config import CostsConfig, EVMappingConfig


def _load_model(registry: ArtifactRegistry, spec: Dict[str, Any]) -> Tuple[SignalModel, Optional[Calibrator], float]:
    """Instantiate a model and optional calibrator from a member specification.

    Parameters
    ----------
    registry : ArtifactRegistry
        The artifact registry used to resolve external models and calibrators.
    spec : Dict[str, Any]
        The ensemble member specification loaded from the configuration.

    Returns
    -------
    Tuple[SignalModel, Optional[Calibrator], float]
        A tuple containing the model instance, an optional calibrator and
        the ensemble weight.
    """
    weight = float(spec.get("weight", 1.0))
    model_type: str = spec.get("type", "")
    # handle builtin models
    if model_type.startswith("builtin:"):
        builtin = model_type.split(":", 1)[1]
        if builtin == "ema_crossover":
            fast = int(spec.get("fast", 12))
            slow = int(spec.get("slow", 48))
            normalize = bool(spec.get("normalize", True))
            model = EMACrossoverModel(fast=fast, slow=slow, normalize=normalize)
        else:
            raise ValueError(f"Unknown builtin model '{builtin}'")
        calibrator_spec = spec.get("calibrator")
        calibrator: Optional[Calibrator]
        if calibrator_spec:
            ctype = calibrator_spec.get("type", "")
            if ctype == "builtin:identity":
                calibrator = IdentityCalibrator()
            elif ctype == "builtin:platt":
                calibrator = PlattCalibrator()
            else:
                raise ValueError(f"Unknown builtin calibrator '{ctype}'")
        else:
            calibrator = None
        return model, calibrator, weight
    # handle externally registered models
    model_name = model_type
    metadata, model_path = registry.resolve(model_name)
    features = spec.get("features") or metadata.get("features") or []
    model = SklearnModel(model_path, features)
    # calibrator if specified
    calibrator_spec = spec.get("calibrator") or {}
    calibrator: Optional[Calibrator] = None
    if calibrator_spec:
        ctype = calibrator_spec.get("type", "")
        if ctype.startswith("builtin:"):
            cbuiltin = ctype.split(":", 1)[1]
            if cbuiltin == "identity":
                calibrator = IdentityCalibrator()
            elif cbuiltin == "platt":
                calibrator = PlattCalibrator()
            else:
                raise ValueError(f"Unknown builtin calibrator '{cbuiltin}'")
        else:
            # external calibrator: assume saved via joblib with metadata specifying
            # the type so we can select the right wrapper; here we support only
            # logistic regression calibration (Platt)
            _, cal_path = registry.resolve(ctype)
            calibrator = PlattCalibrator.from_file(cal_path)
    return model, calibrator, weight


class WeightedEnsemble:
    """Aggregate multiple model predictions into a single score.

    Each ensemble member returns a raw score in [0, 1].  Optional
    calibrators can be applied to each model before weighting.  A final
    calibrator can be applied to the aggregated score to correct bias.
    """

    def __init__(self, registry: ArtifactRegistry, members: Iterable[Dict[str, Any]], final_calibrator: Optional[Dict[str, Any]]) -> None:
        self.members: List[Tuple[SignalModel, Optional[Calibrator], float]] = []
        for spec in members:
            self.members.append(_load_model(registry, spec))
        # prepare final calibrator
        self.final_calibrator: Optional[Calibrator] = None
        if final_calibrator:
            ctype = final_calibrator.get("type", "")
            if ctype == "builtin:identity":
                self.final_calibrator = IdentityCalibrator()
            elif ctype == "builtin:platt":
                self.final_calibrator = PlattCalibrator()
            else:
                raise ValueError(f"Unknown final calibrator '{ctype}'")

    def predict(self, X: Dict[str, Any]) -> float:
        """Compute a weighted average of model predictions.

        Parameters
        ----------
        X : Dict[str, Any]
            A mapping of feature names to values.  For external models this
            will be a ``dict`` keyed by feature names; for builtin models
            the input is a pandas ``Series``.  When a builtin model is
            encountered, we pass the entire ``X`` into its
            ``predict_score`` method.  External models expect a feature
            ``Series`` and will extract their own required features.

        Returns
        -------
        float
            The ensemble prediction.  If calibrators are configured they
            are applied to each model and to the final score.
        """
        if not self.members:
            raise RuntimeError("Ensemble has no members configured")
        scores: List[float] = []
        weights: List[float] = []
        for model, calibrator, w in self.members:
            # Builtin models implement ``predict_score``; external models
            # (e.g. sklearn) also provide ``predict_score``.  We avoid
            # calling a non‑existent ``predict`` attribute on builtin models.
            if hasattr(model, "predict_score"):
                s = model.predict_score(X)  # type: ignore[arg-type]
            else:
                # fallback for models that expose a ``predict`` API
                s = model.predict(X)  # type: ignore[call-arg]
            if calibrator:
                # calibrators implement ``calibrate`` rather than ``predict``
                s = calibrator.calibrate(float(s))
            scores.append(float(s))
            weights.append(float(w))
        # compute weighted average
        score = float(np.average(scores, weights=weights))
        # apply final calibrator if present
        if self.final_calibrator:
            score = float(self.final_calibrator.calibrate(score))  # type: ignore[call-arg]
        return score


class EVMapper:
    """Map a normalised ensemble score into expected return in basis points.

    The mapping uses a linear scaling parameterised by ``ev_per_unit_score_bps`` and
    caps the absolute EV at ``cap_ev_bps`` to control position sizes.  A
    round‑trip trading cost (fees and spread) is subtracted to obtain the
    net EV.
    """

    def __init__(self, ev_cfg: EVMappingConfig, costs_cfg: CostsConfig) -> None:
        self.ev_per_unit_score_bps: float = ev_cfg.ev_per_unit_score_bps
        self.cap_ev_bps: float = ev_cfg.cap_ev_bps
        self.costs_cfg = costs_cfg

    def map(self, score: float) -> float:
        """Convert a score to a net expected value in bps.

        Parameters
        ----------
        score : float
            Normalised score in the range [-1, 1] or [0, 1] depending on
            calibrators.

        Returns
        -------
        float
            Net expected value in basis points after applying scaling,
            capping and subtracting trading costs.
        """
        ev = score * self.ev_per_unit_score_bps
        # cap EV to avoid extreme position sizes
        ev = float(np.clip(ev, -self.cap_ev_bps, self.cap_ev_bps))
        # subtract round‑trip cost
        ev -= self.costs_cfg.roundtrip_bps()
        return ev