"""
Realtime inference engine.

This module implements an event‑driven engine that consumes minute
bars, updates a rolling feature store, applies a weighted ensemble of
signal models, maps the resulting score into expected value (EV),
sizes positions and decides on discrete trade actions. It also
supports a simple file‑based kill switch and writes detailed events to
a monitoring log. A convenience ``replay`` function reads a CSV
containing historical bars and runs the engine to produce a DataFrame
of inference events.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time
import dataclasses
import json
import numpy as np
import pandas as pd

from ..config import InferenceConfig
from ..registry import ArtifactRegistry
from ..fe_pipeline import RealtimeFeaturePipeline
# Import the realtime ensemble utilities from our standalone module.  We
# avoid importing from ``ml_signals.ensemble`` because that name
# conflicts with the existing ensemble package in the training code base.
from ..ensemble_utils import WeightedEnsemble, EVMapper
from ..policy import PositionSizer, PolicyLayer


class RealTimeInferenceEngine:
    """Stateful inference engine for minute‑bar trading signals."""

    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        # initialise subsystems
        self.registry = ArtifactRegistry(cfg.registry.root)
        self.features = RealtimeFeaturePipeline(cfg.data, cfg.features, cfg.warmup_bars)
        # convert EnsembleMember dataclasses into plain dicts
        member_dicts = [dataclasses.asdict(m) for m in cfg.ensemble.members]  # type: ignore[name-defined]
        self.ensemble = WeightedEnsemble(self.registry, member_dicts, cfg.ensemble.final_calibrator)
        self.mapper = EVMapper(cfg.ev, cfg.costs)
        self.sizer = PositionSizer(cfg.risk)
        self.policy = PolicyLayer(cfg.policy)
        # monitoring
        self.monitor_path = Path(cfg.monitoring_path)
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)

    def _kill_switch_engaged(self) -> bool:
        """Return True if the kill switch file exists and is enabled."""
        if not self.cfg.killswitch.enabled:
            return False
        return Path(self.cfg.killswitch.file_path).exists()

    def _log_event(self, event: Dict[str, Any]) -> None:
        """Append an inference event to the monitoring log as JSON.

        Events may contain types such as pandas ``Timestamp`` that are not
        directly serialisable by the standard ``json`` module.  We pass
        ``default=str`` to ``json.dumps`` to coerce any unsupported
        objects into strings.  This prevents ``TypeError`` on
        non‑serialisable types while preserving the underlying value in a
        human‑readable form.
        """
        with self.monitor_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def on_bar(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single bar and return a dictionary describing the decision."""
        t0 = time.perf_counter()
        # update features and check if sufficient history has accumulated
        feat = self.features.update(row)
        warm = bool(feat.get("warm", 0))
        # current price used for sizing; fall back to raw bar if missing
        price = float(feat.get("close", row.get("close", 0.0)))
        usd_vol = float(row.get("usd_volume", row.get("volume", 0.0))) * price

        # handle kill switch: immediately flatten position and skip prediction
        if self._kill_switch_engaged():
            if self.policy.current_pos > 0:
                action = "SELL"
            elif self.policy.current_pos < 0:
                action = "BUY"
            else:
                action = "HOLD"
            delta = -self.policy.current_pos
            self.policy.current_pos = 0.0
            latency = time.perf_counter() - t0
            evt = {
                "ts": row[self.cfg.data.timestamp_col],
                "action": action,
                "delta_units": float(delta),
                "pos_units": float(self.policy.current_pos),
                "ev_bps": 0.0,
                "score": 0.0,
                "latency_ms": int(latency * 1000),
                "killswitch": True,
            }
            self._log_event(evt)
            return evt

        # compute model ensemble score and expected value when warm
        score = self.ensemble.predict(feat) if warm else 0.0
        # map the ensemble score into expected value (bps).  The EVMapper
        # class defines a ``map`` method rather than ``map_to_ev_bps`` to
        # avoid naming conflicts with other mapping utilities.
        ev_bps = self.mapper.map(score) if warm else 0.0
        desired_pos = self.sizer.desired_position(feat, ev_bps, price, usd_vol) if warm else 0.0
        action, delta = self.policy.decide(ev_bps, desired_pos) if warm else ("HOLD", 0.0)
        latency = time.perf_counter() - t0
        evt = {
            "ts": row[self.cfg.data.timestamp_col],
            "price": price,
            "action": action,
            "delta_units": float(delta),
            "pos_units": float(self.policy.current_pos),
            "score": float(score),
            "ev_bps": float(ev_bps),
            "latency_ms": int(latency * 1000),
            "killswitch": False,
        }
        self._log_event(evt)
        return evt


# ---- Data ingestion helpers

def _infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Guess column names for OHLCV data by matching common variants."""
    cols = {c.lower(): c for c in df.columns}
    mapping = {
        "timestamp": cols.get("timestamp") or cols.get("time") or cols.get("date"),
        "open": cols.get("open") or cols.get("o"),
        "high": cols.get("high") or cols.get("h"),
        "low": cols.get("low") or cols.get("l"),
        "close": cols.get("close") or cols.get("c"),
        "volume": cols.get("volume") or cols.get("v"),
        "trades": cols.get("trades") or cols.get("n_trades") or cols.get("trade_count"),
    }
    missing = [k for k, v in mapping.items() if k in ("timestamp", "open", "high", "low", "close", "volume") and v is None]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}; found={list(df.columns)}")
    return mapping


def load_csv_iter(path: Union[str, Path]) -> Iterable[pd.Series]:
    """Yield each row of a CSV as a Series after normalising column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    mapping = _infer_columns(df)
    ts = pd.to_datetime(df[mapping["timestamp"]], utc=True)
    df_norm = pd.DataFrame({
        "timestamp": ts,
        "open": df[mapping["open"]].astype(float),
        "high": df[mapping["high"]].astype(float),
        "low": df[mapping["low"]].astype(float),
        "close": df[mapping["close"]].astype(float),
        "volume": df[mapping["volume"]].astype(float),
    })
    if mapping.get("trades") is not None:
        df_norm["trades"] = df[mapping["trades"]].astype(float)
    else:
        df_norm["trades"] = np.nan
    df_norm.sort_values("timestamp", inplace=True)
    df_norm.reset_index(drop=True, inplace=True)
    for _, row in df_norm.iterrows():
        yield row


def replay(csv_path: Union[str, Path], cfg: InferenceConfig, out_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Replay historical data through the inference engine.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV containing minute‑bar data.
    cfg : InferenceConfig
        Loaded inference configuration.
    out_path : str or Path, optional
        If provided, writes the resulting events to a CSV at this path.

    Returns
    -------
    DataFrame
        A DataFrame of inference events generated by the engine.
    """
    engine = RealTimeInferenceEngine(cfg)
    rows: List[Dict[str, Any]] = []
    for row in load_csv_iter(csv_path):
        evt = engine.on_bar(row)
        rows.append(evt)
    df_out = pd.DataFrame(rows)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
    return df_out