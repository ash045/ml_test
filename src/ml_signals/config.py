"""
Configuration definitions for the realtime inference engine.

This module defines a set of dataclasses representing the various
configuration sections used by the inference pipeline as well as
helpers to load YAML configuration files. Unlike the training
configuration under ``utils/config.py`` in the existing code base, these
structures are purpose‑built for low‑latency online inference. They
provide explicit typing and sensible defaults to guard against
mis‑configured deployments.

The ``load_config`` function merges a user provided ``config.yaml``
file with an optional ``hyperparams.yaml``. The latter contains
hyper‑parameters for the ensemble and EV mapping. Users can
override any of the defaults via YAML.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    # If pyyaml is missing the import will raise a RuntimeError when
    # load_config is called.  We avoid raising here so that this
    # module can be imported in environments where the config is
    # supplied programmatically.
    yaml = None  # type: ignore


@dataclass
class CostsConfig:
    """Execution cost assumptions used to compute expected value.

    ``maker_fee_bps`` and ``taker_fee_bps`` represent round‑trip fees
    charged by the exchange. ``spread_bps`` captures the quoted bid/ask
    spread, while ``slippage_bps_sqrt`` models additional slippage
    proportional to the square root of trade size. ``execution_mode``
    selects whether orders are assumed to be maker or taker orders.
    """
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 4.0
    spread_bps: float = 2.0
    slippage_bps_sqrt: float = 0.5
    execution_mode: Literal["maker", "taker"] = "taker"

    def roundtrip_bps(self) -> float:
        """Return the combined fee and spread cost in basis points.

        This utility centralises the conversion from raw score to
        expected value so that the cost model can be changed in one
        place. Maker orders incur a lower fee than taker orders.
        """
        fee = self.taker_fee_bps if self.execution_mode == "taker" else self.maker_fee_bps
        return fee + self.spread_bps


@dataclass
class RiskConfig:
    """Risk limits used by the position sizer.

    ``vol_target_annual`` is the annualised volatility target the strategy
    attempts to achieve. ``max_gross_leverage`` caps leverage across
    long and short positions. ``adv_participation_cap`` limits how
    much of the average daily volume the strategy can take. ``position_unit``
    controls whether sizing is expressed in contract units or USD.
    """
    vol_target_annual: float = 0.25
    max_gross_leverage: float = 2.0
    adv_participation_cap: float = 0.15
    position_unit: Literal["units", "usd"] = "units"


@dataclass
class PolicyConfig:
    """Behavioural thresholds for mapping EV into trade actions.

    ``entry_ev_bps`` and ``exit_ev_bps`` define the minimum expected value
    required to enter or exit a position. ``persistence_bars`` enforces
    that the ensemble prediction remains consistent across multiple bars
    before taking a trade. ``cooldown_bars`` prevents immediate
    re‑entry after flipping positions. ``hysteresis_bps`` widens the
    exit threshold to avoid whipsawing.
    """
    entry_ev_bps: float = 3.0
    exit_ev_bps: float = 1.0
    persistence_bars: int = 3
    cooldown_bars: int = 2
    hysteresis_bps: float = 1.0


@dataclass
class EVMappingConfig:
    """Configuration for mapping ensemble scores to expected value.

    ``ev_per_unit_score_bps`` scales the score into basis points. ``cap_ev_bps``
    limits the absolute EV to avoid extreme position sizes.
    """
    ev_per_unit_score_bps: float = 10.0
    cap_ev_bps: float = 50.0


@dataclass
class KillSwitchConfig:
    """Settings for the kill switch mechanism.

    If ``enabled`` the inference engine will check for the existence of
    ``file_path`` before each decision. If the file exists the engine
    will liquidate any open positions and output no further trades.
    """
    enabled: bool = True
    file_path: str = "./killswitch.enabled"


@dataclass
class EnsembleMember:
    """Specification of a single ensemble component.

    ``type`` can be a builtin model (e.g. ``builtin:ema_crossover``) or
    the name of a registered artifact. ``weight`` controls the
    contribution of this model to the weighted average. Additional
    keyword arguments (``fast``, ``slow``, ``normalize``) are passed
    through to the model constructor. ``features`` lists the feature
    names required by an external model loaded from disk. ``calibrator``
    optionally specifies a calibrator to apply to the model's raw score
    prior to ensembling.
    """
    type: str
    weight: float = 1.0
    fast: Optional[int] = None
    slow: Optional[int] = None
    normalize: Optional[bool] = None
    features: Optional[List[str]] = None
    calibrator: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleConfig:
    """Configuration for the model ensemble.

    ``members`` is a list of ensemble member definitions. ``final_calibrator``
    defines a calibrator applied to the weighted ensemble score.
    """
    members: List[EnsembleMember] = dataclasses.field(default_factory=list)
    final_calibrator: Optional[Dict[str, Any]] = None


@dataclass
class RegistryConfig:
    """Location of the artifact registry used to store models and calibrators."""
    root: str = "./artifacts"


@dataclass
class DataConfig:
    """Column names and resampling frequency for realtime data."""
    timeframe: str = "1min"
    timestamp_col: str = "timestamp"
    price_col: str = "close"


@dataclass
class FeatureConfig:
    """Feature generation settings for realtime inference.

    ``ema_windows`` defines the spans used for exponential moving averages.
    ``fractional_diff_enabled`` toggles fractional differencing; ``fractional_diff_d``
    is the differencing order. ``context_bars`` controls the number of most
    recent bars kept in memory to compute features incrementally.
    """
    ema_windows: List[int] = dataclasses.field(default_factory=lambda: [3, 5])
    fractional_diff_enabled: bool = False
    fractional_diff_d: float = 0.4
    context_bars: int = 300


@dataclass
class InferenceConfig:
    """Top‑level configuration for the realtime inference engine."""
    symbol: str = "SOL-USD"
    ensemble: EnsembleConfig = dataclasses.field(default_factory=EnsembleConfig)
    ev: EVMappingConfig = dataclasses.field(default_factory=EVMappingConfig)
    costs: CostsConfig = dataclasses.field(default_factory=CostsConfig)
    risk: RiskConfig = dataclasses.field(default_factory=RiskConfig)
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)
    killswitch: KillSwitchConfig = dataclasses.field(default_factory=KillSwitchConfig)
    registry: RegistryConfig = dataclasses.field(default_factory=RegistryConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    features: FeatureConfig = dataclasses.field(default_factory=FeatureConfig)
    monitoring_path: str = "./monitoring/events.jsonl"
    warmup_bars: int = 120


def _dc_build(dc_type, data: Optional[Dict[str, Any]]):
    """Instantiate a dataclass from a raw mapping.

    If ``data`` is ``None`` a default instance is returned. For ``EnsembleConfig``
    the members list is recursively converted into ``EnsembleMember`` objects.
    Only keys present on the dataclass are retained, dropping any unknown
    fields from the YAML.
    """
    if data is None:
        return dc_type()
    if dc_type is EnsembleConfig:
        members = [EnsembleMember(**m) for m in data.get("members", [])]
        return EnsembleConfig(members=members, final_calibrator=data.get("final_calibrator"))
    # generic case: filter kwargs to dataclass fields
    fields = {f.name for f in dataclasses.fields(dc_type)}
    kwargs = {k: v for k, v in (data or {}).items() if k in fields}
    return dc_type(**kwargs)


def _flatten_features(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten nested feature configuration into a flat dict.

    The YAML schema nests fractional differencing settings under
    ``features.fractional_diff``. This helper extracts the enabled flag and
    differencing order into the flat attributes expected by ``FeatureConfig``.
    """
    if raw is None:
        return {}
    out: Dict[str, Any] = {}
    out["ema_windows"] = raw.get("ema_windows", [3, 5])
    fd = raw.get("fractional_diff", {}) or {}
    out["fractional_diff_enabled"] = bool(fd.get("enabled", False))
    out["fractional_diff_d"] = float(fd.get("d", 0.4))
    out["context_bars"] = int(raw.get("context_bars", 300))
    return out


def load_config(config_path: str, hyperparams_path: Optional[str] = None) -> InferenceConfig:
    """Load inference configuration from one or two YAML files.

    ``config_path`` should point to the main configuration YAML (for example
    the existing ``configs/config.yaml``). If inference settings are nested
    under a ``realtime_inference`` key in that file, they will be used
    automatically.  ``hyperparams_path`` can point to a separate hyper‑
    parameter YAML (such as ``configs/inference_hyperparams.yaml``). If
    omitted or equal to ``config_path``, the ensemble and EV settings are
    read from the nested section of the main config. Defaults are applied
    for any missing keys.

    Parameters
    ----------
    config_path : str
        Path to the primary configuration YAML. Required.
    hyperparams_path : Optional[str]
        Path to the hyper‑parameter YAML. If ``None`` or equal to
        ``config_path``, hyper‑parameters are loaded from within the
        ``realtime_inference`` section of the main config.

    Returns
    -------
    InferenceConfig
        A fully populated configuration object.
    """
    if yaml is None:
        raise RuntimeError("pyyaml is required for loading YAML configs")
    cfg_raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    # extract inference section if present
    infer = cfg_raw.get("realtime_inference", {}) if isinstance(cfg_raw, dict) else {}

    # determine where to read hyperparameters from
    if hyperparams_path and hyperparams_path != config_path:
        hp_raw = yaml.safe_load(Path(hyperparams_path).read_text(encoding="utf-8")) or {}
    else:
        # fallback: read from nested inference section
        hp_raw = infer

    # build ensemble and EV configs from hyperparams
    ensemble = _dc_build(EnsembleConfig, (hp_raw or {}).get("ensemble"))
    ev = _dc_build(EVMappingConfig, (hp_raw or {}).get("ev"))

    return InferenceConfig(
        symbol=infer.get("symbol", cfg_raw.get("symbol", "SOL-USD")),
        ensemble=ensemble,
        ev=ev,
        costs=_dc_build(CostsConfig, infer.get("costs", cfg_raw.get("costs"))),
        risk=_dc_build(RiskConfig, infer.get("risk", cfg_raw.get("risk"))),
        policy=_dc_build(PolicyConfig, infer.get("policy", cfg_raw.get("policy"))),
        killswitch=_dc_build(KillSwitchConfig, infer.get("killswitch", cfg_raw.get("killswitch"))),
        registry=_dc_build(RegistryConfig, infer.get("registry", cfg_raw.get("registry"))),
        data=_dc_build(DataConfig, infer.get("data", cfg_raw.get("data"))),
        features=_dc_build(FeatureConfig, _flatten_features(infer.get("features", cfg_raw.get("features")))),
        monitoring_path=infer.get("monitoring_path", cfg_raw.get("monitoring_path", "./monitoring/events.jsonl")),
        warmup_bars=int(infer.get("warmup_bars", cfg_raw.get("warmup_bars", 120))),
    )


def load_runtime_defaults(config_path: str) -> Dict[str, Any]:
    """Read optional runtime overrides from the configuration YAML.

    The ``runtime`` section of the YAML allows the user to specify
    defaults for the CLI arguments such as the CSV input path and
    output path. This helper extracts those values into a simple dict.
    """
    if yaml is None:
        raise RuntimeError("pyyaml is required for loading YAML configs")
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # Extract runtime configuration.  Prefer a nested ``realtime_inference.runtime``
    # section if present, otherwise fall back to the top‑level ``runtime``.  If
    # neither section exists default paths are returned.  This enables the
    # realtime inference script to share the same YAML as the training pipeline.
    runtime_top: Dict[str, Any] = {}
    if isinstance(raw, dict):
        # check nested structure
        infer = raw.get("realtime_inference", {})
        if isinstance(infer, dict) and isinstance(infer.get("runtime"), dict):
            runtime_top = infer["runtime"]  # type: ignore[assignment]
        elif isinstance(raw.get("runtime"), dict):
            runtime_top = raw["runtime"]  # type: ignore[assignment]
    # Provide defaults if keys are missing
    return {
        "csv_path": runtime_top.get("csv_path", "./data/SOL_USD.csv"),
        "out_path": runtime_top.get("out_path", "./out/inference_events.csv"),
    }