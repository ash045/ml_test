"""
Incremental feature generation for realtime inference.

The ``RealtimeFeaturePipeline`` maintains a rolling buffer of recent
minute bars and updates core features, fractional differencing and
denoised prices on the fly. It wraps existing feature engineering
functions from the ``ml_signals`` package and ensures that only the
most recent ``context_bars`` are kept in memory to avoid quadratic
complexity.

On each new bar the pipeline resamples the buffer to the target
timeframe (typically 1 minute), computes core features causally using
``add_core_features``, optionally applies fractional differencing and
denoising, and returns the last row. A ``warm`` flag indicates when
enough history has been accumulated (``warmup_bars``) to produce
meaningful predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .config import DataConfig, FeatureConfig
# import existing pipeline components
from .bars.time_bars import resample_time_bars
from .features.core import add_core_features
from .features.denoise import apply_denoise
from .features.fdiff import add_fracdiff_feature


@dataclass
class RealtimeFeaturePipeline:
    """Stateful wrapper around feature engineering functions for streaming bars."""

    data_cfg: DataConfig
    feat_cfg: FeatureConfig
    warmup_bars: int

    def __post_init__(self) -> None:
        # buffer holds recent raw bars; index is implicit order of arrival
        self.buffer: Optional[pd.DataFrame] = None

    def _ensure_df(self, row: pd.Series) -> None:
        # initialise the buffer on first call
        if self.buffer is None:
            self.buffer = pd.DataFrame([row])
        else:
            # append new row preserving order
            self.buffer.loc[len(self.buffer)] = row

    def update(self, row: pd.Series) -> pd.Series:
        """Consume a new minute bar and return the latest feature row.

        The returned row includes a ``warm`` flag set to 1 when
        ``warmup_bars`` bars have been accumulated after resampling.
        """
        self._ensure_df(row)
        assert self.buffer is not None
        # maintain only recent context to limit memory and computation
        ctx = self.buffer.tail(max(self.feat_cfg.context_bars, self.warmup_bars)).copy()
        # resample to target timeframe and fill missing bars using causal rules
        ctx = resample_time_bars(ctx, self.data_cfg.timeframe, ts_col=self.data_cfg.timestamp_col)
        # compute core features causally; add_core_features expects time index
        ctx = add_core_features(ctx.set_index(self.data_cfg.timestamp_col)).reset_index(drop=False)
        # optional fractional differencing
        if self.feat_cfg.fractional_diff_enabled:
            ctx = add_fracdiff_feature(ctx, self.feat_cfg.fractional_diff_d)
        # optional denoising via causal EMAs
        ctx = apply_denoise(ctx, tuple(self.feat_cfg.ema_windows))
        # mark warm flag based on number of bars after resampling
        last = ctx.iloc[-1].copy()
        last["warm"] = int(len(ctx) >= self.warmup_bars)
        return last