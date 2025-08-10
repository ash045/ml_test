"""
Position sizing and policy logic for realtime inference.

``PositionSizer`` uses an exponentially weighted moving variance to
estimate realised volatility from minute returns and scales position
sizes accordingly. It also enforces participation caps based on
average daily volume. Position sizes are expressed in units (or USD
depending on ``RiskConfig.position_unit``) and capped by the
``max_gross_leverage``.

``PolicyLayer`` converts expected value (EV) into discrete trade
actions (BUY/SELL/HOLD) subject to persistence and cooldown rules.
It keeps track of recent trade signals to avoid reacting to transient
noise and maintains an internal notion of the current position size.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Tuple, Optional
import math
import numpy as np
import pandas as pd

from .config import RiskConfig, PolicyConfig


class PositionSizer:
    """Compute desired position size based on risk settings and EV."""

    def __init__(self, risk: RiskConfig) -> None:
        self.risk = risk
        # exponential weighting factor for variance estimate: 60 bars ~ 1 hour
        self.ewm_var: Optional[float] = None
        self.alpha = 2.0 / (60.0 + 1.0)

    def _update_vol(self, ret1: float) -> float:
        # update exponentially weighted variance of 1‑bar returns
        r = float(ret1)
        v = r * r
        self.ewm_var = v if self.ewm_var is None else (1 - self.alpha) * self.ewm_var + self.alpha * v
        # convert to annualised volatility assuming 60*24*365 minute bars per year
        ann_vol = math.sqrt(max(self.ewm_var, 1e-12)) * math.sqrt(60 * 24 * 365)
        return float(ann_vol)

    def desired_position(self, feature_row: pd.Series, ev_bps: float, price: float, bar_usd_volume: float) -> float:
        """Compute the desired position size based on EV and realised volatility."""
        # update volatility estimate using the most recent 1‑bar return
        ann_vol = self._update_vol(float(feature_row.get("ret_1", feature_row.get("ret1", 0.0))))
        k = max(ann_vol, 1e-9)
        # convert EV in bps into leverage: EV/k yields approximate Sharpe
        raw_lev = (abs(ev_bps) / 10000.0) / k
        raw_lev = min(raw_lev, self.risk.max_gross_leverage)
        # cap size by participation limits
        max_usd = self.risk.adv_participation_cap * bar_usd_volume
        max_units = max_usd / max(price, 1e-9)
        # convert leverage into units (1x leverage = 1 unit) and clamp
        units = raw_lev * 1.0
        units = float(np.clip(units, -max_units, max_units))
        # sign of position matches EV sign
        return math.copysign(units, ev_bps)


class PolicyLayer:
    """Map expected value into discrete trade actions with persistence and cooldown."""

    def __init__(self, policy: PolicyConfig) -> None:
        self.p = policy
        self.sign_history: Deque[int] = deque(maxlen=max(1, policy.persistence_bars))
        self.cooldown_left = 0
        self.current_pos = 0.0

    def _persist_ok(self, sign: int) -> bool:
        # require sign_history to be full and consistent
        if len(self.sign_history) < self.sign_history.maxlen:
            return False
        return all(s == sign for s in self.sign_history)

    def decide(self, ev_bps: float, desired_pos: float) -> Tuple[str, float]:
        """Return (action, delta_units) given the EV and desired position."""
        sign = 0
        # determine directional signal based on exit threshold
        if ev_bps > self.p.exit_ev_bps:
            sign = 1
        elif ev_bps < -self.p.exit_ev_bps:
            sign = -1
        # record history for persistence check
        self.sign_history.append(sign)

        # handle cooldown: gradually close position to zero
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            target = 0.0
            action = "HOLD" if abs(self.current_pos - target) < 1e-9 else ("SELL" if self.current_pos > target else "BUY")
            delta = target - self.current_pos
            self.current_pos = target
            return action, float(delta)

        abs_ev = abs(ev_bps)
        # when flat: need to exceed entry threshold and persist
        if self.current_pos == 0.0:
            if abs_ev < self.p.entry_ev_bps:
                return "HOLD", 0.0
            if not self._persist_ok(sign):
                return "HOLD", 0.0
            # open new position at desired size
            target = desired_pos
            action = "BUY" if target > 0 else "SELL"
            delta = target - self.current_pos
            self.current_pos = target
            return action, float(delta)
        else:
            # already in a position: require EV to exceed exit+ hysteresis to adjust
            if abs_ev < (self.p.exit_ev_bps + self.p.hysteresis_bps):
                return "HOLD", 0.0
            target = desired_pos
            # if reversing sign, enter cooldown period
            if np.sign(target) != np.sign(self.current_pos):
                self.cooldown_left = self.p.cooldown_bars
            action = "BUY" if target > self.current_pos else "SELL"
            delta = target - self.current_pos
            self.current_pos = target
            return action, float(delta)