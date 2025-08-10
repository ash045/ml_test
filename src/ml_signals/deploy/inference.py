"""
Backward‑compatibility shim for realtime inference.

The original project defined an empty ``inference.py`` under
``ml_signals/deploy``. To preserve that import path this module
re‑exports the new engine symbols. Users are encouraged to import
from ``ml_signals.deploy.engine`` directly.
"""

from .engine import RealTimeInferenceEngine, replay

__all__ = ["RealTimeInferenceEngine", "replay"]