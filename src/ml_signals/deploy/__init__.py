"""
Deployment subpackage for realtime inference.

The realtime engine and its helpers live under this namespace. Use
``RealTimeInferenceEngine`` to embed inference into a streaming
application or call ``replay`` to simulate inference on historical data.
"""

from .engine import RealTimeInferenceEngine, replay  # noqa: F401

__all__ = ["RealTimeInferenceEngine", "replay"]