"""Top‑level package for all components of the ML signals framework.

This module exposes a selection of utilities, classes and subpackages
needed across the training, backtesting and realtime inference
pipelines.  It also ensures that the ``ml_signals`` namespace is
recognised as a proper Python package when using the ``src`` layout.

Users should import specific functionality from submodules (e.g.
``ml_signals.utils`` or ``ml_signals.config``) rather than relying on
star imports from this package.
"""

from importlib import import_module as _import_module

__all__ = []

# Ensure that core subpackages are importable when ``ml_signals`` is
# referenced at the package level.  This does not pollute the
# top‑level namespace but guarantees that attribute lookups such as
# ``ml_signals.deploy`` succeed.
for _subpkg in [
    "deploy",
    "ensemble",  # existing package from the original code base
    "ensemble_utils",  # realtime ensemble utilities defined here
    "fe_pipeline",
    "policy",
    "registry",
    "signal_models",
    "config",
]:
    try:
        _import_module(f"{__name__}.{_subpkg}")
    except Exception:
        # If any submodule fails to import (e.g. missing optional deps),
        # skip it silently to avoid breaking package initialisation.
        pass

del _import_module, _subpkg