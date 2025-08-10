#!/usr/bin/env python3
"""
Command line entry point for running realtime inference on historical data.

This script loads the realtime inference configuration and hyper‑parameters
from YAML files, optionally overrides the CSV input and output paths, and
feeds the data through the ``RealTimeInferenceEngine`` defined in
``ml_signals.deploy.engine``. The resulting events are written to the
specified output file and a summary of the last few rows is printed to
stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ml_signals.config import load_config, load_runtime_defaults
from ml_signals.deploy.engine import replay


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the realtime inference replay.

    The only required parameter is ``--config``, which should point to the
    primary configuration file used throughout the pipeline (e.g.
    ``configs/config.yaml``).  Optional ``--csv`` and ``--out`` flags can
    override the runtime paths specified in the YAML.  We intentionally
    omit a ``--hyperparams`` flag here because hyper‑parameters are
    automatically sourced from the same config file when using the
    ``realtime_inference`` section.
    """
    p = argparse.ArgumentParser(description="Run realtime inference replay over an OHLCV CSV.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the unified configuration YAML. If omitted, defaults to "
            "'configs/config.yaml'."
        ),
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to OHLCV CSV (UTC timestamps). Overrides runtime.csv_path.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV for actions/metrics. Overrides runtime.out_path.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the realtime inference CLI.

    This function wires together configuration loading, runtime override
    resolution and invocation of the ``replay`` helper.  Only the path
    to the unified config file is needed; all other settings are
    discovered automatically.
    """
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    # Determine the configuration file path.  Prefer the user‑supplied
    # --config; fallback to the standard 'configs/config.yaml'.  We no
    # longer accept separate hyper‑parameter files because the nested
    # ``realtime_inference`` section encapsulates them.
    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "config.yaml")

    # Load the inference configuration.  Passing ``None`` for the hyperparams
    # path will cause ``load_config`` to read the ensemble and EV settings
    # from the same YAML under ``realtime_inference``.
    cfg = load_config(str(cfg_path), None)

    # Extract runtime defaults (CSV input and output paths).  Use nested
    # overrides if present in the YAML.  Command line arguments take
    # precedence.
    rt = load_runtime_defaults(str(cfg_path))
    csv_path = Path(args.csv) if args.csv else Path(rt["csv_path"]).resolve()
    out_path = Path(args.out) if args.out else Path(rt["out_path"]).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = replay(csv_path, cfg, out_path)
    # Print a sample of the output for quick inspection.
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()