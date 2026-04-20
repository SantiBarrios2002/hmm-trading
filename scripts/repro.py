"""Run an experiment YAML end-to-end and write artifacts to ``runs/<run_id>/``.

Usage:
    python scripts/repro.py configs/example_es_csv.yaml
    python scripts/repro.py my_config.yaml --force --runs-root custom_runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hft_hmm.config import ExperimentConfig
from hft_hmm.experiments.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a walk-forward experiment from a YAML config.",
    )
    parser.add_argument("config", type=Path, help="Path to the experiment YAML.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for run artifacts (default: runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing run directory with the same run_id.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    config = ExperimentConfig.from_yaml(args.config)
    artifacts = run_experiment(config, runs_root=args.runs_root, force=args.force)
    print(str(artifacts.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
