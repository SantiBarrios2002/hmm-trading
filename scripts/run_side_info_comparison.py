"""Run a Gate H side-information comparison from a YAML config.

Usage:
    python scripts/run_side_info_comparison.py configs/example_es_side_info_comparison.yaml
    python scripts/run_side_info_comparison.py my_config.yaml --force --runs-root custom_runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hft_hmm.experiments.side_info_comparison import (
    SideInfoComparisonConfig,
    run_side_info_comparison,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a side-information IOHMM comparison from a YAML config.",
    )
    parser.add_argument("config", type=Path, help="Path to the comparison YAML.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for comparison artifacts (default: runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing comparison directory with the same id.",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    config = SideInfoComparisonConfig.from_yaml(args.config)
    artifacts = run_side_info_comparison(config, runs_root=args.runs_root, force=args.force)
    print(str(artifacts.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
