"""Run a Gate H side-information comparison from a YAML config.

Usage:
    python scripts/run_side_info_comparison.py configs/example_es_side_info_comparison.yaml
    python scripts/run_side_info_comparison.py my_config.yaml --force --runs-root custom_runs
    python scripts/run_side_info_comparison.py my_config.yaml --checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hft_hmm.experiments.side_info_comparison import (
    SideInfoComparisonConfig,
    comparison_id,
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
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help=(
            "Save per-stage checkpoints to runs_root/<cmp_id>.checkpoints/ so a "
            "crashed run can resume by re-invoking the same command. Removed on "
            "success."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Explicit checkpoint directory (overrides --checkpoint default of "
            "runs_root/<cmp_id>.checkpoints/)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    config = SideInfoComparisonConfig.from_yaml(args.config)
    checkpoint_dir: Path | None = args.checkpoint_dir
    if checkpoint_dir is None and args.checkpoint:
        checkpoint_dir = args.runs_root / f"{comparison_id(config)}.checkpoints"
    artifacts = run_side_info_comparison(
        config,
        runs_root=args.runs_root,
        force=args.force,
        checkpoint_dir=checkpoint_dir,
    )
    print(str(artifacts.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
