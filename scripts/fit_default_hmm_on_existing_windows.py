"""Derive Default HMM headline metrics from an existing comparison run's geometry.

The full side-information comparison runner produces all six variants together
and writes artifacts only at the end. After two failed Gate K reruns (10+
hours of compute each, killed by environmental issues), this script provides
a fast path to populate the Default HMM row in ``docs/results_vs_paper.md``
without re-running the slow HMC variants whose numbers already exist in
prior runs (``runs/04269749abff`` for baseline + bucketed,
``runs/d8b6e7eef6c2`` for HMC continuous).

Because both PLR and the Default HMM forward filter are fully deterministic
under a fixed ``WalkForwardConfig``, the Default HMM numbers this script
produces are identical to what the full runner's ``_run_default_hmm_variant``
step would produce on the same windows. The walk-forward windows themselves
are re-derived from the source run's ``config.yaml`` via the same
``walk_forward`` call the runner uses, so window boundaries and per-window
``chosen_k`` match exactly.

Usage:
    python scripts/fit_default_hmm_on_existing_windows.py 04269749abff

Prints headline metrics suitable for the ``docs/results_vs_paper.md``
headline table row, plus a markdown-formatted row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hft_hmm.evaluation import cumulative_return, daily_annualized_sharpe_ratio, hit_rate
from hft_hmm.experiments._data_loading import load_returns_from_source
from hft_hmm.experiments.side_info_comparison import (
    SideInfoComparisonConfig,
    _run_default_hmm_variant,
)
from hft_hmm.experiments.walk_forward import walk_forward


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source_run_id",
        nargs="?",
        default="04269749abff",
        help="Existing run id whose config + window geometry to reuse",
    )
    parser.add_argument("--runs-root", default="runs", help="Root directory of runs/")
    args = parser.parse_args(argv)

    config_path = Path(args.runs_root) / args.source_run_id / "config.yaml"
    if not config_path.is_file():
        print(f"error: config not found at {config_path}", file=sys.stderr)
        return 1

    config = SideInfoComparisonConfig.from_yaml(config_path)
    returns = load_returns_from_source(config.data, frequency=config.frequency)

    baseline_wf = walk_forward(
        returns,
        config.walk_forward,
        cost_bps_per_turnover=config.cost_bps_per_turnover,
        signal_policy=config.signal_policy,
        signal_threshold=config.signal_threshold,
    )

    result = _run_default_hmm_variant(
        returns=returns,
        config=config,
        baseline_windows=baseline_wf.windows,
    )

    pre = result.pre_cost_returns
    post = result.post_cost_returns
    daily_sharpe_pre = daily_annualized_sharpe_ratio(pre)
    daily_sharpe_post = daily_annualized_sharpe_ratio(post)
    cum_pre = cumulative_return(pre)
    cum_post = cumulative_return(post)
    hit_pre = hit_rate(pre)

    sample_start = pre.index.min().isoformat()
    sample_end = pre.index.max().isoformat()
    n_windows = len(result.windows)
    n_forecast = int(pre.shape[0])

    print()
    print(f"source run id:           {args.source_run_id}")
    print(f"sample window:           {sample_start} -> {sample_end}")
    print(f"walk-forward windows:    {n_windows}")
    print(f"forecast bars:           {n_forecast}")
    print(f"chosen K per window:     {set(result.chosen_k_per_window)}")
    print(f"cost_bps_per_turnover:   {result.cost_bps_per_turnover}")
    print()
    print("--- Default HMM (sign policy) ---")
    print(f"pre-cost daily Sharpe:   {daily_sharpe_pre:.4f}")
    print(f"hit rate:                {hit_pre:.4f}")
    print(f"pre-cost cum return:     {cum_pre:.4f}")
    print(f"post-cost daily Sharpe:  {daily_sharpe_post:.4f}")
    print(f"post-cost cum return:    {cum_post:.4f}")
    print()
    print("--- Headline table row (markdown) ---")
    print(
        f"| Default HMM | sign | {daily_sharpe_pre:.4f} | {hit_pre:.4f} | "
        f"{cum_pre:.4f} | §3 Default HMM; PLR emissions, uniform A/π (Gate N) | "
        f"[derived from `{args.source_run_id}` windows] |"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
