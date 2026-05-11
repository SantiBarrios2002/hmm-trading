"""Render the cumulative-return comparison figure for docs/results_vs_paper.md.

Reads ``runs/<run_id>/summary.json`` for the sign-policy and thresholded_hold
passes and writes ``docs/figures/cumulative_return_vs_paper.png``: a grouped
bar chart of pre-cost cumulative returns and separately labelled post-cost
diagnostics per variant, faceted by signal policy.

Usage:
    python scripts/plot_results_vs_paper.py <sign_run_id> <thresholded_run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANT_LABELS = {
    "baseline": "Baseline HMM",
    "volatility_ratio_conditioned": "Volatility IOHMM",
    "seasonality_conditioned": "Seasonality IOHMM",
    "long_only": "Long-only",
}
VARIANT_ORDER = (
    "baseline",
    "volatility_ratio_conditioned",
    "seasonality_conditioned",
    "long_only",
)


def _load_returns(summary_path: Path) -> dict[str, dict[str, float]]:
    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    variants = summary["variants"]
    out: dict[str, dict[str, float]] = {}
    for name in VARIANT_ORDER:
        if name not in variants:
            continue
        s = variants[name]["summary"]
        out[name] = {
            "pre-cost": float(s["pre-cost"]["cumulative_return"]),
            "post-cost": float(s["post-cost"]["cumulative_return"]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sign_run_id")
    parser.add_argument("thresholded_run_id")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/cumulative_return_vs_paper.png"),
    )
    args = parser.parse_args(argv)

    sign = _load_returns(args.runs_root / args.sign_run_id / "summary.json")
    thr = _load_returns(args.runs_root / args.thresholded_run_id / "summary.json")

    variants = [v for v in VARIANT_ORDER if v in sign and v in thr]
    labels = [VARIANT_LABELS[v] for v in variants]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sign_pre = [sign[v]["pre-cost"] for v in variants]
    sign_post = [sign[v]["post-cost"] for v in variants]
    thr_pre = [thr[v]["pre-cost"] for v in variants]
    thr_post = [thr[v]["post-cost"] for v in variants]

    ax.bar(x - 1.5 * width, sign_pre, width, label="sign, pre-cost", color="#246a73")
    ax.bar(x - 0.5 * width, sign_post, width, label="sign, post-cost", color="#c44569")
    ax.bar(x + 0.5 * width, thr_pre, width, label="thresholded_hold, pre-cost", color="#7fb069")
    ax.bar(x + 1.5 * width, thr_post, width, label="thresholded_hold, post-cost", color="#f0a04b")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("cumulative return")
    ax.set_title("Pre-cost comparison plus separate post-cost diagnostic")
    ax.legend(loc="best", frameon=False)
    fig.text(
        0.5,
        0.01,
        f"Runs {args.sign_run_id} (sign) and {args.thresholded_run_id} (thresholded_hold). "
        "Databento ES 1min, 2019-2024. Post-cost bars use the repo's 1 bp turnover "
        "stress test, not a paper-calibrated execution model.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
