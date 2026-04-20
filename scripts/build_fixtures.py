"""Carve ES 1-minute CSV fixtures from the local Databento parquet.

Creates:
  - tests/fixtures/es_1min_sample.csv  (~5 trading days)
  - tests/fixtures/es_1min_month.csv   (full January 2024)

Both filtered to the paper's 01:00-15:15 America/Chicago trading window
(§7 — paper pins the Chicago-business-hours subset for the ES simulation).
Canonical CSV columns: ``timestamp`` (ISO-8601 UTC), ``price``, ``volume``.

The source parquet is gitignored; this script is committed as provenance so any
contributor with databento access can regenerate the tracked fixtures. Running
it twice yields byte-identical output — the committed CSVs are the oracle for
``tests/test_fixtures_build.py``.

If the source parquet is absent, the script exits cleanly with a message,
which is the expected behavior on a fresh clone without databento data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PARQUET = (
    REPO_ROOT / "data" / "databento" / "databento" / "ES_c_0_ohlcv-1m_2024-01-01_2024-02-01.parquet"
)
DEFAULT_OUTPUT_SAMPLE = REPO_ROOT / "tests" / "fixtures" / "es_1min_sample.csv"
DEFAULT_OUTPUT_MONTH = REPO_ROOT / "tests" / "fixtures" / "es_1min_month.csv"

SYMBOL = "ES.c.0"
TRADING_TZ = "America/Chicago"
SESSION_START = dt.time(1, 0)
SESSION_END = dt.time(15, 15)
SAMPLE_TRADING_DAYS = 5


def build_fixture(parquet_path: Path, *, sample_trading_days: int | None) -> pd.DataFrame:
    """Load the ES parquet and return the filtered bars.

    When ``sample_trading_days`` is given, the result is trimmed to the first
    N distinct Chicago trading dates after the session-window filter.
    """
    from hft_hmm.data import load_databento_parquet

    frame = load_databento_parquet(parquet_path, symbol=SYMBOL)
    chicago_ts = frame["timestamp"].dt.tz_convert(TRADING_TZ)
    chicago_time = chicago_ts.dt.time
    in_session = (chicago_time >= SESSION_START) & (chicago_time <= SESSION_END)
    filtered = frame.loc[in_session].reset_index(drop=True)

    if sample_trading_days is not None:
        chicago_dates = chicago_ts.loc[in_session].reset_index(drop=True).dt.date
        unique_dates = np.array(sorted(set(chicago_dates)), dtype=object)
        if unique_dates.size < sample_trading_days:
            raise ValueError(
                f"Source parquet has only {unique_dates.size} Chicago trading days; "
                f"requested {sample_trading_days}."
            )
        cutoff = unique_dates[sample_trading_days - 1]
        keep = chicago_dates <= cutoff
        filtered = filtered.loc[keep].reset_index(drop=True)

    return filtered


def write_fixture(frame: pd.DataFrame, output: Path) -> None:
    """Write ``frame`` to ``output`` as a canonical fixture CSV.

    Timestamps are rendered as ``YYYY-MM-DDTHH:MM:SS+00:00`` so the file is
    round-trippable through :func:`pandas.to_datetime(..., utc=True)`.
    """
    formatted = frame.loc[:, ["timestamp", "price", "volume"]].copy()
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise", utc=False)
    if timestamp.dt.tz is None:
        raise ValueError("timestamp column must be tz-aware before exporting fixtures.")
    formatted["timestamp"] = timestamp.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    output.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(output, index=False, lineterminator="\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build canonical CSV fixtures from the ES January 2024 Databento parquet.",
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=DEFAULT_SOURCE_PARQUET,
        help="Source Databento parquet to carve fixtures from.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=DEFAULT_OUTPUT_SAMPLE,
        help="Output CSV for the 5-trading-day sample fixture.",
    )
    parser.add_argument(
        "--month-output",
        type=Path,
        default=DEFAULT_OUTPUT_MONTH,
        help="Output CSV for the full-month fixture.",
    )
    args = parser.parse_args(argv)

    if not args.source_parquet.exists():
        print(
            f"Source parquet not found at {args.source_parquet}. "
            "Place the databento ES January 2024 slice there to rebuild fixtures.",
            file=sys.stderr,
        )
        return 0

    sample = build_fixture(args.source_parquet, sample_trading_days=SAMPLE_TRADING_DAYS)
    month = build_fixture(args.source_parquet, sample_trading_days=None)

    write_fixture(sample, args.sample_output)
    write_fixture(month, args.month_output)

    print(f"Wrote {_display_path(args.sample_output)} ({len(sample)} rows)")
    print(f"Wrote {_display_path(args.month_output)} ({len(month)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
