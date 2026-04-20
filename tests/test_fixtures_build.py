"""Tests for `scripts/build_fixtures.py` — idempotency against the committed CSVs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_fixtures.py"
SOURCE_PARQUET = (
    REPO_ROOT / "data" / "databento" / "databento" / "ES_c_0_ohlcv-1m_2024-01-01_2024-02-01.parquet"
)
TRACKED_SAMPLE = REPO_ROOT / "tests" / "fixtures" / "es_1min_sample.csv"
TRACKED_MONTH = REPO_ROOT / "tests" / "fixtures" / "es_1min_month.csv"

needs_source_parquet = pytest.mark.skipif(
    not SOURCE_PARQUET.exists(),
    reason="Source databento parquet is not present on this machine.",
)


def test_build_script_exits_cleanly_when_parquet_missing(tmp_path: Path) -> None:
    """Clone-fresh case: run in a sandboxed cwd where the parquet does not exist."""
    sandbox = tmp_path / "repo"
    (sandbox / "scripts").mkdir(parents=True)
    shutil.copy(BUILD_SCRIPT, sandbox / "scripts" / "build_fixtures.py")

    completed = subprocess.run(
        [sys.executable, str(sandbox / "scripts" / "build_fixtures.py")],
        capture_output=True,
        text=True,
        cwd=sandbox,
    )
    assert completed.returncode == 0
    assert "Source parquet not found" in completed.stderr


def test_tracked_fixture_has_expected_columns() -> None:
    frame = pd.read_csv(TRACKED_SAMPLE)
    assert list(frame.columns) == ["timestamp", "price", "volume"]
    # All rows parseable as UTC datetime.
    parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    assert not parsed.isna().any()
    # Chicago-session filter: 01:00-15:15 local time.
    chicago = parsed.dt.tz_convert("America/Chicago").dt.time
    assert chicago.min() >= pd.Timestamp("01:00").time()
    assert chicago.max() <= pd.Timestamp("15:15").time()


@needs_source_parquet
def test_build_fixtures_is_idempotent(tmp_path: Path) -> None:
    """Running the script twice against the same parquet produces byte-identical CSVs."""
    from hft_hmm.data import load_databento_parquet  # noqa: F401 -- ensures install is present

    baseline_sample = TRACKED_SAMPLE.read_bytes()
    baseline_month = TRACKED_MONTH.read_bytes()
    sample_output = tmp_path / "tests" / "fixtures" / "es_1min_sample.csv"
    month_output = tmp_path / "tests" / "fixtures" / "es_1min_month.csv"

    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--sample-output",
            str(sample_output),
            "--month-output",
            str(month_output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    sample_after_first = sample_output.read_bytes()
    month_after_first = month_output.read_bytes()

    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--sample-output",
            str(sample_output),
            "--month-output",
            str(month_output),
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert sample_output.read_bytes() == sample_after_first
    assert month_output.read_bytes() == month_after_first
    assert TRACKED_SAMPLE.read_bytes() == baseline_sample
    assert TRACKED_MONTH.read_bytes() == baseline_month
    assert "Wrote" in completed.stdout
    assert "es_1min_sample.csv" in completed.stdout


@needs_source_parquet
def test_build_fixtures_matches_tracked_bytes(tmp_path: Path) -> None:
    """The committed CSVs are the bit-for-bit output of the current script."""
    baseline_sample = TRACKED_SAMPLE.read_bytes()
    baseline_month = TRACKED_MONTH.read_bytes()
    sample_output = tmp_path / "tests" / "fixtures" / "es_1min_sample.csv"
    month_output = tmp_path / "tests" / "fixtures" / "es_1min_month.csv"

    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--sample-output",
            str(sample_output),
            "--month-output",
            str(month_output),
        ],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )
    assert sample_output.read_bytes() == baseline_sample
    assert month_output.read_bytes() == baseline_month
