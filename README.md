# HMM Trading

Public academic project for ASPTA focused on Hidden Markov Models for intraday momentum trading with side information.

[![CI](https://github.com/SantiBarrios2002/hmm-trading/actions/workflows/ci.yml/badge.svg)](https://github.com/SantiBarrios2002/hmm-trading/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository is a course project implementation inspired by:
- Christensen, Turner, Godsill, *Hidden Markov Models Applied To Intraday Momentum Trading With Side Information*
- the ASPTA project brief

The intent is a clean, reviewable, academically defensible replication pipeline, not a production trading system and not a claim of exact numerical reproduction of the paper.

## Current Status

The repository currently contains:
- canonical market-data loaders and validation for CSV, Databento parquet, and yfinance inputs
- preprocessing utilities for log returns, resampling, and chronological train/test splits
- a piecewise linear regression baseline and a Gaussian HMM modeling wrapper
- model-selection helpers (AIC/BIC) and log-space forward filtering
- sign-based signal generation plus backtest metrics with turnover-cost accounting
- a walk-forward experiment runner with deterministic YAML configs, run hashing, and saved artifacts
- tracked ES 1-minute CSV fixtures for integration tests and reproducibility checks
- repository automation for tests, linting, formatting, and typing

The core single-asset HMM replication pipeline is implemented through the evaluation layer. The main remaining scope is:
- side-information features such as volatility ratio and intraday seasonality
- spline-based predictor fitting and IOHMM-style transition conditioning approximations
- figure-generation and presentation-oriented project artifacts

## Repository Contents

- [`README.md`](README.md): project overview and setup
- [`pyproject.toml`](pyproject.toml): packaging and tool configuration
- [`requirements.txt`](requirements.txt): runtime dependencies
- [`requirements-dev.txt`](requirements-dev.txt): wrapper for editable install with `.[dev]`
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): development roadmap
- [`GITHUB_ISSUES.md`](GITHUB_ISSUES.md): suggested issue and milestone breakdown
- `src/hft_hmm/`: data, preprocessing, modeling, inference, strategy, evaluation, and experiment modules
- `tests/`: unit and integration tests, plus tracked fixtures
- `configs/`: reproducible experiment YAMLs
- `scripts/`: fixture-generation and experiment reproduction entry points
- `docs/`: source paper and supporting documentation

## Setup

### Python version

The project targets Python 3.11.

If you use `pyenv`, a typical setup is:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

### Local environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For development tools and test/lint dependencies, install:

```bash
pip install -r requirements-dev.txt
```

This wrapper delegates to [`pyproject.toml`](pyproject.toml), where development extras are the source of truth. You can run the equivalent command directly:

```bash
pip install -e ".[dev]"
```

CI uses `pip install -e ".[dev]"` in the GitHub workflow so there is a single dependency source of truth in [`pyproject.toml`](pyproject.toml). The requirements files remain useful for explicit local runtime or dev installs.

## Baseline Run

Run the example walk-forward experiment with:

```bash
python scripts/repro.py configs/example_es_csv.yaml
```

This writes a deterministic artifact directory under `runs/<run_id>/` containing:
- `config.yaml`
- `metrics.json`
- `log.jsonl`
- `figures/`

## Development Standards

- Public functions should be tested.
- Paper-faithful logic and engineering approximations should be clearly separated.
- Core implementation should live in `src/` rather than only in notebooks.
- Each change should be reviewable in isolation.

## Quality Checks

When implementation files are present, the intended checks are:

```bash
pytest -q
ruff check .
black --check .
mypy src
```

The GitHub Actions workflow now enforces the baseline checks on every push and pull request.

## Implemented Scope

The implemented codebase currently covers:
- market-data ingestion and schema validation
- return preprocessing and time-aware splitting
- Gaussian HMM training and selection over hidden-state counts
- log-space forward filtering and expected-return inference
- sign-based trading signals and compact backtest summaries
- walk-forward retraining experiments with reproducibility metadata and artifact logging

The code is organized so modules declare whether they are paper-faithful, engineering approximations, or evaluation-layer utilities. This keeps the replication scope explicit where the repository deliberately departs from the paper.

## Data Note

Raw datasets live under `data/` and are excluded from version control because they exceed GitHub's repository size limits. See [`data/README.md`](data/README.md) for the expected layout, schema, and acquisition instructions. Small fixtures used by the test suite live under `tests/fixtures/` instead.

The default development dataset is daily market data loaded through Yahoo Finance helpers, which keeps local iteration lightweight. Paper-replication runs target local Databento 1-minute ES parquet files under `data/databento/`, loaded through the repository's canonical `timestamp` / `price` / `volume` contract.

## License

This repository is licensed under the MIT License. See [`LICENSE`](LICENSE).
