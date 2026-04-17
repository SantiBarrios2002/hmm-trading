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
- project metadata and dependency definitions
- implementation planning documents
- issue breakdown for staged delivery
- a minimal `src/hft_hmm/` package scaffold
- smoke tests that validate the package baseline
- repository automation for linting and tests

Planned implementation areas:
- Gaussian HMM baseline for returns
- EM / Baum-Welch training
- filtering and one-step-ahead prediction
- model selection over hidden-state counts
- signal generation and evaluation
- side-information features
- explicitly labeled approximations where the paper is not replicated exactly

## Repository Contents

- [`README.md`](README.md): project overview and setup
- [`pyproject.toml`](pyproject.toml): packaging and tool configuration
- [`requirements.txt`](requirements.txt): runtime dependencies
- [`requirements-dev.txt`](requirements-dev.txt): development dependencies (includes runtime)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): development roadmap
- [`GITHUB_ISSUES.md`](GITHUB_ISSUES.md): suggested issue and milestone breakdown
- `src/hft_hmm/`: initial package scaffold
- `tests/`: repository smoke tests
- `docs/`, `notebooks/`, `scripts/`: reserved project directories

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

Alternatively, you can install development extras from [`pyproject.toml`](pyproject.toml):

```bash
pip install ".[dev]"
```

CI uses `pip install -e ".[dev]"` in the GitHub workflow so there is a single dependency source of truth in [`pyproject.toml`](pyproject.toml). The requirements files remain useful for explicit local runtime or dev installs.

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

## Initial Package Scaffold

The current package is intentionally small. It provides:
- importable package structure under `src/hft_hmm/`
- version metadata
- a small project metadata helper used by tests and tooling

This keeps the repository honest: CI already validates imports, formatting, linting, and test execution before the modeling code lands.

## Data Note

The local archive `databento.zip` is intentionally excluded from version control because it exceeds GitHub's standard repository file size limits and should not be committed directly.

## License

This repository is licensed under the MIT License. See [`LICENSE`](LICENSE).
