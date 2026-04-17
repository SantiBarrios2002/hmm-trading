# Hidden Markov Models Applied to Intraday Momentum Trading

Academic implementation project for ASPTA based on:
- Christensen, Turner, Godsill, *Hidden Markov Models Applied To Intraday Momentum Trading With Side Information*
- ASPTA course project brief

## Project goal
This repository implements a clean, reviewable, academically grounded replication of the paper's main pipeline:
1. define a Gaussian HMM for returns,
2. learn model parameters,
3. perform filtering / prediction,
4. evaluate the resulting signal,
5. extend the baseline with side information.

The goal is **understanding and defensible simulation**, not production trading and not exact numerical reproduction of the paper's Sharpe ratio.

## Scope
### Included
- Gaussian HMM baseline
- EM / Baum-Welch training
- model selection over number of hidden states
- forward inference / filtering distribution
- expected-return prediction from state probabilities
- side-information features (volatility ratio, seasonality)
- IOHMM-style transition conditioning as an explicitly labeled approximation
- tests for every public function

### Explicitly not required in first milestone
- full MCMC replication
- bridge sampling for marginal likelihood
- exact 1-minute ES futures replication
- production-grade backtesting engine

## Repository rules
1. Every public function must be tested.
2. Every public function must cite the paper section / page / equation or be marked as an engineering approximation.
3. Final logic lives in `src/`, not only in notebooks.
4. Each issue is implemented on its own branch and merged through PR review.
5. Every PR states what is paper-faithful and what is approximate.

## Recommended environment
- Python 3.11
- dependencies in `requirements.txt`
- version pin hint in `.python-version`

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## About `.python-version`
This file contains just the Python version string, for example:
```text
3.11
```
Tools such as `pyenv`, `mise`, and some editors read it automatically and select that Python version when you enter the repo. It does **not** install Python by itself. It is a hint for local tooling.

If you use `pyenv`, the typical flow is:
```bash
pyenv install 3.11.9
pyenv local 3.11.9
```
That creates or updates `.python-version` so the repo uses that interpreter.

## Suggested structure
```text
src/hft_hmm/
  data/
  diagnostics/
  features/
  models/
  backtest/
  plots/
  utils/
tests/
notebooks/
scripts/
docs/
```

## Development workflow
1. pick one issue from `GITHUB_ISSUES.md`
2. create a branch
3. implement only that milestone
4. add tests
5. open a PR with references and validation evidence

Example:
```bash
git checkout -b feat/07-gaussian-hmm-baseline
```

## Quality commands
```bash
pytest -q
ruff check .
black --check .
mypy src
```

## Minimal milestone order
1. repo scaffold
2. data loading and return preprocessing
3. baseline Gaussian HMM
4. model selection
5. forward inference
6. simple signal and evaluation
7. side-information features
8. IOHMM-style approximation

## Notes for the report / presentation
Always state:
- dataset and frequency used
- what differs from the paper
- which routines come from libraries
- which parts were implemented directly
- what was validated through tests
