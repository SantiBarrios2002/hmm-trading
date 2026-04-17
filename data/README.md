# Local data directory

This directory holds large raw datasets used during development and experiments. **Nothing in here is tracked in git** — the contents exceed GitHub's repository size limits and are not meant to be versioned alongside the code. The `.gitignore` excludes everything except this README.

## Layout

```
data/
├── README.md                 # (this file, tracked)
├── databento.zip             # full databento download archive (not tracked)
└── databento/
    └── databento/            # extracted parquet files (not tracked)
        ├── ES_c_0_ohlcv-1m_2019-01-01_2024-12-31.parquet
        ├── NQ_c_0_ohlcv-1m_2019-01-01_2024-12-31.parquet
        ├── ...
```

## Contents

- **`databento/databento/*.parquet`** — 1-minute OHLCV bars for ~29 continuous-contract futures (ES, NQ, RTY, YM, CL, GC, 6E, ZN, etc.) covering 2019-01-01 through 2024-12-30 in UTC. The primary replication target is `ES_c_0_ohlcv-1m_2019-01-01_2024-12-31.parquet` (e-mini S&P500 continuous contract).
- **`databento.zip`** — the original archive the parquet files were extracted from.

## Schema (databento parquet)

- Index: `ts_event` (tz-aware UTC `DatetimeIndex`)
- Columns: `rtype`, `publisher_id`, `instrument_id`, `open`, `high`, `low`, `close`, `volume`, `symbol`

The project's canonical contract (`MarketDataSpec` in `src/hft_hmm/data.py`) is `timestamp` / `price` / `volume`. Issue 02b adds a loader that maps `ts_event` → `timestamp`, `close` → `price`, and preserves `volume`.

## Acquisition

The databento files were exported from [databento.com](https://databento.com) using a continuous-contract dataset definition. To reproduce, either:

- copy the archive from wherever it is stored locally, or
- re-export from databento with the same symbols, frequency (`ohlcv-1m`), and date range (`2019-01-01` to `2024-12-31`).

## Test fixtures

Unit tests must not depend on anything in `data/`. Small, committed fixtures live under `tests/fixtures/` and should be kept to tens of KB. Issue 02b adds a compact single-day parquet fixture for the databento loader tests.
