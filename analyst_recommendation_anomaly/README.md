# Analyst Recommendation and Earnings Forecast Anomaly

This project studies a monthly long-short anomaly based on IBES analyst recommendations and CRSP returns. It provides two signals:

1. **Consensus:** prefer firms with lower IBES `meanrec` (1 is favorable and 5 is unfavorable), implemented as `-meanrec` so a larger signal is always better.
2. **Revision:** prefer firms whose consensus recommendation improved since the prior IBES observation, implemented as `previous_meanrec - meanrec`.

The backtest goes long the top signal quantile and short the bottom signal quantile. It uses equal-weighted legs by default.

## Project layout

```text
analyst_recommendation_anomaly/
├── data/raw/                  # WRDS downloads (ignored by git)
├── data/processed/            # Research panels (ignored by git)
├── notebooks/
│   └── analyst_recommendation_anomaly.ipynb
├── outputs/                   # Backtest tables and plots
├── src/
│   ├── config.py
│   ├── download_data.py
│   ├── signals.py
│   └── backtest.py
└── tests/
    └── test_no_lookahead.py
```

## Setup

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WRDS access requires an institutional WRDS account with IBES and CRSP subscriptions. Set your username without storing a password in source code:

```bash
export WRDS_USERNAME="your_username"
```

The `wrds` package will use your normal WRDS/libpq authentication setup and may prompt interactively. Never commit credentials or raw licensed data.

## Download

```bash
python -m src.download_data --start 2000-01-01 --end 2025-12-31
```

Existing Parquet files are reused automatically. Pass `--force` only when you
intentionally want to replace the local cache with a fresh WRDS download.

The downloader writes:

- `data/raw/ibes_recommendations.parquet`
- `data/raw/crsp_monthly.parquet`
- `data/raw/crsp_names.parquet`

WRDS installations can expose different CRSP releases or schemas. The SQL defaults use the long-standing `ibes.recdsum`, `crsp.msf`, `crsp.msenames`, and `crsp.msedelist` names. If your subscription uses CRSP's newer CIZ tables, inspect available tables with `python -m src.download_data --list-library crsp` and adapt the query constants in `src/download_data.py`.

## Run and test

Open `notebooks/analyst_recommendation_anomaly.ipynb` and run it from top to bottom after downloading data.

Run the leakage/unit tests with:

```bash
pytest -q
```

## Timing conventions and leakage controls

- `statpers` is the IBES snapshot date. A configurable one-calendar-day publication lag creates `available_date`.
- A signal is usable at month-end only when `available_date <= formation_date`.
- Month `t` weights are joined to returns from calendar month `t+1`; current-month returns are never used as outcomes.
- The downloader requests one extra CRSP month beyond the final signal month solely to observe that final holding-period return.
- Recommendation revisions are computed only from the current and previous snapshots for the same CRSP security.
- Historical CRSP name-date intervals are used for CUSIP links; ticker-only matching is deliberately avoided.
- CRSP ordinary and delisting returns are combined as `(1 + ret) * (1 + dlret) - 1`.
- Portfolio breakpoints are recomputed cross-sectionally each formation month.
- Missing next-month returns are not silently replaced by zero; coverage is reported.

## Important research limitations

- IBES `recdsum.cusip` and the CRSP historical CUSIP link should be audited on the licensed data. Ambiguous matches are rejected by the linker.
- The summary file is a snapshot/consensus strategy, not an analyst-level brokerage-quality strategy. Studying broker type requires detail-file metadata and a defensible historical broker classification.
- Equal weighting can overstate implementability in small stocks. Apply price/market-cap screens, value weighting, transaction costs, and liquidity constraints for a serious replication.
- The included return series is raw long-short performance. Add factor regressions and robust inference before claiming abnormal returns.
- Survivorship bias is avoided only if the full historical CRSP/IBES files are queried rather than a current-security list.

For variable definitions and licensed-data details, consult the WRDS data dictionaries available through your account. The public [CRSP monthly stock demo](https://wrds-www.wharton.upenn.edu/demo/crsp/form/) documents variables including `PERMNO`, returns, prices, shares, and historical identifiers.
