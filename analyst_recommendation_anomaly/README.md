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
- Recommendation revisions are computed only from the current and previous retained snapshots for the same CRSP security. Exact duplicate rows are collapsed; conflicting rows for a PERMNO/snapshot date are all excluded with a warning. A revision can therefore span an excluded snapshot, and the as-of join can retain an earlier available signal.
- Historical CRSP name-date intervals are used for CUSIP links; ticker-only matching is deliberately avoided.
- CRSP ordinary and delisting returns are combined as `(1 + ret) * (1 + dlret) - 1`.
- Portfolio breakpoints are recomputed cross-sectionally each formation month.
- Missing next-month returns are not silently replaced by zero; coverage is reported.
- `performance_summary` reports total, observed, and missing months. With missing held outcomes it warns and calculates mean, volatility, Sharpe, and average turnover only over observed months; these are incomplete-sample statistics and may be biased. CAGR and maximum drawdown are `NaN` whenever the return series has gaps, and the wealth series remains unknown after the first gap. Use `performance_summary(results, strict=True)` to raise an error instead.

## Important research limitations

- IBES `recdsum.cusip` and the CRSP historical CUSIP link should be audited on the licensed data. Ambiguous matches are rejected by the linker.
- The summary file is a snapshot/consensus strategy, not an analyst-level brokerage-quality strategy. Studying broker type requires detail-file metadata and a defensible historical broker classification.
- Equal weighting can overstate implementability in small stocks. Apply price/market-cap screens, value weighting, transaction costs, and liquidity constraints for a serious replication.
- The included return series is raw long-short performance. Add factor regressions and robust inference before claiming abnormal returns.
- Survivorship bias is avoided only if the full historical CRSP/IBES files are queried rather than a current-security list.

For variable definitions and licensed-data details, consult the WRDS data dictionaries available through your account. The public [CRSP monthly stock demo](https://wrds-www.wharton.upenn.edu/demo/crsp/form/) documents variables including `PERMNO`, returns, prices, shares, and historical identifiers.

## Individual analyst recommendations download

Run this from the workspace directory (the parent of this project):

```bash
zsh analyst_recommendation_anomaly/download_ibes_analysts.command
```

Enter your WRDS credentials in Terminal. The launcher downloads announcements
from 2000-01-01 through 2025-12-31. Override dates by appending, for example,
`--start 1999-01-01 --end 2025-12-31`. Earlier history can supply the prior rating
needed to calculate revisions at the start of a backtest.

The downloader detects `recddet` or `recdet` in `ibes` and checks for `ticker`,
`amaskcd`, `emaskcd`, `ireccd`, and `anndats` before downloading. Your account
must have access to the detail recommendations product. To inspect fields only:

```bash
zsh analyst_recommendation_anomaly/download_ibes_analysts.command --inspect
```

Use `--library NAME --table NAME` if WRDS exposes the data under different names.
All vendor columns are retained, including activation/review dates and times
when present. The filter uses announcement dates; it does not assert those dates
are the first tradable availability dates. Missing analyst IDs and repeated
records are retained for subsequent research checks.

Each run creates `data/raw/ibes_analyst_<UTC timestamp>_<unique suffix>/`, containing
annual `recommendations_*.parquet` files and `manifest.json`. Each file is written
atomically; the manifest reports date range, source, row counts and completion.
A failed run remains marked incomplete. A new invocation starts a fresh extract;
it does not resume or mix masked IDs from separate vendor releases. Existing
summary and CRSP files are untouched. The strategy notebook still uses summary
data until a separate analyst-level signal pipeline is implemented.

Load one completed extraction from a notebook:

```python
import json
from pathlib import Path
import pandas as pd

extract_dir = RAW_DIR / 'ibes_analyst_REPLACE_WITH_YOUR_FOLDER'
manifest = json.loads((extract_dir / 'manifest.json').read_text())
if not manifest['complete']:
    raise ValueError('This extraction did not finish')
analyst_recommendations = pd.concat(
    [pd.read_parquet(extract_dir / item['file']) for item in manifest['files']],
    ignore_index=True,
)
```

## Analyst Sharpe strategy (US CRSP first)

The default analyst-level strategy enters on upward crossings of a revision-signal
z-score above **1.68** and holds each position for **14 trading sessions**. It uses
the prior **20 trading sessions** for z-score normalization. Each analyst’s latest
available revision contributes with **no age cutoff**, requiring at least two
analysts with valid historical skill scores. All five rating levels
are retained. This implementation is separate from the existing monthly notebook. The full assumptions and data limitations
are recorded in [analyst_sharpe_strategy.md](analyst_sharpe_strategy.md), with
configurable defaults in [configs/analyst_sharpe.json](configs/analyst_sharpe.json).

From this project directory, run the self-contained synthetic demonstration:

```bash
../.venv/bin/python -m src.run_analyst_sharpe --demo
../.venv/bin/python -m pytest -q
```

The demo uses artificial weekday data and changing five-level ratings and is
**not evidence of investment performance**. It exercises analyst warmup, daily
revision signals, z-score crossings, fixed holding periods, and gross/net
accounting for skill-weighted and unweighted strategies without a WRDS connection.

For licensed data, first download daily stock returns and the CRSP value-weighted
total-return benchmark. Existing monthly files are insufficient:

```bash
../.venv/bin/python -m src.download_analyst_market --start 2000-01-01 --end 2024-12-31
```

Alternatively, run `zsh analyst_recommendation_anomaly/download_analyst_daily.command`
from the workspace parent directory. Authentication uses the existing WRDS setup;
no credentials are stored by these scripts. Each download creates a new
`data/raw/analyst_daily_*` directory and a completion manifest. The downloader
preserves delisting-only rows and rejects conflicting joins or off-calendar
events for review. It targets legacy CRSP tables through 2024; later dates need
a separately validated daily CIZ adapter.

Use one completed IBES detail extraction and the newly downloaded daily directory:

```bash
../.venv/bin/python -m src.run_analyst_sharpe \
  --extract data/raw/ibes_analyst_REPLACE_WITH_YOUR_FOLDER \
  --market-dir data/raw/analyst_daily_REPLACE_WITH_YOUR_FOLDER \
  --names data/raw/crsp_names.parquet \
  --start 2005-01-03 --end 2024-12-31 \
  --config configs/analyst_sharpe.json
```

The dates above are an example, not a selected optimal evaluation period. Inputs
before `--start` supply warmup history. Start/end must fall within the daily
benchmark's date coverage. Signals are evaluated daily, including pre-start
warmup. Entry follows a qualifying crossing at the next session close; exit is
14 sessions after the entry close. Signals while held are discarded. The old
periodic strategy is available only with `strategy_mode="scheduled"`.

For independently prepared, historically linked inputs, use `--events FILE`
instead of `--extract`. The normalized Parquet schema is:

```text
permno: integer
analyst_id: string (stable within the supplied data vintage)
recommendation_date: timezone-naive date
available_date: timezone-naive date, no earlier than recommendation_date
rating: +2 / +1 / 0 / -1 / -2 (higher is more bullish)
event_type: rating / withdraw / invalid (optional; default rating)
```

The market directory must contain a completed `manifest.json` with a `files`
list of objects containing `file` paths, daily Parquet files with
`permno,date,prc,shrcd,exchcd,is_terminal` and either `ret_total` or `ret,dlret`,
and `benchmark.parquet` with `date,ret`. Dates are US session dates; prices are
used for eligibility/execution checks, while total returns drive accounting.

Runs write unique directories under `outputs/analyst_sharpe/`, with a configuration
and SHA-256 input/code manifest, normalized events, forward outcomes and status,
analyst scores, revision scores/exclusions, trailing z-scores, analyst contributions,
accepted/rejected orders, trades, completed/open positions, daily holdings/results,
and summary CSV. Scheduled mode instead records rebalance schedules and targets. A failed run remains marked
incomplete with its error. Untradable new entries are logged as `untradable_at_execution` and skipped.
Missing held returns and unavailable held-position exit prices
stop the run instead of inventing returns or trades. Raw licensed extracts and
generated artifacts are ignored by Git.

Real-data runs use bounded-memory execution: daily files are processed one at a
time, historical labels are built in temporary security buckets, and large
diagnostic panels are streamed to Parquet. Progress, peak memory and the last
completed stage are saved in the run directory. The reference in-memory engine
remains available for synthetic demonstrations. Neither implementation establishes
historical vendor identity correctness, calibrated execution costs, or profitability.

## Final-project presentation notebook

Open [notebooks/analyst_revision_presentation.ipynb](notebooks/analyst_revision_presentation.ipynb)
and choose **Restart Kernel and Run All**. This is the presentation for the
revision-z-score strategy, not the older monthly consensus notebook.

The notebook contains:

- Research question, exact signal/timing rules, and explicit assumptions.
- A run cell for the complete strategy, or loading a previously completed run.
- Net/gross returns, CAGR, volatility, zero-risk-free Sharpe, Sortino, Calmar,
  drawdown depth/duration, empirical tail returns, tracking error, information
  ratio, descriptive benchmark beta and correlation.
- Benchmark/unweighted comparisons, annual/monthly returns, rolling risk,
  cumulative wealth, drawdown charts, and a monthly heatmap.
- Entries, completed/open positions, net trade win rate, profit factor, holding
  durations, cash/investment, turnover, fees, and closed-position P&L by stock.
- A stock-level signal/z-score walkthrough, analyst skill/call-age diagnostics,
  data coverage, execution decision counts, and internal consistency checks.
- Optional one-factor threshold/cost sensitivity, with the baseline held at
  1.68 and no automatic parameter selection.
- CSV tables, PNG/PDF figures, a conclusions file, and run provenance.

Install the optional notebook dependencies from the **workspace directory**:

```bash
.venv/bin/python -m pip install -r analyst_recommendation_anomaly/requirements-notebook.txt
```

Select that environment as the notebook's Python kernel. The settings cell has
three explicit modes; synthetic results are labeled throughout every report:

| Mode | Settings | Behavior |
|---|---|---|
| Synthetic demonstration | `RUN_MODE = "demo"` | Generates artificial data, runs the strategy and all analysis; not an investment result |
| Licensed-data research | `RUN_MODE = "real"`, `EXTRACT_DIR`, `MARKET_DIR`, `START`, `END` | Runs the actual extracts; missing data raises an error, with no synthetic fallback |
| Existing completed run | `RUN_MODE = "existing"`, `EXISTING_RUN_DIR` | Loads saved revision-z-score artifacts without rerunning the backtest |

Use `RUN_SENSITIVITY=False` when original daily inputs are unavailable or to skip
scenario replays. The baseline presentation still runs from saved artifacts.
`DOWNLOAD_DAILY=True` is an explicit opt-in to downloading inside the notebook;
WRDS credentials and the required products must already be available. The
Terminal download launcher is usually easier for authentication.

For a one-command **executed notebook and HTML presentation**, from the workspace:

```bash
.venv/bin/python analyst_recommendation_anomaly/scripts/run_presentation.py --mode demo
```

If your existing `python3` environment already has the notebook dependencies,
`python3` can run the same command; the runner pins its local Jupyter kernel to
that interpreter. The core strategy defaults to the workspace `.venv` interpreter.

Run the real presentation after downloading daily inputs:

```bash
.venv/bin/python analyst_recommendation_anomaly/scripts/run_presentation.py \
  --mode real \
  --extract analyst_recommendation_anomaly/data/raw/ibes_analyst_REPLACE \
  --market-dir analyst_recommendation_anomaly/data/raw/analyst_daily_REPLACE \
  --start 2005-01-03 --end 2024-12-31
```

Analyze a saved run:

```bash
.venv/bin/python analyst_recommendation_anomaly/scripts/run_presentation.py \
  --mode existing \
  --run-dir analyst_recommendation_anomaly/outputs/analyst_sharpe/REPLACE_WITH_RUN
```

The runner writes a unique folder under `outputs/presentations/` containing
`analyst_revision_presentation.html`, an executed `.ipynb`, figures, tables and
execution metadata. HTML hides input code by default for presenting; pass
`--show-code` to include it. The notebook always retains the analysis code.
HTML embeds plot images; mathematical typesetting may require the viewer's
MathJax support/network access. A failed execution is saved separately and
remains explicitly marked incomplete.

Treat the current report as exploratory: no factor-adjusted alpha significance,
walk-forward holdout results, or calibrated capacity/impact estimate is claimed.
Those require additional validated data and research design. Partial calendar
periods and open positions at the sample end are labeled rather than discarded.

### Troubleshooting notebook strategy execution

The strategy cell now streams live progress and writes a persistent log under
`outputs/presentations/process_logs/`. Every run creates its manifest before
loading data, and updates `stage`, `peak_rss_mib`, and `progress.log` as it proceeds.
A failed subprocess reports its exit code or signal and the log location, even
if Python never produced a traceback. `SIGKILL` can indicate memory pressure or
an external kill; the code alone does not establish which.

After updating the notebook, reload it from disk, restore your desired settings,
and **Restart Kernel and Run All**. A still-open IDE copy may contain the previous
`subprocess.run(..., capture_output=True)` cell, which hides progress.

For real runs, the presentation loads exact saved coverage/age aggregates and
only the example stock's signal history. Cost/threshold scenarios replay the
saved signals and market data in one pass instead of loading all years into RAM.
Complete event, outcome, score and contribution panels remain on disk for
additional analysis. Missing held returns still stop the run explicitly; the
memory fix does not impute prices, change signal timing, or suppress data errors.

The revision strategy now defaults to `revision_freshness_sessions: null` (no call-age
filter). In the presentation notebook use `RUN_MODE = "real"` to recompute results;
`existing` mode retains the saved run’s rules. Remove any explicit freshness=14
override, or set `CONFIG_OVERRIDES = {"revision_freshness_sessions": None}`.

### Missing terminal payoffs in the presentation

The project JSON and presentation notebook now select the approved approximate
last-close exit policy for missing terminal payoffs (for example SWY / PERMNO
76149 on 2015-01-29). The core config class remains strict unless overridden:

```python
CONFIG_OVERRIDES = {
    "revision_freshness_sessions": None,
    "missing_terminal_policy": "last_close",
}
```

This models an exit at that day's recorded close using its observed ordinary
return, charges exit costs, and omits the unknown after-delisting payoff. It
requires both a valid same-day price and ordinary return. Affected trades are
flagged `terminal_close_proxy`; the presentation exports
`approximate_terminal_exits.csv` and labels its figures approximate. These are
assumption-dependent results, not repaired terminal-return data. Use `"error"`
to retain strict behavior. Saved runs must be recomputed after a policy change.

The current position limit is `top_n=100`, targeting 1% of current NAV per new
position (subject to cash and costs). This is a maximum of simultaneous holdings,
not a fixed number of trades. Existing saved runs retain their original limit.

### Optional long/short portfolio

The separate `configs/analyst_long_short.json` enables `portfolio_mode="long_short"`.
The standard `analyst_sharpe.json` defaults to `long_only`. In the notebook, set
`CONFIG_OVERRIDES["portfolio_mode"] = "long_short"`; merely opening the separate
JSON file does not switch the notebook. The report displays the saved portfolio
mode and uses gross/net exposure charts for long/short runs.
For a synthetic backend smoke run:

```bash
../.venv/bin/python -m src.run_analyst_sharpe --demo --config configs/analyst_long_short.json
```

Long entries require a strict upward z-score crossing above +1.68; shorts require
a strict downward crossing below −1.68. Both require a valid previous z-score,
execute next close, and hold 14 trading sessions. The shared limit is 100 names
across both sides, with target absolute size 1% of NAV per new position, ranked
by absolute z-score and then PERMNO. No fixed 50/50 allocation or dollar-neutral
constraint is imposed; existing positions cannot flip sides or reset their clock.

New entries are limited to 100% gross exposure after entry costs. Short proceeds
cannot finance extra gross leverage. Market moves and borrow costs may push gross
exposure above 100%; no forced margin liquidation or daily resizing is modeled.
The short leg is a signed total-return liability, not a share-level stock-loan
ledger. Cash includes short-sale proceeds and is not all available for reinvestment.
Use `gross_exposure`, `net_exposure`, and `free_collateral` to interpret the account.

`short_borrow_bps_annual` is configurable and charged on the prior-close short
notional for each held trading session using 252 sessions per year, including
exit day but excluding entry day. Its default is explicitly **zero**, an
optimistic research assumption, not an estimate of actual borrow costs. No
locates, recalls, margin rules, rebates, or availability constraints are modeled.
Both buy and sell executions incur normal trading costs. Costs and side-aware
trade P&L, including borrow fees, are exported by the backend. The gross shadow
account excludes both trading and borrow costs. Short losses are not capped at
100% of the initial position value; insolvent portfolios stop with an error.

The verified BIIB halt on November 6, 2020 and SOV halt on October 24, 2005
carry the prior valuation without execution and retain the full reopening return. Under the user-approved
`missing_terminal_policy="last_close"`, a held stock with exchange code 0 and
missing price and returns uses an explicitly non-executable prior-value exit
approximation, with fees and an `off_exchange_close_proxy` flag. This includes
KEG on April 8, 2005. The presentation exports affected positions and valuation
events. Other unresolved missing observations remain errors; historical signal
and analyst-label data are not filled.

### Direct positive-Sharpe revision signal, 200 positions

Use `configs/analyst_raw_signal_long_short.json` for the new `revision_signal` mode. Only positive-Sharpe analysts contribute; weights sum to one per stock, with at least two contributors. Raw weighted revisions above +1 enter long; below −1 enter short. No z-score or crossing is required. The shared cap is 200, with 0.5% target entry sizing and 14-session holds. Persistent qualifying signals can re-enter on the session after exit.

The presentation notebook is configured for a fresh real run of this variant, with sensitivity disabled. Restart its kernel and Run All. Older saved results remain unchanged; their z-score panels do not satisfy the new contributor-count rule. A synthetic check is:

```bash
../.venv/bin/python -m src.run_analyst_sharpe --demo --config configs/analyst_raw_signal_long_short.json
```

The dedicated notebook for this experiment is [`notebooks/analyst_raw_signal_long_short_200.ipynb`](notebooks/analyst_raw_signal_long_short_200.ipynb). It loads the raw-signal configuration directly and defaults to a real run with sensitivity disabled. Open it and choose Restart Kernel → Run All. The presentation runner accepts `--notebook` to execute this separate notebook.
