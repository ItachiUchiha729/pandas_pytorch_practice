# Analyst Anomaly Research Roadmap

This document records feasible extensions to the analyst recommendation project and the data required for each one. The recommended progression starts with strategies that use the existing IBES summary and CRSP pipeline, then adds more complex datasets only after the basic timing and identifier checks pass.

## Recommended core project

Compare three related signals from 2000–2025:

1. Analyst recommendation consensus
2. Recommendation revisions
3. Recommendation revisions combined with price momentum

For every specification, report equal- and value-weighted results, gross and net returns, turnover, portfolio counts, subperiod performance, and robustness to price and size screens.

### Existing signals

IBES recommendation scores generally run from favorable low values to unfavorable high values. The project therefore defines signals so that larger values are always more favorable:

```text
consensus_signal = -meanrec
revision_signal  = previous_meanrec - current_meanrec
```

A positive revision is an upgrade, while a negative revision is a downgrade.

## Feasible project ideas

### 1. Consensus recommendation strategy

Sort stocks by average analyst recommendation. Go long the most favorable decile and short the least favorable decile.

Data:

- IBES recommendation summary: `ticker`, `cusip`, `statpers`, `meanrec`, `numrec`
- CRSP monthly stock data: `permno`, `date`, `ret`, `dlret`, `prc`, `shrout`
- A historical IBES-to-CRSP link

Status: already implemented.

### 2. Recommendation-revision strategy

Sort on changes in consensus rather than its level. Compare one-month and multi-month changes and test whether large upgrades and downgrades have asymmetric effects.

Data: the same IBES summary and CRSP files used by the consensus strategy.

Status: already implemented for changes between consecutive IBES snapshots.

### 3. Recommendation plus momentum

Test whether recommendations are more informative when they agree with past price movement. Construct conventional 12–2 month momentum using only returns known at formation.

Example double sort:

```text
Long:  recommendation upgrades and high momentum
Short: recommendation downgrades and low momentum
```

Data: existing IBES recommendation data and CRSP monthly returns. No additional database is required.

Status: best next extension.

### 4. Analyst coverage

Use `numrec` as a measure of analyst attention. Test recommendation effects separately among low- and high-coverage firms.

Hypothesis: recommendation changes may be more informative for firms receiving less analyst attention.

Data: existing IBES `numrec`, CRSP returns, and market capitalization.

Status: immediately feasible.

### 5. Analyst disagreement

Measure dispersion across recommendations and test whether disagreement predicts returns or changes the strength of the consensus signal.

Possible portfolios:

- Favorable consensus with low disagreement
- Favorable consensus with high disagreement
- Unfavorable consensus with low disagreement
- Unfavorable consensus with high disagreement

Data: a dispersion field in the accessible IBES summary schema or analyst-level detail recommendations. Inspect the licensed schema before implementing because optional `recdsum` fields vary across WRDS installations.

Status: feasible after a schema inspection.

### 6. Recommendation plus fundamentals

Combine recommendations with characteristics such as:

- Earnings-to-price
- Book-to-market
- Profitability
- Accruals
- Firm size

Data: IBES and CRSP plus Compustat, CRSP/Compustat Merged, or the WRDS Financial Ratios with IBES product.

Status: feasible if the WRDS account has Compustat or financial-ratio access.

### 7. Earnings-forecast revisions

Replace recommendations with changes in analyst EPS forecasts. Scale forecast changes by price to improve comparability across firms.

```text
forecast_revision = (current_EPS_forecast - prior_EPS_forecast) / lagged_price
```

Data: IBES summary or detail earnings forecasts, forecast period identifiers, snapshot or announcement dates, CRSP prices, and subsequent returns.

Status: feasible with IBES Historical Estimates access.

### 8. Earnings surprise and post-earnings-announcement drift

Combine standardized unexpected earnings with contemporaneous recommendation revisions. Test whether analysts amplify, confirm, or underreact to earnings news.

Data:

- IBES forecasts and actual earnings
- Earnings announcement dates
- Recommendation revisions
- CRSP daily or monthly returns

Special care is required for forecast cutoffs, announcement timing, trading-day alignment, and split-adjusted versus unadjusted EPS.

Status: feasible but substantially more difficult; suitable for collaboration with the PEAD team.

### 9. Individual analyst skill

Rank analysts using prior forecast accuracy or recommendation performance, then test whether recommendations from historically skilled analysts predict returns more strongly.

Data: IBES Detail History with stable analyst identifiers and CRSP returns.

Status: technically feasible but high risk. IBES analyst identifiers have been reassigned historically, so data vintages and identifier changes require careful treatment.

### 10. Brokerage-house quality

Compare recommendations across broker types or historical broker performance.

Data: IBES Detail History, broker identifiers, and a defensible point-in-time broker classification.

Status: not recommended as the first extension. Broker identity changes and the lack of a clean historical investment-bank-versus-brokerage classification create major measurement risk.

### 11. Target-price implied return

Construct expected upside from the analyst target price relative to the contemporaneous stock price, then test whether extreme implied returns predict realized returns.

Data: IBES target-price detail and CRSP prices/returns.

Status: feasible only if the WRDS subscription includes target-price history.

## Data download feasibility

| Idea | Required WRDS products | Feasibility |
|---|---|---|
| Consensus | IBES Summary + CRSP | High |
| Revisions | IBES Summary + CRSP | High |
| Momentum interaction | IBES Summary + CRSP | High |
| Coverage | IBES Summary + CRSP | High |
| Disagreement | IBES Summary optional fields or Detail + CRSP | Medium |
| Fundamentals | IBES + CRSP + Compustat/Financial Ratios | Medium; subscription dependent |
| Forecast revisions | IBES Historical Estimates + CRSP | Medium; subscription dependent |
| PEAD interaction | IBES forecasts/actuals + CRSP | Medium-hard |
| Analyst skill | IBES Detail + CRSP | Hard |
| Broker quality | IBES Detail + external classification + CRSP | Hard |
| Target prices | IBES target-price history + CRSP | Medium; entitlement dependent |

The current downloader caches licensed data as Parquet under `data/raw/`. Existing cache files are reused automatically; `--force` is required to replace them.

## Required anti-leakage rules

Every extension must preserve these rules:

1. Use a signal only after its observable publication or snapshot date.
2. Apply a conservative delay when the intraday release time is unknown.
3. Form portfolios at month-end and use returns beginning in the next calendar month.
4. Compute rolling signals with trailing data only; never use centered windows.
5. Use historical identifier links and validity dates rather than current tickers.
6. Include delisting returns and never replace an unknown held-stock return with zero.
7. Use lagged market capitalization for value weights.
8. Build breakpoints separately at every formation date.
9. Include transaction costs and count entries, exits, and rebalancing in turnover.
10. Do not select specifications after viewing their full-sample performance without reporting the resulting multiple-testing problem.

## Suggested implementation order

1. Complete and validate the IBES recommendation and CRSP Parquet downloads.
2. Run the existing consensus and revision notebook on real data.
3. Inspect link rate, duplicates, date coverage, missing outcomes, and delisting coverage.
4. Add 12–2 momentum and two-dimensional portfolio sorts.
5. Add coverage-conditioned portfolios using `numrec`.
6. Add value weighting, Newey-West inference, and factor regressions.
7. Only then evaluate Compustat characteristics, earnings forecasts, analyst identity, or broker identity.

## Minimum final deliverables

- Reproducible WRDS extraction scripts
- Cached local Parquet inputs excluded from Git
- Point-in-time signal and return panel
- Equal- and value-weighted portfolio results
- Gross and net performance with turnover
- Cumulative-return plots and portfolio-count diagnostics
- Subperiod, size, price, and lag robustness tables
- Explicit leakage audit
- Clear separation between synthetic demonstrations and empirical WRDS results

