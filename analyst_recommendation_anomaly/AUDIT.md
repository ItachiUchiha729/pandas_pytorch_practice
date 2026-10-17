# Signal and trading-cost audit

Audited the saved notebook rerun and local WRDS files on 2026-10-13. Reproducible command from this directory:

```sh
../.venv/bin/python scripts/audit_signals.py
```

The audit writes diagnostics under `outputs/audit/`; it does not change strategy definitions, downloaded files, notebook outputs, or the saved monthly result CSVs. It independently constructs a sorted as-of join, compares it to the production join on a deterministic security sample, then reproduces both strategies' saved gross returns, net returns, costs, and turnover to absolute tolerance 1e-12 (with NumPy's default relative tolerance).

## What is correct

- All 1,304,056 retained recommendation observations satisfy the consensus and revision formulas; no nonmissing recommendation score is outside 1–5.
- Consensus is negative mean recommendation; revision is prior minus current mean recommendation within PERMNO. Thus lower ratings and upgrades rank higher. IBES's 1=Strong Buy through 5=Sell scale is documented by [S&P Global](https://www.marketplace.spglobal.com/en/support/query-library/query-(237)).
- No eligible row uses a signal whose configured availability date is after formation. This verifies the implemented one-calendar-day lag, not the historical correctness of the vendor timestamp or identifier mapping.
- Portfolios have zero net exposure and gross exposure 2: +100% long and -100% short. Consensus longs have average mean recommendation 1.387 versus shorts 3.144. Revision longs average +0.249 and shorts -0.283. Four short stock-months have a nonnegative revision because the strategy chooses rank deciles, not explicitly negative-only shorts.
- Outcome matching uses the following calendar month. Missing held returns remain missing.

## Issues in signal interpretation and construction

1. **Unlimited carry-forward:** 2,577 of 746,321 eligible stock-months use snapshots older than 45 days; 1,819 are older than one year. The oldest is 7,656 days old. Held stale observations number 1,152/148,992 for consensus and 464/147,728 for revisions. A freshness rule is needed; 45 days here is an audit threshold, not a tuned strategy setting.
2. **Revision continuity:** 4,642 nonmissing revisions compare snapshots more than 45 days apart; 123 compare across a change in IBES ticker. These are flagged for review, not all proven erroneous. Current grouping by PERMNO does not enforce a continuous IBES series or consecutive monthly observations.
3. **Consensus change is not an individual upgrade:** 275,492 nonzero revisions coincide with a change in recommendation count. A change in the analyst pool can change the mean even when no continuing analyst revises a rating. Summary data cannot distinguish those explanations; a pure analyst-upgrade strategy requires detail data.
4. **Identifier validity remains unverified:** the link uses CUSIP and CRSP name validity dates, not an independently validated historical IBES–CRSP link. The checks establish internal consistency, not vendor-level historical identity correctness.
5. **Configuration trap:** `BacktestConfig.publication_lag_days` is not applied by `build_formation_panel`. The notebook separately passes 1 into `make_recommendation_signals`. Changing the config alone will not change timing.

## Trading-cost analysis

Cost = sum of absolute changes in target portfolio weights × cost_bps / 10,000. This is a cost per dollar bought or sold. Full replacement of both legs implies turnover 4 and cost 40 bps at the configured rate. There should be no division by two under this convention; a round-trip cost quote would require a different interpretation.

| Cost per dollar traded | Consensus Sharpe | Revision Sharpe | Consensus annualized mean | Revision annualized mean |
|---|---:|---:|---:|---:|
| 0 bps | 0.071 | 0.513 | 0.873% | 3.104% |
| 2 bps | 0.059 | 0.373 | 0.725% | 2.258% |
| 5 bps | 0.041 | 0.164 | 0.503% | 0.990% |
| 10 bps | 0.011 | -0.186 | 0.134% | -1.124% |
| 20 bps | -0.049 | -0.885 | -0.605% | -5.352% |

On observed months, mean target-weight turnover is 0.616 for consensus and 3.523 for revisions. At 10 bps, mean monthly cost is 6.16 and 35.23 bps respectively. Arithmetic annualized cost drag is 0.739 and 4.228 percentage points. Break-even trading rates are 11.81 and 7.34 bps respectively; these are sample estimates, not recommended trading-cost assumptions.

The cost model is internally consistent but simplified: it compares successive target weights, not return-drifted positions, and omits borrow fees, funding, market impact and terminal liquidation. No calibrated execution-cost estimate is available in this project.

## Coverage problems still limit the result

Consensus statistics use 172/300 months; revisions use 217/300. Missing-month selection may bias both estimates. They are not directly comparable samples and are not full-period Sharpe ratios.

The latest CRSP record is December 2024, although IBES extends through December 2025. December 2024 formation has no January 2025 outcome at all: 460 consensus and 458 revision positions are missing solely at that final formation date. This is consistent with [CRSP's documented retirement of legacy data after December 2024](https://www.crsp.org/wp-content/uploads/mdaz_202506_quarterly.pdf).

Before that last formation month, there are still 182 missing consensus stock-month outcomes and 93 revision outcomes. Most missing outcomes overall have no following-month stock row, not merely a null return field. The extraction is anchored on `crsp.msf`, so a delisting-only event without a matching monthly stock row cannot be retained by its left join. Establishing which missing outcomes can be recovered requires querying the event data or validated CIZ data; the downloaded files alone cannot establish this.

## Recommended sequence

First repair the data horizon and investigate absent outcomes. Then enforce and document freshness and revision-series continuity, and decide explicitly whether the research hypothesis concerns consensus changes or individual analyst actions. Finally model drift-adjusted trading and borrow costs and compare strategies on a common evaluation sample. Do not select cost assumptions or signal settings simply because they improve the historical Sharpe.
