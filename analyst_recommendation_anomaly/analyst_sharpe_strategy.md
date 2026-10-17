## Direct weighted-revision variant (current experiment)

The separate configuration `configs/analyst_raw_signal_long_short.json` selects `strategy_mode="revision_signal"`. This variant supersedes the z-score entry rule for the new experiment; older saved runs keep their original definitions.

For each stock, keep only analysts with a valid latest revision and strictly positive historical Sharpe. Set each weight to that analyst's Sharpe divided by the sum of contributing positive Sharpes for that stock. Weights sum to one; require at least two positive-Sharpe contributors. The raw signal is the weighted mean of their five-category rating revisions. The unweighted comparison uses the same positive-Sharpe pool with equal weights.

Enter long when the raw signal is strictly above +1 and short when it is strictly below −1. Equality does not qualify. No z-score, variance denominator, prior threshold crossing, or 20-session warmup is used for entry. Evaluate eligibility every session and execute at the next close. Rank competing entries by absolute raw signal then PERMNO. The common 200-position cap targets 0.5% of NAV per new position, subject to the existing gross-exposure budget and fees. A cap is not a requirement to hold 200 stocks.

Hold for 14 subsequent trading sessions without resizing or resetting the clock. A signal that remains beyond the threshold may enter again on the session after exit; no same-close exit/re-entry is allowed. Latest revisions retain no age cutoff. Historical skill definition, availability conventions, US CRSP screens, transaction costs and approved missing-data policies are unchanged. Short borrow fees remain configured at zero. Saved z-score panels cannot be reused without rebuilding positive-only contributor counts and unweighted signals.

# Analyst-weighted revision z-score strategy

## 1. Current strategy

Trade unusual increases in **analyst-weighted recommendation revisions**. Evaluate
the signal daily, enter when its z-score crosses above **1.68**, and hold each
position for **14 trading sessions after execution**. There is no scheduled
portfolio-wide rebalance in this mode.

The z-score compares today's stock signal with its previous **20 trading
sessions**. Each analyst’s latest available revision contributes with **no call-age
limit**, provided the analyst has a valid historical skill score. Preserve all five recommendation levels so that
Buy → Strong Buy is a positive revision.

This is the default `revision_zscore` mode in
[configs/analyst_sharpe.json](configs/analyst_sharpe.json). The implementation is in
[src/revision_zscore.py](src/revision_zscore.py), using historical analyst-skill
and data preparation functions from [src/analyst_sharpe.py](src/analyst_sharpe.py).
The earlier recommendation-level, two-week rebalance implementation remains
available only through `strategy_mode="scheduled"`; it does not implement the
current entry/exit rules.

## 2. Universe and inputs

Start with the existing **US CRSP universe**, not TOPIX. Require share codes
10/11, exchange codes 1/2/3, and an absolute formation-close price of at least
USD 5. Use historical PERMNO links and classifications, not a current-survivor
list. TOPIX/Japan is a later extension.

Inputs are one completed IBES detail extraction, historical CRSP names for
linking, daily stock total returns and execution prices, and the daily CRSP
value-weighted market total-return benchmark. The benchmark dates define the
trading calendar; weekends and exchange holidays are not sessions. Monthly
CRSP files cannot implement this strategy.

The existing CUSIP link assumes the vendor identifier is historically valid;
ambiguous links are excluded. One broker-mask/analyst-mask pair identifies an
analyst within a single extraction vintage. Broker changes start separate
histories. No identities are stitched across extracts.

## 3. Ratings, event availability, and revisions

Map the vendor's five codes to a bullish-increasing scale:

| IBES code | Encoded rating |
|---|---:|
| 1 | +2 |
| 2 | +1 |
| 3 | 0 |
| 4 | −1 |
| 5 | −2 |

For stock $i$, analyst $a$, and a new call $j$, define:

$$
\Delta R_{i,a,j}=R_{i,a,j}-R_{i,a,j-1}.
$$

Thus Buy (+1) → Strong Buy (+2) contributes +1. Strong Buy (+2) → Hold (0)
contributes −2. An unchanged reaffirmation contributes zero. An initiation with
no known prior call has no defined revision and is excluded from revision
coverage. Both calls may be older than 14 sessions; neither has an age cutoff.

At each cutoff, use only events already available. The raw adapter sets
availability to the latest nonmissing announcement, activation, or review date
plus one calendar day. Intraday times are unused. This delays inclusion of some records. Removing the age filter does not change
this availability convention; it remains a separate data-timing assumption.

Collapse exact duplicates. For each analyst–stock–announcement date, retain the
latest available version; conflicts at the same availability date invalidate
state. Then compare the latest dated call with its immediately preceding known
dated call. A future or unavailable predecessor cannot be used. A newly
available older record can affect today's reconstruction, but never rewrites a
past signal. Withdrawals and invalid records break comparison continuity.

Only the latest call per analyst–stock pair contributes. A new reaffirmation
replaces an earlier revision with zero; revisions are not accumulated across
multiple calls from the same analyst.

## 4. No call-age cutoff

The default `revision_freshness_sessions=null` disables the age filter. Use each
analyst’s latest available call minus its immediately preceding known dated call,
regardless of age. This revision persists until a newer available call replaces
it, an invalid/withdrawal event breaks continuity, or the analyst loses a valid
historical skill score. Reaffirmations replace the revision with zero. This can
retain revisions from analysts who have stopped issuing calls; no inactivity
expiry is imposed.

Availability is still enforced at every cutoff. The two-analyst minimum and
14-session holding period are unchanged. Calls predating the supplied calendar
can contribute; their diagnostic session age is a lower bound counted from the
calendar start, not an exact historical age.

For reproducing older runs, a positive integer freshness limit remains supported.
With a limit of 14, ages 0–13 qualify and age 14 does not; calls predating the
calendar are excluded because their exact freshness cannot be established.
Changing the age rule requires a new backtest, not reloading saved results.

## 5. Historical analyst skill

Retain the existing **direction-based** skill calculation. Keeping five levels
for revisions does not double the historical return of Strong Buy versus Buy.
For a historical call:

$$
X_j=\operatorname{sign}(R_j)
\left(r^{stock}_{j,30}-r^{benchmark}_{j,30}\right).
$$

Measure compounded total returns over 30 trading sessions after the first close
on or after availability, excluding that start day's return. The complete
outcome becomes available one calendar day after the end close. Known terminal
proceeds remain in zero-interest cash through the outcome horizon.

Using only completed, available outcomes, calculate:

$$
S_{a,t}=\frac{\operatorname{Mean}(X_j)}{\operatorname{Std}_{ddof=1}(X_j)}.
$$

Require at least 25 valid directional calls and standard deviation greater than
`1e-12`. Neutral calls do not enter skill history. Expanding history is the
default; `lookback_days` optionally restricts history by call availability date.
History includes linked US calls outside today's portfolio screens. Revisions
of the same dated call do not increase the historical observation count.

This is an unannualized event-level quality statistic, not portfolio Sharpe.
Overlapping calls are dependent. No score clipping, shrinkage, or claim of
statistical independence is introduced.

## 6. Daily analyst-weighted revision signal

Let $A_{i,t}$ contain analysts with a valid score and a latest available call
whose revision is defined, with no default age cutoff. Default
influence is positive-only:

$$
v_{a,t}=\max(S_{a,t},0),\qquad
Q_{i,t}=\frac{\sum_{a\in A_{i,t}}v_{a,t}\Delta R_{i,a,t}}
{\sum_{a\in A_{i,t}}|v_{a,t}|}.
$$

Signed contrarian weights are available through `negative_scores="signed"`.
The normalized revision signal lies between −4 and +4. Recompute historical
skill weights at each daily cutoff.

For an **entry**, require at least two distinct scored analysts and a
positive aggregation denominator. Zero revisions and zero-weight analysts count
toward this coverage minimum. The IBES summary `numrec` screen is not used.

For the historical z-score series, a known absence of weighted revisions
is assigned zero. Days with one contributing analyst still have a measurable
signal, but cannot trigger an entry. A missing stock price or terminal state
produces a missing signal, not a zero. The baseline is computed from daily
signals actually available on their own dates; current weights are never
retroactively applied to the previous 20 days.

## 7. Trailing z-score and entry trigger

At session $t$, use exactly the previous 20 trading sessions, excluding today:

$$
\mu_{i,t}=\operatorname{Mean}(Q_{i,t-20},\ldots,Q_{i,t-1}),\qquad
\sigma_{i,t}=\operatorname{Std}_{ddof=1}(Q_{i,t-20},\ldots,Q_{i,t-1}),
$$

$$
Z_{i,t}=\frac{Q_{i,t}-\mu_{i,t}}{\sigma_{i,t}}.
$$

Require all 20 observations and standard deviation greater than `1e-12`.
Missing sessions break the window; they are not skipped or forward-filled.
Zero-variance windows have no valid z-score.

The entry condition is a strict upward crossing:

$$
Z_{i,t}>1.68\quad\text{and}\quad Z_{i,t-1}\le1.68.
$$

Both z-scores must be valid. A first valid z-score already above threshold does
not establish a crossing. Staying above threshold does not repeatedly trigger
entries. The threshold is configurable using `entry_zscore`; 1.68 is a chosen
research setting, not a validated optimal value.

Apply universe and coverage gates at the signal close. Default behavior does
not additionally require a positive raw revision signal: an unusual increase
can cross the threshold while the raw signal is still negative. Set
`require_positive=true` to add that separate filter.

## 8. Execution, sizing, and fixed exits

Execute accepted entries at the **next session close**. A position entered at
session index $e$ exits at the close of $e+14$, earning exactly 14 subsequent
close-to-close returns. Neither a new recommendation nor a later z-score resets
the clock or causes an early discretionary exit. A terminal security event can
end the position earlier.

Retain a maximum of 100 concurrent long positions. Each new position targets
1/100 of current NAV after that day's scheduled exits. Existing positions drift
with returns and are never resized to fund or accommodate new entries. If cash
including transaction costs is insufficient, scale the batch of new buys
proportionally. Unused cash stays uninvested.

When too many stocks cross together, accept descending z-score, then ascending
PERMNO, until slots are filled. Crossings while a stock is already held are
discarded, including a crossing immediately before its scheduled exit. Capacity-
or cash-rejected crossings are also discarded, not queued. A later independent
crossing is required for re-entry.

Apply returns to existing holdings, execute scheduled exits, and then execute
new entries at each close. Reject new entries with missing/nonpositive prices,
missing execution-day rows, or terminal events as `untradable_at_execution`;
rejections consume neither cash nor a position slot and are not queued.
For held-position exits, mark invalid execution prices as errors rather than
inventing an exit price. Valid lower-ranked entry candidates can use slots left
by rejected entries. Positions still open at the requested end remain
marked to market with their planned exits reported; do not use future returns
to filter out late entries or force liquidation at the sample boundary.

## 9. Accounting and remaining assumptions

| Topic | Convention |
|---|---|
| Capital and currency | Initial USD 10 million; all prices, returns and NAV in USD |
| Holdings | Fractional amounts; distributions reinvested through total returns; no periodic resizing |
| Trading costs | 10 basis points per dollar bought or sold, including initial entries and scheduled exits; configurable |
| Cash and financing | Zero-interest cash, no borrowing; entry amounts include the cash needed for costs |
| Terminal events | Compound ordinary and delisting returns; move known terminal proceeds to cash; no modeled discretionary sale cost on terminal payout |
| Missing outcomes | Missing historical outcomes do not count toward skill and retain diagnostic status; missing held returns stop the run |
| Trading feasibility | Valid close-price execution assumed; no separate liquidity/suspension model, lot rounding, calibrated impact, taxes, or capacity claim |
| Final positions | Remain open at sample end; no terminal liquidation cost |
| Comparator | Unweighted mean of the same scored analysts' revisions; its own trailing z-score and crossings, same eligibility and execution rules |
| Performance | Daily net account and gross shadow account following the same accepted entries; zero-risk-free Sharpe and 252-session annualization explicitly labeled |
| Timing | Daily close cutoffs; conservative date-only publication and outcome lags retained |
| Legacy settings | `max_age_days` applies only to scheduled mode; revision mode defaults to `revision_freshness_sessions=null` (no age limit) |
| Data boundary | Legacy daily downloader ends at 2024-12-31; later dates require validated CIZ inputs |
| Research limits | Real-data execution streams market files and diagnostics; no established vendor identity correctness, calibrated costs, profitability, or dependence-adjusted significance |

## 10. Running and validation

From `analyst_recommendation_anomaly/`:

```bash
../.venv/bin/python -m src.run_analyst_sharpe --demo --config configs/analyst_sharpe.json
../.venv/bin/python -m pytest -q
```

The demo contains artificial five-level rating changes and daily returns. It is
an integration check, not evidence of investment performance. Licensed-data
commands and input schemas are in [README.md](README.md#analyst-sharpe-strategy-us-crsp-first).
Normalized input ratings must preserve the five levels; earlier collapsed
three-level inputs cannot reconstruct Strong Buy/Buy changes and should be
rebuilt from the raw extract.

Each run records configuration and input/code fingerprints, normalized events,
forward outcomes and status, analyst scores, daily weighted/unweighted revisions,
trailing means/standard deviations/z-scores/crossings, analyst contributions,
accepted/rejected orders, trades, holdings, completed/open positions, daily
returns and summary statistics. Failure manifests remain marked incomplete.

Tests must verify rating changes and predecessor availability, unlimited-age persistence and the optional age-13/14
boundary, the 20-session lagged baseline, strict threshold crossings, zero
variance, execution lag, exactly 14 holding returns, no resizing or duplicate
entries, capacity/cash constraints, delistings, missing outcomes, and invariance
of past signals and trades to future-data changes.

Real-data execution uses [src/streaming_strategy.py](src/streaming_strategy.py).
It maintains the same entry/exit rules and point-in-time analyst updates, while
processing one market file/security bucket at a time. Large panels are streamed
to Parquet. Absent-security dates are omitted from the stored signal panel but
remain missing in the internal rolling window, so gaps cannot be bridged. Exact
coverage/call-age aggregates and the final analyst snapshot support the
presentation without reloading full diagnostic histories. The run manifest and
progress log record execution stages and peak resident memory.

## Missing terminal payoffs (optional approximation)

The core class defaults to `missing_terminal_policy="error"` and stops on missing
held returns. The project JSON and presentation notebook explicitly select the
user-approved `"last_close"` approximation. It applies only to a terminal observation
with a finite, positive same-day close and a valid ordinary daily return. It
models selling at that close, applies the ordinary return, charges normal exit
costs, and moves the proceeds to cash. It omits the unknown after-delisting
payoff and is therefore an approximation, not repaired CRSP data. It never
fills ordinary missing held returns or missing close prices. Each affected exit
is labeled `terminal_close_proxy`; the presentation exports these positions and
marks its figures as approximate. Historical skill labels remain strict, and
raw input files are unchanged. This policy can materially affect performance.

### Verified trading-halt valuation exception

BIIB (CRSP PERMNO 76841) was halted on November 6, 2020 ([court opinion, page 12](https://www.ca1.uscourts.gov/sites/ca1/files/opnfiles/22-1773P-01A.pdf)). For an already-held position, when both price and return are missing on this specific nonterminal date, carry the prior market value and prohibit execution. CRSP’s November 9 return of −28.1666% captures the change from the November 5 close of $328.90 to $236.26; apply that return unchanged. A scheduled exit during the halt is deferred until trading resumes and marked `holding_period_after_halt`. `valuation_events.parquet` records affected strategy positions and the source. This stale valuation affects daily risk timing. Other missing observations still fail; historical analyst outcome labels and signal inputs are not filled.

### Approved off-exchange exit approximation

Under `missing_terminal_policy="last_close"`, a held stock with current CRSP `exchcd=0` and missing price, ordinary return, total return and delisting return is closed at its prior marked value on the first such session. This is a non-executable research approximation, not an observed exit. It charges exit costs (and configured short borrow fees for that session), preserves all prior P&L, and is recorded as `off_exchange_close_proxy` in trades, completed positions, and valuation events. Long and short positions use the same rule. No future observation is used to trigger it. Ordinary missing returns on listed stocks remain errors, and historical analyst labels/signals are unchanged. KEG (PERMNO 63079) on 2005-04-08 motivated this user-approved extension; local prices and returns remain absent until its 2007 return to the exchange.

### Sovereign verified halt, October 24, 2005

SOV / PERMNO 10562 is a second curated halt exception, supported by [contemporaneous reporting](https://elpais.com/economia/2005/10/25/actualidad/1130225581_850215.html). Its CRSP price and return are absent on October 24. October 25 RET is −0.085152, matching $21.38 / $23.37 − 1 within rounding. Carry held value during the halt, prohibit execution, retain the full reopening return, and defer scheduled exits until a tradable close. Record the event and source in valuation_events. No ordinary missing observations or analyst outcome labels are filled.
