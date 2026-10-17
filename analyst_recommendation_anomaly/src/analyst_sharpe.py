"""Point-in-time analyst quality strategy on daily, historically linked CRSP data.

Dates represent US market session dates, not intraday timestamps. Signal cutoffs
are after the close; execution is at the following session's close.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import combine_crsp_returns
from .signals import link_recommendations_to_crsp


@dataclass(frozen=True)
class AnalystSharpeConfig:
    strategy_mode: str = "revision_zscore"
    portfolio_mode: str = "long_only"
    short_borrow_bps_annual: float = 0.0
    revision_freshness_sessions: int | None = None
    zscore_window: int = 20
    entry_zscore: float = 1.68
    entry_signal_threshold: float = 1.0
    holding_sessions: int = 14
    missing_terminal_policy: str = "error"  # opt-in terminal/off-exchange last-close approximations
    horizon_sessions: int = 30
    min_calls: int = 25
    min_active_analysts: int = 2
    max_age_days: int = 180
    lookback_days: int | None = None
    negative_scores: str = 'exclude'  # or signed
    top_n: int = 100
    min_price: float = 5.0
    allowed_share_codes: tuple[int, ...] = (10, 11)
    allowed_exchange_codes: tuple[int, ...] = (1, 2, 3)
    cost_bps: float = 10.0
    initial_nav: float = 10_000_000.0
    min_std: float = 1e-12
    publication_lag_days: int = 1
    outcome_lag_days: int = 1
    require_positive: bool = False

    def __post_init__(self):
        for name in ('horizon_sessions', 'min_calls', 'min_active_analysts',
                     'max_age_days', 'top_n', 'publication_lag_days', 'outcome_lag_days',
                     'zscore_window', 'holding_sessions'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        freshness = self.revision_freshness_sessions
        if freshness is not None and (not isinstance(freshness, int) or isinstance(freshness, bool) or freshness < 1):
            raise ValueError('revision_freshness_sessions must be a positive integer or None')
        if self.missing_terminal_policy not in ('error', 'last_close'):
            raise ValueError('missing_terminal_policy must be error or last_close')
        if self.portfolio_mode not in ('long_only', 'long_short'):
            raise ValueError('portfolio_mode must be long_only or long_short')
        if not np.isfinite(self.short_borrow_bps_annual) or self.short_borrow_bps_annual < 0:
            raise ValueError('short_borrow_bps_annual must be finite and nonnegative')
        if self.portfolio_mode == 'long_short' and (self.strategy_mode not in ('revision_zscore', 'revision_signal') or self.require_positive):
            raise ValueError('long_short requires a revision strategy and require_positive=False')
        if self.strategy_mode not in ("revision_zscore", "revision_signal", "scheduled"):
            raise ValueError("Unknown strategy_mode")
        if not np.isfinite(self.entry_signal_threshold) or self.entry_signal_threshold <= 0:
            raise ValueError('entry_signal_threshold must be finite and positive')
        if self.strategy_mode == 'revision_signal' and self.negative_scores != 'exclude':
            raise ValueError('revision_signal requires strictly positive Sharpe contributors')
        if self.zscore_window < 2 or not np.isfinite(self.entry_zscore) or self.entry_zscore <= 0:
            raise ValueError("zscore_window must be >= 2 and entry_zscore finite and positive")
        if self.min_calls < 2:
            raise ValueError('Sample standard deviation needs at least two calls')
        if self.lookback_days is not None and self.lookback_days < 1:
            raise ValueError('lookback_days must be positive or None')
        if self.negative_scores not in ('exclude', 'signed'):
            raise ValueError('negative_scores must be exclude or signed')
        if not np.isfinite(self.cost_bps) or not 0 <= self.cost_bps < 10000:
            raise ValueError('cost_bps must be in [0, 10000)')
        for name in ('initial_nav', 'min_std'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.min_price) or self.min_price < 0:
            raise ValueError('min_price must be finite and nonnegative')


def _dates(frame, columns):
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors='raise').dt.normalize()
        if frame[column].isna().any() or frame[column].dt.tz is not None:
            raise ValueError(f'{column} must contain nonmissing timezone-naive dates')
    return frame


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize a linked event panel; conflicts terminate state, not revert it.

    Required: permno, analyst_id, recommendation_date, available_date, rating.
    Rating preserves five levels (-2, -1, 0, 1, 2), higher is more bullish. event_type defaults to rating;
    withdraw and invalid terminate the prior recommendation state.
    """
    frame = _dates(events, ['recommendation_date', 'available_date'])
    if frame[['permno', 'analyst_id']].isna().any().any():
        raise ValueError('Missing security or analyst identifiers')
    if (frame.available_date < frame.recommendation_date).any():
        raise ValueError('Availability cannot precede recommendation date')
    frame['analyst_id'] = frame.analyst_id.astype(str)
    frame['permno'] = pd.to_numeric(frame.permno, errors='raise').astype('int64')
    if 'event_type' not in frame:
        frame['event_type'] = 'rating'
    frame['event_type'] = frame.event_type.fillna('rating')
    if not frame.event_type.isin(['rating', 'withdraw', 'invalid']).all():
        raise ValueError('Unknown event_type')
    frame['rating'] = pd.to_numeric(frame.rating, errors='coerce')
    bad = frame.event_type.eq('rating') & ~frame.rating.isin([-2, -1, 0, 1, 2])
    frame.loc[bad, 'event_type'] = 'invalid'
    frame.loc[frame.event_type.ne('rating'), 'rating'] = np.nan
    columns = ['permno', 'analyst_id', 'recommendation_date', 'available_date', 'rating', 'event_type']
    frame = frame[columns].drop_duplicates()
    keys = ['permno', 'analyst_id', 'recommendation_date', 'available_date']
    conflicts = frame.duplicated(keys, keep=False)
    frame.loc[conflicts, 'event_type'] = 'invalid'
    frame.loc[conflicts, 'rating'] = np.nan
    frame = frame.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    # Stable IDs do not change when later data is appended.
    frame['event_id'] = pd.util.hash_pandas_object(frame[columns], index=False).astype(str)
    return frame


def load_ibes_extract(extract: Path, names: pd.DataFrame,
                      config: AnalystSharpeConfig) -> tuple[pd.DataFrame, dict]:
    """Load one completed extract. Use conservative date-only availability.

    Each broker/analyst pair is an independent analyst identity. Max of vendor
    announcement, activation and review dates + lag avoids assuming that a
    retrospectively revised record was observable on its announcement date.
    """
    extract = Path(extract)
    manifest = json.loads((extract / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Analyst extraction is incomplete')
    pieces = []
    stats = {'raw_rows': 0, 'us_identified_rows': 0, 'linked_rows': 0}
    for item in manifest['files']:
        raw = pd.read_parquet(extract / item['file'])
        stats['raw_rows'] += len(raw)
        if 'usfirm' in raw:
            raw = raw[pd.to_numeric(raw.usfirm, errors='coerce').eq(1)].copy()
        raw = raw.dropna(subset=['amaskcd', 'emaskcd', 'anndats', 'cusip'])
        stats['us_identified_rows'] += len(raw)
        raw['statpers'] = pd.to_datetime(raw.anndats).dt.normalize()
        dates = pd.concat([pd.to_datetime(raw[c], errors='coerce') for c in
                           ('anndats', 'actdats', 'revdats') if c in raw], axis=1)
        raw['available_date'] = dates.max(axis=1).dt.normalize() + pd.Timedelta(config.publication_lag_days, unit="D").as_unit("ns")
        raw['analyst_id'] = raw.emaskcd.astype(str) + ':' + raw.amaskcd.astype(str)
        raw['rating'] = pd.to_numeric(raw.ireccd, errors='coerce').map({1: 2, 2: 1, 3: 0, 4: -1, 5: -2})
        raw['event_type'] = np.where(raw.rating.notna(), 'rating', 'invalid')
        raw['recommendation_date'] = raw.statpers
        linked = link_recommendations_to_crsp(raw, names)
        stats['linked_rows'] += len(linked)
        pieces.append(linked[['permno', 'analyst_id', 'recommendation_date',
                              'available_date', 'rating', 'event_type']])
    if not pieces:
        raise ValueError('No files in analyst extract')
    events = normalize_events(pd.concat(pieces, ignore_index=True))
    stats['normalized_events'] = len(events)
    stats['invalid_state_events'] = int(events.event_type.ne('rating').sum())
    stats['identity_scope'] = 'single extract, broker:analyst pair'
    return events, stats


def prepare_market(daily: pd.DataFrame, benchmark: pd.DataFrame):
    """Daily input uses CRSP RET/DLRET or explicit ret_total; benchmark uses ret.

    is_terminal marks the last economic stock outcome, whose proceeds move to
    zero-interest cash. Unresolved terminal returns are errors when held.
    """
    stocks = _dates(daily, ['date'])
    bench = _dates(benchmark, ['date']).sort_values('date').reset_index(drop=True)
    if stocks.duplicated(['permno', 'date']).any() or bench.date.duplicated().any():
        raise ValueError('Duplicate security-session or benchmark-session rows')
    if len(bench) < 2:
        raise ValueError('At least two benchmark sessions are required')
    if bench.date.diff().dropna().gt(pd.Timedelta(7, unit="D").as_unit("ns")).any():
        raise ValueError('Benchmark is not a continuous daily calendar; monthly data cannot be used')
    if not stocks.date.isin(bench.date).all():
        raise ValueError('Stock dates outside benchmark trading calendar')
    stocks['permno'] = pd.to_numeric(stocks.permno, errors='raise').astype('int64')
    if 'ret_total' not in stocks:
        stocks['ret_total'] = combine_crsp_returns(stocks.ret, stocks.dlret)
    for col in ['ret_total', 'prc', 'shrcd', 'exchcd']:
        stocks[col] = pd.to_numeric(stocks[col], errors='coerce')
    stocks['prc'] = stocks.prc.abs()
    if 'is_terminal' not in stocks:
        stocks['is_terminal'] = False
    if not stocks.is_terminal.isin([True, False, 0, 1]).all():
        raise ValueError('is_terminal must be a nonmissing boolean')
    stocks['is_terminal'] = stocks.is_terminal.astype(bool)
    if 'dlret' in stocks:
        # An ordinary daily return cannot stand in for an unknown terminal loss.
        missing_terminal = stocks.is_terminal & pd.to_numeric(stocks.dlret, errors='coerce').isna()
        stocks.loc[missing_terminal, 'ret_total'] = np.nan
    bench['ret'] = pd.to_numeric(bench.ret, errors='coerce')
    if not np.isfinite(bench.ret).all() or bench.ret.lt(-1).any():
        raise ValueError('Missing or invalid benchmark returns')
    if np.isinf(stocks.ret_total).any() or stocks.ret_total.lt(-1).any():
        raise ValueError('Invalid stock returns')
    for permno, group in stocks.groupby('permno'):
        terminal = group.loc[group.is_terminal, 'date']
        if len(terminal) > 1 or (len(terminal) and group.date.max() > terminal.iloc[0]):
            raise ValueError(f'Rows after terminal event for {permno}; resolve identifier lifecycle')
    return stocks.sort_values(['date', 'permno']).reset_index(drop=True), bench


def build_outcomes(events, stocks, bench, config):
    """Vectorized session-window outcomes; retain unavailable/missing labels."""
    calendar = pd.DatetimeIndex(bench.date)
    horizon = config.horizon_sessions
    result = events.copy()
    result['start_date'] = pd.NaT
    result['end_date'] = pd.NaT
    result['outcome_available_date'] = pd.NaT
    result['stock_return'] = np.nan
    result['benchmark_return'] = np.nan
    result['outcome_status'] = 'not_directional'
    stock_groups = {key: group.set_index('date') for key, group in stocks.groupby('permno')}
    bret = bench.ret.to_numpy()
    for permno, group in result.groupby('permno'):
        positions = group.index[group.event_type.eq('rating') & group.rating.ne(0)]
        if not len(positions):
            continue
        start = calendar.searchsorted(result.loc[positions, 'available_date'])
        end = start + horizon
        in_range = end < len(calendar)
        result.loc[positions, 'outcome_status'] = 'not_matured_in_data'
        positions, start, end = positions[in_range], start[in_range], end[in_range]
        if not len(positions):
            continue
        result.loc[positions, 'start_date'] = calendar[start].to_numpy()
        result.loc[positions, 'end_date'] = calendar[end].to_numpy()
        result.loc[positions, 'outcome_available_date'] = (calendar[end] + pd.Timedelta(config.outcome_lag_days, unit="D").as_unit("ns")).to_numpy()
        result.loc[positions, 'outcome_status'] = 'missing_stock_outcome'
        if permno not in stock_groups:
            continue
        history = stock_groups[permno]
        aligned = history.reindex(calendar)
        returns = aligned.ret_total.to_numpy(copy=True)
        terminal = history.index[history.is_terminal]
        valid_start = aligned.prc.notna().to_numpy()[start]
        if len(terminal):
            terminal_index = calendar.get_loc(terminal[0])
            returns[terminal_index + 1:] = 0.0  # Proceeds held in cash to label horizon.
            valid_start &= start < terminal_index
        # Window [start+1, start+horizon] excludes the execution-day return.
        sw = np.lib.stride_tricks.sliding_window_view(returns[1:], horizon)
        bw = np.lib.stride_tricks.sliding_window_view(bret[1:], horizon)
        sr = np.prod(1 + sw[start], axis=1) - 1
        br = np.prod(1 + bw[start], axis=1) - 1
        valid = valid_start & np.isfinite(sr)
        good = positions[valid]
        result.loc[good, 'stock_return'] = sr[valid]
        result.loc[good, 'benchmark_return'] = br[valid]
        result.loc[good, 'outcome_status'] = 'complete'
    # Skill remains direction-based even though current revisions retain all five levels.
    result['analyst_return'] = np.sign(result.rating) * (result.stock_return - result.benchmark_return)
    return result


def analyst_scores(outcomes, cutoff, config):
    # Revised versions of the same dated call must not count as extra calls.
    # The latest known version can invalidate a formerly valid historical label.
    known = outcomes[outcomes.available_date.le(cutoff)].sort_values('available_date')
    known = known.drop_duplicates(['analyst_id', 'permno', 'recommendation_date'], keep='last')
    history = known[known.outcome_status.eq('complete') &
                    known.outcome_available_date.le(cutoff) & known.rating.ne(0)]
    if config.lookback_days is not None:
        history = history[history.available_date.ge(cutoff - pd.Timedelta(config.lookback_days, unit="D").as_unit("ns"))]
    scores = history.groupby('analyst_id').analyst_return.agg(n_calls='count', mean='mean', std='std')
    scores = scores[scores.n_calls.ge(config.min_calls) & scores['std'].gt(config.min_std)].copy()
    scores['score'] = scores['mean'] / scores['std']
    return scores[np.isfinite(scores.score)].reset_index()


def stock_signals(events, scores, market_at_cutoff, cutoff, config):
    active = events[events.available_date.le(cutoff)].sort_values(
        ['recommendation_date', 'available_date', 'event_id'])
    active = active.drop_duplicates(['permno', 'analyst_id'], keep='last')
    active = active[active.event_type.eq('rating') &
                    active.recommendation_date.ge(cutoff - pd.Timedelta(config.max_age_days, unit="D").as_unit("ns"))]
    active = active.merge(scores, on='analyst_id', how='inner', validate='many_to_one')
    active['influence'] = active.score.clip(lower=0) if config.negative_scores == 'exclude' else active.score
    active['contribution'] = active.influence * active.rating
    active['abs_influence'] = active.influence.abs()
    signal = active.groupby('permno').agg(
        n_active_scored=('analyst_id', 'nunique'), numerator=('contribution', 'sum'),
        denominator=('abs_influence', 'sum'), unweighted_score=('rating', 'mean'))
    signal['signal'] = signal.numerator / signal.denominator.replace(0, np.nan)
    signal = signal.reset_index().merge(
        market_at_cutoff[['permno', 'prc', 'shrcd', 'exchcd', 'is_terminal']], on='permno', how='outer')
    signal['eligible'] = (signal.n_active_scored.ge(config.min_active_analysts) &
                          signal.denominator.gt(0) & signal.prc.ge(config.min_price) &
                          signal.shrcd.isin(config.allowed_share_codes) &
                          signal.exchcd.isin(config.allowed_exchange_codes) &
                          signal.is_terminal.eq(False))
    signal['exclusion_reason'] = np.select(
        [signal.prc.isna(), ~signal.shrcd.isin(config.allowed_share_codes),
         ~signal.exchcd.isin(config.allowed_exchange_codes), signal.prc.lt(config.min_price),
         signal.is_terminal.eq(True), ~signal.n_active_scored.ge(config.min_active_analysts),
         ~signal.denominator.gt(0)],
        ['missing_cutoff_price', 'share_code', 'exchange_code', 'price', 'terminal',
         'insufficient_scored_analysts', 'zero_influence'], default='')
    return signal, active


def make_schedule(calendar, start, end):
    """Every 14 calendar days from start, rolled forward; execute next close."""
    calendar = pd.DatetimeIndex(calendar)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start < calendar.min() or end > calendar.max():
        raise ValueError('Requested date range must lie inside the benchmark calendar')
    schedule = []
    for anchor in pd.date_range(start, end, freq='14D'):
        i = calendar.searchsorted(anchor)
        if i + 1 < len(calendar) and calendar[i + 1] <= end:
            schedule.append((calendar[i], calendar[i + 1]))
    return pd.DataFrame(schedule, columns=['cutoff', 'execution_date']).drop_duplicates()


def build_targets(events, outcomes, stocks, bench, start, end, config):
    schedule = make_schedule(bench.date, start, end)
    if schedule.empty:
        raise ValueError('Date range contains no executable rebalance')
    by_date = {date: frame for date, frame in stocks.groupby('date')}
    targets, signals, score_history, contributions = [], [], [], []
    for row in schedule.itertuples(index=False):
        scores = analyst_scores(outcomes, row.cutoff, config)
        current, active = stock_signals(events, scores, by_date.get(row.cutoff, stocks.iloc[:0]), row.cutoff, config)
        current['cutoff'] = row.cutoff
        current['execution_date'] = row.execution_date
        scores['cutoff'] = row.cutoff
        active['cutoff'] = row.cutoff
        signals.append(current)
        score_history.append(scores)
        contributions.append(active)
        for strategy, column in [('analyst_sharpe', 'signal'), ('unweighted', 'unweighted_score')]:
            eligible = current[current.eligible]
            if config.require_positive:
                eligible = eligible[eligible[column].gt(0)]
            selected = eligible.sort_values([column, 'permno'], ascending=[False, True]).head(config.top_n)
            for item in selected.itertuples(index=False):
                targets.append({'strategy': strategy, 'cutoff': row.cutoff,
                                'execution_date': row.execution_date, 'permno': int(item.permno),
                                'weight': 1 / config.top_n})
    return (schedule, pd.DataFrame(targets, columns=['strategy', 'cutoff', 'execution_date', 'permno', 'weight']),
            pd.concat(signals, ignore_index=True), pd.concat(score_history, ignore_index=True),
            pd.concat(contributions, ignore_index=True))


def rebalance_account(positions, cash, weights, rate):
    """Solve post-cost NAV exactly so target weights require no borrowing."""
    nav = cash + sum(positions.values())
    if nav <= 0:
        raise ValueError('Portfolio NAV exhausted')
    keys = sorted(set(positions) | set(weights))
    if any(w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-12:
        raise ValueError('Targets must be unlevered long-only weights')
    def traded(value):
        return sum(abs(weights.get(k, 0) * value - positions.get(k, 0)) for k in keys)
    low, high = 0., nav
    for _ in range(70):
        mid = (low + high) / 2
        if mid + rate * traded(mid) > nav:
            high = mid
        else:
            low = mid
    post_nav = (low + high) / 2
    cost = rate * traded(post_nav)
    trade_rows = [{'permno': k, 'pretrade_value': positions.get(k, 0),
                   'trade_value': weights.get(k, 0) * post_nav - positions.get(k, 0),
                   'target_weight': weights.get(k, 0)} for k in keys]
    new_positions = {k: w * post_nav for k, w in weights.items() if w > 0}
    new_cash = nav - cost - sum(new_positions.values())
    if new_cash < -1e-7:
        raise AssertionError('Negative cash after rebalance')
    return new_positions, max(new_cash, 0.), cost, traded(post_nav) / nav, trade_rows


def simulate(targets, schedule, stocks, bench, end, config):
    """Strict accounting: stop on unknown held returns or unexecutable trades."""
    execution_dates = set(schedule.execution_date)
    dates = bench[bench.date.gt(schedule.cutoff.min()) & bench.date.le(pd.Timestamp(end))]
    market = {date: frame.set_index('permno') for date, frame in stocks.groupby('date')}
    rows, trades, holdings = [], [], []
    for strategy in ['analyst_sharpe', 'unweighted']:
        target_groups = {date: frame.set_index('permno').weight.to_dict() for date, frame in
                         targets[targets.strategy.eq(strategy)].groupby('execution_date')}
        pos, gross_pos = {}, {}
        cash = gross_cash = config.initial_nav
        benchmark_nav = config.initial_nav
        for day in dates.itertuples(index=False):
            before, gross_before = cash + sum(pos.values()), gross_cash + sum(gross_pos.values())
            quotes = market.get(day.date, stocks.iloc[:0].set_index('permno'))
            for positions, is_gross in [(pos, False), (gross_pos, True)]:
                for permno in list(positions):
                    if permno not in quotes.index or not np.isfinite(quotes.loc[permno, 'ret_total']):
                        raise ValueError(f'Missing held return: {strategy}, {permno}, {day.date.date()}; no imputation performed')
                    positions[permno] *= 1 + quotes.loc[permno, 'ret_total']
                    if quotes.loc[permno, 'is_terminal']:
                        if is_gross:
                            gross_cash += positions.pop(permno)
                        else:
                            cash += positions.pop(permno)
            cost = turnover = 0.
            if day.date in execution_dates:
                weights = target_groups.get(day.date, {})
                for permno in set(pos) | set(weights):
                    if permno not in quotes.index or not np.isfinite(quotes.loc[permno, 'prc']) or quotes.loc[permno, 'prc'] <= 0 or quotes.loc[permno, 'is_terminal']:
                        raise ValueError(f'Cannot execute {permno} on {day.date.date()}; resolve price/tradability')
                pos, cash, cost, turnover, new_trades = rebalance_account(pos, cash, weights, config.cost_bps / 10000)
                gross_pos, gross_cash, _, _, _ = rebalance_account(gross_pos, gross_cash, weights, 0.)
                trades.extend(dict(strategy=strategy, date=day.date, **trade) for trade in new_trades)
            nav, gross_nav = cash + sum(pos.values()), gross_cash + sum(gross_pos.values())
            if not np.isfinite(nav) or nav <= 0 or not np.isfinite(gross_nav) or gross_nav <= 0:
                raise ValueError(f'Portfolio NAV exhausted or invalid on {day.date.date()}')
            benchmark_nav *= 1 + day.ret
            rows.append({'strategy': strategy, 'date': day.date, 'net_return': nav / before - 1,
                         'gross_return': gross_nav / gross_before - 1, 'nav': nav, 'gross_nav': gross_nav,
                         'benchmark_return': day.ret, 'benchmark_nav': benchmark_nav, 'cost_dollars': cost,
                         'turnover': turnover, 'cash': cash, 'n_holdings': len(pos)})
            holdings.extend({'strategy': strategy, 'date': day.date, 'permno': k,
                             'value': v, 'weight': v / nav} for k, v in pos.items())
    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(holdings)


def summarize(daily, config):
    rows = []
    for strategy, frame in daily.groupby('strategy'):
        r, gross = frame.net_return, frame.gross_return
        vol = r.std(ddof=1)
        wealth = frame.nav / config.initial_nav
        elapsed = len(frame) / 252
        rows.append({'strategy': strategy, 'sessions': len(frame),
                     'total_return': wealth.iloc[-1] - 1,
                     'gross_total_return': frame.gross_nav.iloc[-1] / config.initial_nav - 1,
                     'benchmark_total_return': frame.benchmark_nav.iloc[-1] / config.initial_nav - 1,
                     'annualized_return_252': wealth.iloc[-1] ** (1 / elapsed) - 1,
                     'annualized_volatility': vol * np.sqrt(252),
                     'sharpe_zero_rf': r.mean() / vol * np.sqrt(252) if vol > 0 else np.nan,
                     'gross_sharpe_zero_rf': gross.mean() / gross.std(ddof=1) * np.sqrt(252) if gross.std(ddof=1) > 0 else np.nan,
                     'max_drawdown': (wealth / wealth.cummax().clip(lower=1) - 1).min(),
                     'total_cost_dollars': frame.cost_dollars.sum(),
                     'total_traded_nav': frame.turnover.sum(),
                     'mean_holdings': frame.n_holdings.mean(),
                     'mean_cash_weight': (frame.cash / frame.nav).mean()})
    return pd.DataFrame(rows)


def run_strategy(events, daily, benchmark, start, end, config=None):
    config = config or AnalystSharpeConfig()
    events = normalize_events(events)
    stocks, bench = prepare_market(daily, benchmark)
    outcomes = build_outcomes(events, stocks, bench, config)
    if config.strategy_mode in ("revision_zscore", "revision_signal"):
        from .revision_zscore import run_revision_strategy
        return run_revision_strategy(events, outcomes, stocks, bench, start, end, config)
    schedule, targets, signals, scores, contributions = build_targets(events, outcomes, stocks, bench, start, end, config)
    results, trades, holdings = simulate(targets, schedule, stocks, bench, end, config)
    return {'events': events, 'outcomes': outcomes, 'schedule': schedule, 'targets': targets,
            'signals': signals, 'analyst_scores': scores, 'contributions': contributions,
            'daily_results': results, 'trades': trades, 'holdings': holdings,
            'summary': summarize(results, config)}
