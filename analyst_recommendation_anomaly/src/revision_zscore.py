"""Daily analyst-weighted revisions, trailing z-score entries, fixed session exits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analyst_sharpe import analyst_scores, summarize


def revision_signals(events, scores, quotes, calendar, cutoff, config):
    """Compare the latest known call to its immediately preceding dated call.

    Prior calls may be old, but must already be available. Withdrawals/invalid
    observations break revision continuity. Same-date versions are corrections,
    not separate calls. Reaffirmations contribute zero and replace earlier calls.
    """
    known = events[events.available_date.le(cutoff)].sort_values(
        ['permno', 'analyst_id', 'recommendation_date', 'available_date', 'event_id'])
    known = known.drop_duplicates(['permno', 'analyst_id', 'recommendation_date'], keep='last')
    groups = known.groupby(['permno', 'analyst_id'])
    known['previous_rating'] = groups.rating.shift()
    known['previous_event_type'] = groups.event_type.shift()
    known['previous_recommendation_date'] = groups.recommendation_date.shift()
    active = known.drop_duplicates(['permno', 'analyst_id'], keep='last').copy()
    active['revision'] = active.rating - active.previous_rating
    active['announcement_session'] = calendar.searchsorted(active.recommendation_date)
    active['age_sessions'] = calendar.get_loc(cutoff) - active.announcement_session
    active = active[active.event_type.eq('rating') & active.previous_event_type.eq('rating') &
                    active.age_sessions.ge(0)]
    if config.revision_freshness_sessions is not None:
        active = active[active.recommendation_date.ge(calendar[0]) &
                        active.age_sessions.lt(config.revision_freshness_sessions)]
    active = active.merge(scores, on='analyst_id', how='inner', validate='many_to_one')
    if config.strategy_mode == 'revision_signal':
        active = active[active.score.gt(0)].copy()
    active['influence'] = active.score.clip(lower=0) if config.negative_scores == 'exclude' else active.score
    active['contribution'] = active.influence * active.revision
    active['abs_influence'] = active.influence.abs()
    signal = active.groupby('permno').agg(
        n_active_scored=('analyst_id', 'nunique'), numerator=('contribution', 'sum'),
        denominator=('abs_influence', 'sum'), unweighted_score=('revision', 'mean'))
    signal['signal'] = signal.numerator / signal.denominator.replace(0, np.nan)
    signal = quotes[['permno', 'prc', 'shrcd', 'exchcd', 'is_terminal']].merge(
        signal.reset_index(), on='permno', how='outer')
    # A known absence of weighted revisions is zero, not a missing price.
    for column in ['n_active_scored', 'numerator', 'denominator', 'signal', 'unweighted_score']:
        signal[column] = signal[column].fillna(0.)
    valid_market = (signal.prc.notna() & signal.prc.gt(0) & signal.is_terminal.eq(False))
    signal.loc[~valid_market, ['signal', 'unweighted_score']] = np.nan
    universe = (valid_market & signal.prc.ge(config.min_price) &
                signal.shrcd.isin(config.allowed_share_codes) &
                signal.exchcd.isin(config.allowed_exchange_codes))
    signal['eligible'] = universe & signal.n_active_scored.ge(config.min_active_analysts) & signal.denominator.gt(0)
    signal['exclusion_reason'] = np.select(
        [~valid_market, ~universe, ~signal.n_active_scored.ge(config.min_active_analysts), ~signal.denominator.gt(0)],
        ['missing_price_or_terminal', 'universe_screen', 'insufficient_scored_analysts', 'zero_influence'], default='')
    return signal, active


def entry_strength_column(prefix, config):
    if config.strategy_mode == 'revision_signal':
        return 'signal' if prefix == 'weighted' else 'unweighted_score'
    return f'{prefix}_z'


def raw_signal_triggers(signals, config):
    """Legacy *_crossing columns hold level triggers in raw-signal mode.

    Z-score columns remain missing: raw scores are never mislabeled as z-scores.
    No previous observation or rolling-window warmup is required.
    """
    for prefix, column in [('weighted', 'signal'), ('unweighted', 'unweighted_score')]:
        for stat in ['mean', 'std', 'z']:
            signals[f'{prefix}_{stat}'] = np.nan
        signals[f'{prefix}_crossing'] = signals[column].gt(config.entry_signal_threshold)
        signals[f'{prefix}_short_crossing'] = signals[column].lt(-config.entry_signal_threshold)
    return signals


def trailing_zscores(signals, calendar, config):
    """Require all 20 immediately preceding sessions; never bridge missing days."""
    if config.strategy_mode == 'revision_signal':
        return raw_signal_triggers(signals.copy(), config)
    pieces = []
    for _, history in signals.groupby('permno', sort=True):
        history = history.set_index('date').reindex(calendar).copy()
        history.index.name = 'date'
        history['permno'] = history.permno.ffill().bfill()
        history['eligible'] = history.eligible.eq(True)
        for source, prefix in [('signal', 'weighted'), ('unweighted_score', 'unweighted')]:
            window = history[source].shift(1).rolling(config.zscore_window, min_periods=config.zscore_window)
            mean, std = window.mean(), window.std(ddof=1)
            z = (history[source] - mean) / std.where(std.gt(config.min_std))
            history[f'{prefix}_mean'] = mean
            history[f'{prefix}_std'] = std
            history[f'{prefix}_z'] = z
            history[f'{prefix}_crossing'] = (z.gt(config.entry_zscore) & z.shift(1).le(config.entry_zscore))
            history[f'{prefix}_short_crossing'] = (z.lt(-config.entry_zscore) & z.shift(1).ge(-config.entry_zscore))
        pieces.append(history.reset_index())
    if not pieces:
        raise ValueError('No daily stock observations for z-score construction')
    return pd.concat(pieces, ignore_index=True).sort_values(['date', 'permno']).reset_index(drop=True)


def daily_revision_panel(events, outcomes, stocks, bench, end, config):
    calendar = pd.DatetimeIndex(bench.loc[bench.date.le(pd.Timestamp(end)), 'date'])
    by_date = {date: group for date, group in stocks.groupby('date')}
    signals, scores_history, contributions = [], [], []
    # Compute warmup before the requested simulation start as well.
    for cutoff in calendar:
        scores = analyst_scores(outcomes, cutoff, config)
        current, active = revision_signals(events, scores, by_date.get(cutoff, stocks.iloc[:0]), calendar, cutoff, config)
        current['date'] = cutoff
        active['cutoff'] = cutoff
        scores['cutoff'] = cutoff
        signals.append(current)
        scores_history.append(scores)
        contributions.append(active)
    return (trailing_zscores(pd.concat(signals, ignore_index=True), calendar, config),
            pd.concat(scores_history, ignore_index=True), pd.concat(contributions, ignore_index=True))


def simulate_entries(signals, stocks, bench, start, end, config):
    """Execute next-close entries; exits occur entry_index + holding_sessions.

    Existing positions are never resized. Exit proceeds fund new positions.
    New buys target 1/top_n of NAV, scaled together only if cash including costs
    is insufficient. Gross shadow account follows identical accepted entries.
    """
    if config.portfolio_mode == 'long_short':
        from .long_short import simulate_long_short
        return simulate_long_short(signals, stocks, bench, start, end, config)
    calendar = pd.DatetimeIndex(bench.date)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start < calendar[0] or end > calendar[-1]:
        raise ValueError('Requested date range must lie inside the benchmark calendar')
    start_index = calendar.searchsorted(start)
    if start_index + 1 >= len(calendar) or calendar[start_index + 1] > end:
        raise ValueError('Date range contains no next-session execution')
    market = {date: group.set_index('permno') for date, group in stocks.groupby('date')}
    signal_days = {date: group for date, group in signals.groupby('date')}
    daily, trades, holdings, orders, completed, open_positions = [], [], [], [], [], []
    valuation_events = []
    bench_returns = bench.set_index('date').ret
    rate = config.cost_bps / 10000
    for strategy, prefix in [('analyst_sharpe', 'weighted'), ('unweighted', 'unweighted')]:
        pos, gross_pos, metadata = {}, {}, {}
        cash = gross_cash = config.initial_nav
        benchmark_nav = config.initial_nav
        for index in range(start_index + 1, len(calendar)):
            date, cutoff = calendar[index], calendar[index - 1]
            if date > end:
                break
            before, gross_before = cash + sum(pos.values()), gross_cash + sum(gross_pos.values())
            quotes = market.get(date, stocks.iloc[:0].set_index('permno'))
            held_before = set(pos)
            cost, traded = 0., 0.
            for permno in list(pos):
                ret, proxy_exit = _held_return(quotes, permno, date, strategy, config)
                pos[permno] *= 1 + ret
                gross_pos[permno] *= 1 + ret
                if _verified_halt(quotes, permno, date):
                    valuation_events.append(_halt_event(strategy, permno, date))
                    continue  # No execution until trading resumes.
                off_exchange = proxy_exit and _off_exchange_proxy(quotes, permno, config)
                if off_exchange:
                    valuation_events.append(_off_exchange_event(strategy, permno, date))
                terminal = bool(quotes.loc[permno, 'is_terminal']) or off_exchange
                if terminal or index >= metadata[permno]['exit_index']:
                    if not terminal:
                        _check_execution(quotes, permno, date)
                    value = pos.pop(permno)
                    fee = 0. if terminal and not proxy_exit else value * rate
                    cash += value - fee
                    gross_cash += gross_pos.pop(permno)
                    cost += fee
                    traded += 0. if terminal and not proxy_exit else value
                    info = metadata.pop(permno)
                    reason = 'off_exchange_close_proxy' if off_exchange else 'terminal_close_proxy' if proxy_exit else 'terminal' if terminal else ('holding_period_after_halt' if index > info['exit_index'] else 'holding_period')
                    completed.append(dict(strategy=strategy, permno=permno, **info, exit_date=date,
                                          exit_reason=reason, held_sessions=index - info['entry_index']))
                    trades.append(dict(strategy=strategy, date=date, permno=permno, action='exit',
                                       reason=reason, value=value, cost_dollars=fee))
            today = signal_days.get(cutoff, signals.iloc[:0])
            candidates = today[today.eligible & today[f'{prefix}_crossing']].sort_values(
                [entry_strength_column(prefix, config), 'permno'], ascending=[False, True])
            if config.require_positive:
                source = 'signal' if prefix == 'weighted' else 'unweighted_score'
                candidates = candidates[candidates[source].gt(0)]
            accepted = []
            free = config.top_n - len(pos)
            for candidate in candidates.itertuples(index=False):
                permno = int(candidate.permno)
                # A crossing while held is discarded, including one just before exit.
                reason = 'already_held' if permno in held_before else 'capacity' if len(accepted) >= free else 'entry'
                if reason == 'entry' and cash <= 1e-8:
                    reason = 'no_cash'
                if reason == 'entry' and not _can_execute(quotes, permno):
                    reason = 'untradable_at_execution'
                orders.append(dict(strategy=strategy, cutoff=cutoff, execution_date=date,
                                   permno=permno, zscore=getattr(candidate, f'{prefix}_z'), entry_signal=getattr(candidate, 'signal' if prefix=='weighted' else 'unweighted_score'), status=reason))
                if reason == 'entry':
                    _check_execution(quotes, permno, date)
                    accepted.append(permno)
            nav_before_buys = cash + sum(pos.values())
            gross_nav_before_buys = gross_cash + sum(gross_pos.values())
            if accepted:
                amount = min(nav_before_buys / config.top_n, cash / (len(accepted) * (1 + rate)))
                gross_amount = min(gross_nav_before_buys / config.top_n, gross_cash / len(accepted))
                for permno in accepted:
                    fee = amount * rate
                    cash -= amount + fee
                    gross_cash -= gross_amount
                    pos[permno], gross_pos[permno] = amount, gross_amount
                    exit_index = index + config.holding_sessions
                    metadata[permno] = {'entry_date': date, 'entry_index': index, 'exit_index': exit_index,
                                       'planned_exit_date': calendar[exit_index] if exit_index < len(calendar) else pd.NaT}
                    cost += fee
                    traded += amount
                    trades.append(dict(strategy=strategy, date=date, permno=permno, action='entry',
                                       reason='raw_signal_threshold' if config.strategy_mode=='revision_signal' else 'zscore_crossing', value=amount, cost_dollars=fee))
            if cash < -1e-7 or gross_cash < -1e-7:
                raise AssertionError('Negative cash')
            cash, gross_cash = max(0., cash), max(0., gross_cash)
            nav, gross_nav = cash + sum(pos.values()), gross_cash + sum(gross_pos.values())
            if not np.isfinite(nav) or nav <= 0 or not np.isfinite(gross_nav) or gross_nav <= 0:
                raise ValueError(f'Portfolio NAV exhausted or invalid on {date.date()}')
            benchmark_nav *= 1 + bench_returns.loc[date]
            daily.append(dict(strategy=strategy, date=date, net_return=nav / before - 1,
                              gross_return=gross_nav / gross_before - 1, nav=nav, gross_nav=gross_nav,
                              benchmark_return=bench_returns.loc[date], benchmark_nav=benchmark_nav,
                              cost_dollars=cost, turnover=traded / (nav + cost), cash=cash, n_holdings=len(pos)))
            holdings.extend(dict(strategy=strategy, date=date, permno=k, value=v, weight=v / nav,
                                 entry_date=metadata[k]['entry_date'], planned_exit_date=metadata[k]['planned_exit_date'])
                            for k, v in pos.items())
        open_positions.extend(dict(strategy=strategy, permno=k, value=v, **metadata[k]) for k, v in pos.items())
    return {'valuation_events': pd.DataFrame(valuation_events, columns=HALT_COLUMNS),
            'daily_results': pd.DataFrame(daily),
            'trades': pd.DataFrame(trades, columns=['strategy', 'date', 'permno', 'action', 'reason', 'value', 'cost_dollars']),
            'holdings': pd.DataFrame(holdings, columns=['strategy', 'date', 'permno', 'value', 'weight', 'entry_date', 'planned_exit_date']),
            'orders': pd.DataFrame(orders, columns=['strategy', 'cutoff', 'execution_date', 'permno', 'zscore', 'entry_signal', 'status']),
            'completed_positions': pd.DataFrame(completed, columns=['strategy', 'permno', 'entry_date', 'entry_index', 'exit_index', 'planned_exit_date', 'exit_date', 'exit_reason', 'held_sessions']),
            'open_positions': pd.DataFrame(open_positions, columns=['strategy', 'permno', 'value', 'entry_date', 'entry_index', 'exit_index', 'planned_exit_date'])}


HALT_SOURCE = 'https://www.ca1.uscourts.gov/sites/ca1/files/opnfiles/22-1773P-01A.pdf'
VERIFIED_HALTS = {
    76841: {pd.Timestamp('2020-11-06'): HALT_SOURCE},
    10562: {pd.Timestamp('2005-10-24'): 'https://elpais.com/economia/2005/10/25/actualidad/1130225581_850215.html'},
}
HALT_COLUMNS = ['strategy', 'date', 'permno', 'event', 'source']


def _verified_halt(quotes, permno, date):
    """Explicitly verified halts only; never fill general gaps or analyst labels.

    BIIB: Nov 5 2020 close 328.90 to Nov 9 close 236.26, RET -0.281666.
    SOV: Oct 21 2005 close 23.37 to Oct 25 close 21.38, RET -0.085152.
    Missing execution prices stay missing. Reopening returns apply unchanged.
    """
    known_dates = VERIFIED_HALTS.get(int(permno))
    if known_dates is None or pd.Timestamp(date) not in known_dates:
        return False
    if permno not in quotes.index:
        return False
    row = quotes.loc[permno]
    return (not bool(row['is_terminal']) and pd.isna(row['ret_total'])
            and pd.isna(row['prc']) and pd.isna(row.get('ret', np.nan))
            and pd.isna(row.get('dlret', np.nan)))


def _halt_event(strategy, permno, date):
    return dict(strategy=strategy, date=date, permno=permno,
                event='verified_halt_carry', source=VERIFIED_HALTS[int(permno)][pd.Timestamp(date)])


def _off_exchange_proxy(quotes, permno, config):
    """Opt-in last-value liquidation when exchange code 0 has no observation.

    This is an explicitly non-executable research approximation, not a measured
    return or a halt carry. It uses only the current row and prior held value.
    """
    if config.missing_terminal_policy != 'last_close' or permno not in quotes.index:
        return False
    row = quotes.loc[permno]
    exchange = row.get('exchcd', np.nan)
    return (pd.notna(exchange) and exchange == 0 and not bool(row['is_terminal'])
            and all(pd.isna(row.get(k, np.nan)) for k in ['prc', 'ret_total', 'ret', 'dlret']))


def _off_exchange_event(strategy, permno, date):
    return dict(strategy=strategy, date=date, permno=permno,
                event='off_exchange_close_proxy',
                source='CRSP current row: exchcd=0, price/returns missing; approved prior-value exit approximation')


def _held_return(quotes, permno, date, strategy, config):
    """Strict returns except a documented halt and opt-in exit approximations."""
    if _verified_halt(quotes, permno, date):
        return 0., False
    if permno in quotes.index:
        row = quotes.loc[permno]
        value = row['ret_total']
        if pd.notna(value) and np.isfinite(value):
            return value, False
        if _off_exchange_proxy(quotes, permno, config):
            return 0., True
        if bool(row['is_terminal']) and config.missing_terminal_policy == 'last_close':
            ordinary = row.get('ret', np.nan)
            price = row['prc']
            if (pd.notna(ordinary) and np.isfinite(ordinary) and ordinary >= -1
                    and pd.notna(price) and np.isfinite(price) and price > 0):
                # Model liquidation at the recorded close, retaining RET (including
                # distributions) but omitting the unknown after-delisting payoff.
                return ordinary, True
        if bool(row['is_terminal']):
            raise ValueError(f'Missing held return: {strategy}, {permno}, {date.date()}; '
                             'terminal payoff unavailable. Supply verified delisting data or explicitly '
                             'set missing_terminal_policy="last_close" for a flagged approximation '
                             '(requires a valid same-day close and ordinary return).')
    raise ValueError(f'Missing held return: {strategy}, {permno}, {date.date()}')


def _can_execute(quotes, permno):
    return (permno in quotes.index and pd.notna(quotes.loc[permno, 'prc'])
            and np.isfinite(quotes.loc[permno, 'prc'])
            and quotes.loc[permno, 'prc'] > 0 and not quotes.loc[permno, 'is_terminal'])


def _check_execution(quotes, permno, date):
    if not _can_execute(quotes, permno):
        raise ValueError(f'Cannot execute {permno} on {date.date()}; resolve price/tradability')


def run_revision_strategy(events, outcomes, stocks, bench, start, end, config):
    signals, scores, contributions = daily_revision_panel(events, outcomes, stocks, bench, end, config)
    results = simulate_entries(signals, stocks, bench, start, end, config)
    results.update(events=events, outcomes=outcomes, signals=signals, analyst_scores=scores, contributions=contributions)
    results['summary'] = summarize(results['daily_results'], config)
    return results
