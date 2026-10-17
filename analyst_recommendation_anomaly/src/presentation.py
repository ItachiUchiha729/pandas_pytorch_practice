"""Analysis tables for the revision z-score research notebook.

All calculations use the complete common daily sample. Missing returns are an
error, not a reason to drop a day. Cash and risk-free rate are zero by design.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {'analyst_sharpe': 'Analyst weighted', 'unweighted': 'Unweighted revisions', 'benchmark': 'CRSP market benchmark'}


def load_run(path):
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Run is incomplete; inspect its manifest error before presenting results.')
    if manifest['config'].get('strategy_mode') not in ('revision_zscore','revision_signal'):
        raise ValueError('This notebook requires a revision_zscore run, not the legacy scheduled strategy.')
    small = ['daily_results', 'orders', 'trades', 'holdings', 'completed_positions', 'open_positions']
    if (path / 'valuation_events.parquet').exists():
        small.append('valuation_events')
    if manifest.get('engine') == 'streaming':
        # Exact aggregate diagnostics + one stock walkthrough, never 47M rows in RAM.
        frames = {name: pd.read_parquet(path / f'{name}.parquet') for name in small}
        accepted = frames['orders'].query("strategy == 'analyst_sharpe' and status == 'entry'")
        if len(accepted):
            example = int(accepted.permno.value_counts().index[0])
        else:
            import pyarrow.parquet as pq
            first = next(pq.ParquetFile(path / 'signals.parquet').iter_batches(batch_size=1, columns=['permno']), None)
            if first is None:
                raise ValueError('No signal observations')
            example = int(first.column(0)[0].as_py())
        frames['signals'] = pd.read_parquet(path / 'signals.parquet', filters=[('permno', '=', example)])
        frames['analyst_scores'] = pd.read_parquet(path / 'latest_analyst_scores.parquet')
        frames['outcomes'] = pd.read_parquet(path / 'outcomes.parquet', columns=['outcome_status'])
        frames['contributions'] = pd.DataFrame(columns=['age_sessions','available_date','cutoff'])
        frames['daily_signal_coverage'] = pd.read_parquet(path / 'daily_signal_coverage.parquet').set_index('date')
        frames['revision_age_counts'] = pd.read_parquet(path / 'revision_age_counts.parquet')
        frames['streaming_audit'] = json.loads((path / 'streaming_audit.json').read_text())
    else:
        required = small + ['signals', 'analyst_scores', 'contributions', 'outcomes']
        frames = {name: pd.read_parquet(path / f'{name}.parquet') for name in required}
    returns = return_panel(frames['daily_results'])
    if returns.empty:
        raise ValueError('No daily performance sample')
    return manifest, frames


def return_panel(daily):
    if daily.duplicated(['strategy', 'date']).any():
        raise ValueError('Duplicate strategy/date results')
    returns = daily.pivot(index='date', columns='strategy', values='net_return').sort_index()
    if not {'analyst_sharpe', 'unweighted'}.issubset(returns.columns):
        raise ValueError('Both strategy series are required for the common-sample comparison')
    benchmark = daily.groupby('date').benchmark_return
    if benchmark.nunique(dropna=False).gt(1).any():
        raise ValueError('Benchmark differs between strategies')
    returns['benchmark'] = benchmark.first().reindex(returns.index)
    if not np.isfinite(returns.to_numpy()).all() or returns.lt(-1).any().any():
        raise ValueError('Missing or invalid daily returns; do not drop or bridge gaps')
    returns.columns.name = None
    return returns


def drawdown_panel(returns):
    wealth = (1 + returns).cumprod()
    return wealth.div(wealth.cummax().clip(lower=1.)) - 1


def max_underwater_sessions(returns):
    longest = current = 0
    for underwater in drawdown_panel(returns).lt(-1e-12):
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return longest


def performance_table(daily):
    panel = return_panel(daily)
    rows = []
    for name, r in panel.items():
        wealth = (1 + r).cumprod()
        volatility = r.std(ddof=1)
        downside = np.sqrt(np.mean(np.minimum(r, 0.) ** 2))
        active = r - panel.benchmark
        tracking_error = active.std(ddof=1)
        cagr = wealth.iloc[-1] ** (252 / len(r)) - 1
        drawdown = drawdown_panel(r).min()
        q05 = r.quantile(.05)
        rows.append(dict(series=name, sessions=len(r), total_return=wealth.iloc[-1]-1,
                         cagr_252=cagr, annualized_mean=r.mean()*252,
                         volatility_252=volatility*np.sqrt(252),
                         sharpe_zero_rf=r.mean()/volatility*np.sqrt(252) if volatility > 0 else np.nan,
                         sortino_zero_target=r.mean()/downside*np.sqrt(252) if downside > 0 else np.nan,
                         max_drawdown=drawdown, calmar=cagr/abs(drawdown) if drawdown < 0 else np.nan,
                         longest_underwater_sessions=max_underwater_sessions(r),
                         positive_day_fraction=r.gt(0).mean(), best_day=r.max(), worst_day=r.min(),
                         daily_return_q05=q05, expected_shortfall_return_05=r[r.le(q05)].mean(),
                         tracking_error_252=tracking_error*np.sqrt(252),
                         information_ratio=active.mean()/tracking_error*np.sqrt(252) if tracking_error > 0 else np.nan,
                         benchmark_beta=r.cov(panel.benchmark)/panel.benchmark.var(ddof=1) if panel.benchmark.var(ddof=1) > 0 else np.nan,
                         benchmark_correlation=r.corr(panel.benchmark) if volatility > 1e-12 and panel.benchmark.std(ddof=1) > 1e-12 else np.nan))
    return pd.DataFrame(rows).set_index('series')


def calendar_returns(panel, frequency):
    """Include partial first/last periods; tables explicitly report session counts."""
    periods = panel.index.to_period(frequency)
    result = panel.groupby(periods).agg(lambda r: (1+r).prod()-1)
    result['sessions'] = panel.groupby(periods).size()
    result.index.name = 'period'
    return result


def rolling_statistics(panel, window=126):
    mean = panel.rolling(window, min_periods=window).mean()
    std = panel.rolling(window, min_periods=window).std(ddof=1)
    return mean.div(std.where(std.gt(0))) * np.sqrt(252), std * np.sqrt(252)


def closed_trade_returns(frames):
    """Trade returns include both fees; open trades never enter closed-trade stats."""
    closed = frames['completed_positions'].copy()
    columns = ['strategy', 'permno', 'entry_date', 'exit_date', 'entry_value', 'exit_value',
               'entry_fee', 'exit_fee', 'net_pnl', 'net_trade_return', 'held_sessions', 'exit_reason']
    if closed.empty:
        return pd.DataFrame(columns=columns)
    trades = frames['trades']
    entries = trades[trades.action.eq('entry')][['strategy','permno','date','value','cost_dollars']].rename(
        columns={'date':'entry_date','value':'entry_value','cost_dollars':'entry_fee'})
    exits = trades[trades.action.eq('exit')][['strategy','permno','date','value','cost_dollars']].rename(
        columns={'date':'exit_date','value':'exit_value','cost_dollars':'exit_fee'})
    closed = closed.merge(entries, on=['strategy','permno','entry_date'], how='left', validate='one_to_one')
    closed = closed.merge(exits, on=['strategy','permno','exit_date'], how='left', validate='one_to_one')
    if closed[['entry_value','entry_fee','exit_value','exit_fee']].isna().any().any():
        raise ValueError('Completed positions do not reconcile to entry/exit trades')
    direction = closed['side'].map({'long':1., 'short':-1.}) if 'side' in closed else 1.
    borrow = closed['borrow_cost_dollars'] if 'borrow_cost_dollars' in closed else 0.
    closed['net_pnl'] = direction*(closed.exit_value-closed.entry_value) - closed.exit_fee - closed.entry_fee - borrow
    if 'side' in closed:
        columns += ['side', 'borrow_cost_dollars']
    closed['net_trade_return'] = closed.net_pnl / (closed.entry_value + closed.entry_fee)
    return closed[columns]


def trading_table(frames):
    closed = closed_trade_returns(frames)
    rows = []
    for name, days in frames['daily_results'].groupby('strategy'):
        done = closed[closed.strategy.eq(name)]
        pnl = done.net_pnl
        losses = -pnl[pnl.lt(0)].sum()
        trades = frames['trades'][frames['trades'].strategy.eq(name)]
        held = frames['holdings'][frames['holdings'].strategy.eq(name)]
        open_count = int(frames['open_positions'].strategy.eq(name).sum())
        rows.append(dict(strategy=name, entries=int(trades.action.eq('entry').sum()),
                         completed_positions=len(done), open_positions=open_count,
                         win_rate=pnl.gt(0).mean() if len(done) else np.nan,
                         average_trade_return=done.net_trade_return.mean(), median_trade_return=done.net_trade_return.median(),
                         profit_factor=pnl[pnl.gt(0)].sum()/losses if losses > 0 else np.nan,
                         mean_held_sessions=done.held_sessions.mean(),
                         early_terminal_exits=int(done.exit_reason.isin(['terminal','terminal_close_proxy']).sum()),
                         approximate_off_exchange_exits=int(done.exit_reason.eq('off_exchange_close_proxy').sum()),
                         approximate_terminal_exits=int(done.exit_reason.eq('terminal_close_proxy').sum()),
                         total_cost_dollars=days.cost_dollars.sum(), total_traded_nav=days.turnover.sum(),
                         mean_positions=days.n_holdings.mean(), max_positions=days.n_holdings.max(),
                         mean_cash_weight=(days.cash/days.nav).mean(),
                         mean_invested_weight=(1-days.cash/days.nav).mean(),
                         max_single_position_weight=held.weight.abs().max() if len(held) else 0.,
                         long_entries=int((trades.action.eq('entry') & (trades.side.eq('long') if 'side' in trades else True)).sum()),
                         short_entries=int((trades.action.eq('entry') & (trades.side.eq('short') if 'side' in trades else False)).sum()),
                         mean_gross_exposure=days.gross_exposure.mean() if 'gross_exposure' in days else (1-days.cash/days.nav).mean(),
                         mean_net_exposure=days.net_exposure.mean() if 'net_exposure' in days else (1-days.cash/days.nav).mean(),
                         borrow_cost_dollars=days.borrow_cost_dollars.sum() if 'borrow_cost_dollars' in days else 0.,
                         gross_total_return=days.gross_nav.iloc[-1]/(days.gross_nav.iloc[0]/(1+days.gross_return.iloc[0]))-1))
    return pd.DataFrame(rows).set_index('strategy')


def audit_table(frames, config):
    daily, orders = frames['daily_results'], frames['orders']
    closed, scores, active = frames['completed_positions'], frames['analyst_scores'], frames['contributions']
    freshness = config.get('revision_freshness_sessions')
    age_check = ('Revision age filter disabled' if freshness is None
                 else 'Active revision ages are strictly below the freshness limit')
    checks = {
        'Complete aligned daily strategy and benchmark returns': bool(np.isfinite(return_panel(daily).to_numpy()).all()),
        'No duplicate strategy/date rows': not daily.duplicated(['strategy','date']).any(),
        'Cash never negative': bool(daily.cash.ge(-1e-7).all()),
        'Position count respects limit': bool(daily.n_holdings.le(config['top_n']).all()),
        'Orders execute after their signal date': bool((orders.execution_date > orders.cutoff).all()),
        'Scheduled exits have exactly the configured holding period': bool(closed.loc[closed.exit_reason.eq('holding_period'), 'held_sessions'].eq(config['holding_sessions']).all()),
        'Halt-delayed exits exceed the scheduled holding period': bool(closed.loc[closed.exit_reason.eq('holding_period_after_halt'), 'held_sessions'].gt(config['holding_sessions']).all()),
        'Historical scores meet minimum valid-call count': bool(scores.n_calls.ge(config['min_calls']).all()),
        age_check: True if freshness is None else bool((active.age_sessions.ge(0) & active.age_sessions.lt(freshness)).all()),
        'Contributing calls are available by cutoff': bool((active.available_date <= active.cutoff).all()),
    }
    if 'streaming_audit' in frames:
        streamed = frames['streaming_audit']
        checks['Historical scores meet minimum valid-call count'] = streamed['scores_meet_minimum']
        checks[age_check] = True if freshness is None else streamed['contributions_fresh']
        checks['Contributing calls are available by cutoff'] = streamed['contributions_available']
    for name, group in daily.groupby('strategy'):
        computed = config['initial_nav'] * (1+group.sort_values('date').net_return).cumprod()
        checks[f'{LABELS[name]} NAV reconciles to daily returns'] = bool(np.allclose(computed, group.sort_values('date').nav, rtol=1e-10, atol=1e-5))
    return pd.DataFrame({'check':list(checks), 'passed':list(checks.values())})


def order_table(orders):
    if orders.empty:
        return pd.DataFrame(columns=['strategy','status','orders'])
    return orders.groupby(['strategy','status']).size().rename('orders').reset_index()


def replay_sensitivity(frames, daily_market, benchmark, config, start, end,
                       thresholds=(1.28, 1.68, 2.0), costs=(0., 5., 10., 20.)):
    """One-factor-at-a-time, in-sample scenarios. Never choose a winning config.

    Reuse point-in-time signals; change only crossing threshold OR execution cost.
    Baseline holdings are not reused: each scenario reruns trade/cash accounting.
    """
    from .revision_zscore import simulate_entries
    threshold_field = 'entry_signal_threshold' if config.strategy_mode=='revision_signal' else 'entry_zscore'
    baseline_threshold = getattr(config, threshold_field)
    scenarios = [('baseline', baseline_threshold, config.cost_bps)]
    scenarios += [('threshold', x, config.cost_bps) for x in thresholds if x != baseline_threshold]
    scenarios += [('cost', baseline_threshold, x) for x in costs if x != config.cost_bps]
    rows = []
    for kind, threshold, cost in scenarios:
        settings = replace(config, **{threshold_field:float(threshold)}, cost_bps=float(cost))
        signals = frames['signals'].sort_values(['permno','date']).copy()
        for prefix in ['weighted','unweighted']:
            z = signals[f'{prefix}_z']
            previous = signals.groupby('permno')[f'{prefix}_z'].shift(1)
            signals[f'{prefix}_crossing'] = z.gt(threshold) & previous.le(threshold)
            signals[f'{prefix}_short_crossing'] = z.lt(-threshold) & previous.ge(-threshold)
        if config.strategy_mode == 'revision_signal':
            from .revision_zscore import raw_signal_triggers
            signals = raw_signal_triggers(signals, settings)
        result = simulate_entries(signals, daily_market, benchmark, start, end, settings)
        stats = performance_table(result['daily_results']).loc['analyst_sharpe']
        rows.append(dict(scenario=kind, threshold=threshold, cost_bps=cost,
                         total_return=stats.total_return, cagr_252=stats.cagr_252,
                         sharpe_zero_rf=stats.sharpe_zero_rf, max_drawdown=stats.max_drawdown,
                         entries=int((result['trades'].strategy.eq('analyst_sharpe') & result['trades'].action.eq('entry')).sum())))
    return pd.DataFrame(rows)
