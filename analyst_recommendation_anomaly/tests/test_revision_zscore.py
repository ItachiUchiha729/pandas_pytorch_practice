from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analyst_sharpe import AnalystSharpeConfig, normalize_events, prepare_market, build_outcomes, run_strategy
from src.revision_zscore import revision_signals, trailing_zscores, simulate_entries
from src.run_analyst_sharpe import demo_inputs


def fixture(n=70, count=1):
    dates = pd.bdate_range('2020-01-01', periods=n)
    # An exchange holiday must not count as a session.
    dates = dates.difference(pd.DatetimeIndex(['2020-01-20']))
    daily = pd.concat([pd.DataFrame(dict(date=dates, permno=k, ret_total=.01,
                                         prc=10., shrcd=10, exchcd=1, is_terminal=False))
                       for k in range(1, count + 1)], ignore_index=True)
    bench = pd.DataFrame(dict(date=dates, ret=0.))
    return daily, bench, dates


def events_for(dates):
    return normalize_events(pd.DataFrame([
        dict(permno=1, analyst_id=a, recommendation_date=dates[i], available_date=dates[i+1], rating=r)
        for a in ['a', 'b'] for i, r in [(0, 1), (4, 2)]
    ]))


def scores():
    return pd.DataFrame(dict(analyst_id=['a', 'b'], score=[.6, .2]))


def signal_panel(dates, count=1, crossings=None):
    crossings = crossings or {dates[1]: [1]}
    return pd.DataFrame([dict(date=date, permno=k, eligible=True, signal=1., unweighted_score=1.,
                              weighted_z=2.0 + k/100, unweighted_z=2.0 + k/100,
                              weighted_crossing=k in crossings.get(date, []),
                              unweighted_crossing=k in crossings.get(date, []))
                         for date in dates for k in range(1, count+1)])


def test_five_level_revision_freshness_13_in_14_out_and_previous_old_call():
    daily, _, dates = fixture()
    events = events_for(dates)
    config = AnalystSharpeConfig()
    config = replace(config, revision_freshness_sessions=14)
    for age, expected in [(13, True), (14, False)]:
        cutoff = dates[4+age]
        signal, active = revision_signals(events, scores(), daily[daily.date.eq(cutoff)], dates, cutoff, config)
        assert bool(signal.eligible.iloc[0]) is expected
        if expected:
            assert active.revision.eq(1).all()  # Buy -> Strong Buy survives normalization.
            assert signal.signal.iloc[0] == 1
        else:
            assert active.empty and signal.signal.iloc[0] == 0
    # An older previous call is retained to form the revision.
    assert active.empty


def test_unavailable_predecessor_withdrawal_and_reaffirmation():
    daily, _, dates = fixture()
    config = AnalystSharpeConfig()
    events = events_for(dates)
    cutoff = dates[6]
    late = events.copy()
    late.loc[late.recommendation_date.eq(dates[0]), 'available_date'] = dates[9]
    signal, active = revision_signals(late, scores(), daily[daily.date.eq(cutoff)], dates, cutoff, config)
    assert active.empty  # No future predecessor can create a revision today.
    withdrawn = pd.concat([events, pd.DataFrame([
        dict(permno=1, analyst_id=a, recommendation_date=dates[2], available_date=dates[3],
             rating=np.nan, event_type='withdraw') for a in ['a', 'b']])], ignore_index=True)
    _, active = revision_signals(normalize_events(withdrawn), scores(), daily[daily.date.eq(cutoff)], dates, cutoff, config)
    assert active.empty
    repeat = pd.concat([events, pd.DataFrame([
        dict(permno=1, analyst_id=a, recommendation_date=dates[5], available_date=dates[6],
             rating=2, event_type='rating') for a in ['a', 'b']])], ignore_index=True)
    signal, active = revision_signals(normalize_events(repeat), scores(), daily[daily.date.eq(cutoff)], dates, cutoff, config)
    assert active.revision.eq(0).all()
    assert signal.signal.iloc[0] == 0


def test_skill_remains_direction_based_for_strong_and_normal_buy():
    daily, bench, dates = fixture()
    rows = pd.DataFrame([dict(permno=1, analyst_id=str(r), recommendation_date=dates[0],
                              available_date=dates[1], rating=r) for r in [1, 2]])
    daily, bench = prepare_market(daily, bench)
    outcomes = build_outcomes(normalize_events(rows), daily, bench, AnalystSharpeConfig())
    assert outcomes.analyst_return.iloc[0] == pytest.approx(outcomes.analyst_return.iloc[1])


def test_zscore_uses_previous_twenty_sessions_strict_crossing_and_flat_history():
    _, _, dates = fixture(40)
    config = AnalystSharpeConfig()
    values = np.tile([-1., 1.], 20)[:len(dates)]
    values[21:23] = 3.
    panel = pd.DataFrame(dict(date=dates, permno=1, eligible=True, signal=values, unweighted_score=values))
    result = trailing_zscores(panel, dates, config)
    expected = (values[21] - values[1:21].mean()) / values[1:21].std(ddof=1)
    assert result.weighted_z.iloc[21] == pytest.approx(expected)
    assert result.weighted_crossing.iloc[21]
    assert not result.weighted_crossing.iloc[22]  # Still above, not a new crossing.
    threshold = replace(config, entry_zscore=expected)
    assert not trailing_zscores(panel, dates, threshold).weighted_crossing.iloc[21]
    changed = panel.copy()
    changed.loc[21, 'signal'] = 100
    modified = trailing_zscores(changed, dates, config)
    assert modified.weighted_mean.iloc[21] == result.weighted_mean.iloc[21]
    flat = trailing_zscores(panel.assign(signal=1., unweighted_score=1.), dates, config)
    assert flat.weighted_z.isna().all() and not flat.weighted_crossing.any()
    missing = trailing_zscores(panel.drop(index=10), dates, config)
    assert pd.isna(missing.weighted_z.iloc[21])


def test_exact_fourteen_returns_no_resizing_no_same_day_or_held_reentry():
    daily, bench, dates = fixture()
    config = AnalystSharpeConfig(cost_bps=0)
    signals = signal_panel(dates, crossings={dates[1]: [1], dates[4]: [1], dates[15]: [1]})
    result = simulate_entries(signals, daily, bench, dates[0], dates[25], config)
    trades = result['trades'].query("strategy == 'analyst_sharpe'")
    assert trades.action.tolist() == ['entry', 'exit']
    assert trades.date.tolist() == [dates[2], dates[16]]
    assert trades.value.iloc[1] == pytest.approx(trades.value.iloc[0] * 1.01**14)
    assert result['completed_positions'].held_sessions.eq(14).all()
    held = result['holdings'].query("strategy == 'analyst_sharpe'")
    assert held.value.iloc[1] == pytest.approx(held.value.iloc[0]*1.01)
    assert result['orders'].query("status == 'already_held'").shape[0] == 4


def test_capacity_cash_costs_and_unclosed_final_positions():
    daily, bench, dates = fixture(count=3)
    config = AnalystSharpeConfig(top_n=2, cost_bps=10)
    signals = signal_panel(dates, count=3, crossings={dates[1]: [1,2,3]})
    result = simulate_entries(signals, daily, bench, dates[0], dates[5], config)
    trades = result['trades'].query("strategy == 'analyst_sharpe'")
    assert set(trades.permno) == {2,3}  # Highest z-scores first.
    assert result['completed_positions'].empty
    assert len(result['open_positions']) == 4
    initial = result['daily_results'].query("strategy == 'analyst_sharpe'").iloc[1]
    assert initial.cash == pytest.approx(0, abs=1e-7)
    assert initial.nav + initial.cost_dollars == pytest.approx(config.initial_nav)
    assert initial.nav == pytest.approx(config.initial_nav / 1.001)
    assert result['daily_results'].cash.ge(0).all()


def test_missing_held_returns_and_terminal_events():
    daily, bench, dates = fixture()
    signals = signal_panel(dates)
    bad = daily.copy()
    bad.loc[bad.date.eq(dates[3]), 'ret_total'] = np.nan
    with pytest.raises(ValueError, match='Missing held return'):
        simulate_entries(signals, bad, bench, dates[0], dates[20], AnalystSharpeConfig())
    terminal = daily[daily.date.le(dates[3])].copy()
    terminal.loc[terminal.date.eq(dates[3]), ['is_terminal', 'ret_total']] = [True, -.5]
    result = simulate_entries(signals, terminal, bench, dates[0], dates[20], AnalystSharpeConfig(cost_bps=0))
    assert result['completed_positions'].exit_reason.eq('terminal').all()
    assert result['completed_positions'].held_sessions.eq(1).all()


def test_future_data_invariance_for_revision_signals_and_trades():
    events, daily, bench, start, _ = demo_inputs()
    end = bench.date.iloc[205]
    base = run_strategy(events, daily, bench, start, end)
    changed = events.copy()
    changed.loc[changed.available_date.gt(end), 'rating'] *= -1
    changed_prices = daily.copy()
    changed_prices.loc[changed_prices.date.gt(end), 'ret_total'] = -.8
    altered = run_strategy(changed, changed_prices, bench, start, end)
    for key in ['signals', 'trades', 'daily_results']:
        pd.testing.assert_frame_equal(base[key], altered[key])
    truncated = run_strategy(events[events.available_date.le(end)], daily[daily.date.le(end)],
                             bench[bench.date.le(end)], start, end)
    for key in ['signals', 'trades', 'daily_results']:
        pd.testing.assert_frame_equal(base[key], truncated[key])


def test_threshold_and_window_configuration():
    assert AnalystSharpeConfig().entry_zscore == 1.68
    for kwargs in [dict(entry_zscore=float('nan')), dict(zscore_window=1), dict(holding_sessions=0)]:
        with pytest.raises(ValueError):
            AnalystSharpeConfig(**kwargs)


def test_no_freshness_limit_keeps_old_revisions_and_two_analyst_minimum():
    daily, _, dates = fixture()
    events = events_for(dates)
    config = AnalystSharpeConfig()
    assert config.revision_freshness_sessions is None
    for cutoff in [dates[18], dates[-1]]:
        signal, active = revision_signals(events, scores(), daily[daily.date.eq(cutoff)], dates, cutoff, config)
        assert signal.eligible.iloc[0]
        assert len(active) == 2 and active.age_sessions.ge(14).all()
        assert active.revision.eq(1).all()
        signal, _ = revision_signals(events, scores().iloc[:1], daily[daily.date.eq(cutoff)], dates, cutoff, config)
        assert not signal.eligible.iloc[0]
    # Calls before the supplied calendar remain usable when no age limit is set.
    shortened = dates[20:]
    signal, active = revision_signals(events, scores(), daily[daily.date.eq(shortened[-1])], shortened, shortened[-1], config)
    assert signal.eligible.iloc[0] and len(active) == 2


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_optional_freshness(limit):
    with pytest.raises(ValueError, match='revision_freshness_sessions'):
        AnalystSharpeConfig(revision_freshness_sessions=limit)


@pytest.mark.parametrize('failure', ['terminal', 'missing_row', 'missing_price', 'nullable_price', 'zero_price'])
def test_untradable_entry_is_rejected_without_using_capacity(failure):
    from src.streaming_strategy import PortfolioBook
    daily, bench, dates = fixture(n=25, count=2)
    signals = signal_panel(dates, count=2, crossings={dates[1]: [1, 2]})
    # Stock 2 ranks first but cannot be purchased on the next session.
    mask = daily.date.eq(dates[2]) & daily.permno.eq(2)
    if failure == 'terminal':
        daily.loc[mask, 'is_terminal'] = True
    elif failure == 'missing_row':
        daily = daily[~mask].copy()
    elif failure == 'nullable_price':
        daily['prc'] = daily.prc.astype('Float64')
        daily.loc[mask, 'prc'] = pd.NA
    else:
        daily.loc[mask, 'prc'] = np.nan if failure == 'missing_price' else 0.
    config = AnalystSharpeConfig(top_n=1)
    reference = simulate_entries(signals, daily, bench, dates[0], dates[-1], config)
    rejected = reference['orders'].query('permno == 2')
    assert rejected.status.eq('untradable_at_execution').all() and len(rejected) == 2
    assert reference['trades'].query("action == 'entry'").permno.eq(1).all()
    assert reference['completed_positions'].held_sessions.eq(14).all()
    for strategy, prefix in [('analyst_sharpe', 'weighted'), ('unweighted', 'unweighted')]:
        book = PortfolioBook(strategy, prefix, dates, config)
        for index in range(1, len(dates)):
            book.step(index, daily[daily.date.eq(dates[index])],
                      signals[signals.date.eq(dates[index-1])], 0.)
        for name in ['orders', 'daily_results', 'trades', 'completed_positions']:
            expected = reference[name].query('strategy == @strategy').reset_index(drop=True)
            pd.testing.assert_frame_equal(book.frames()[name], expected, check_dtype=False)


def test_nullable_held_return_reports_security_and_date():
    from src.streaming_strategy import PortfolioBook
    daily, bench, dates = fixture(n=25)
    daily['ret_total'] = daily.ret_total.astype('Float64')
    daily.loc[daily.date.eq(dates[3]), 'ret_total'] = pd.NA
    signals = signal_panel(dates)
    config = AnalystSharpeConfig()
    message = f'Missing held return: analyst_sharpe, 1, {dates[3].date()}'
    with pytest.raises(ValueError, match=message):
        simulate_entries(signals, daily, bench, dates[0], dates[-1], config)
    book = PortfolioBook('analyst_sharpe', 'weighted', dates, config)
    with pytest.raises(ValueError, match=message):
        for index in range(1, len(dates)):
            book.step(index, daily[daily.date.eq(dates[index])],
                      signals[signals.date.eq(dates[index-1])], 0.)


def test_missing_terminal_close_proxy_is_explicit_costed_and_flagged():
    from src.streaming_strategy import PortfolioBook
    daily, bench, dates = fixture(n=25)
    daily['ret'] = daily.ret_total
    mask = daily.date.eq(dates[3])
    daily.loc[mask, 'is_terminal'] = True
    daily.loc[mask, 'ret_total'] = np.nan
    signals = signal_panel(dates)
    strict = AnalystSharpeConfig()
    with pytest.raises(ValueError, match='terminal payoff unavailable'):
        simulate_entries(signals, daily, bench, dates[0], dates[-1], strict)
    config = replace(strict, missing_terminal_policy='last_close')
    result = simulate_entries(signals, daily, bench, dates[0], dates[-1], config)
    exits = result['trades'].query("action == 'exit'")
    assert len(exits) == 2 and exits.reason.eq('terminal_close_proxy').all()
    assert np.allclose(exits.cost_dollars, exits.value * config.cost_bps / 10000)
    assert result['completed_positions'].exit_reason.eq('terminal_close_proxy').all()
    entry = result['trades'].query("action == 'entry'").iloc[0]
    assert exits.iloc[0].value == pytest.approx(entry.value * 1.01)
    book = PortfolioBook('analyst_sharpe', 'weighted', dates, config)
    for index in range(1, len(dates)):
        book.step(index, daily[daily.date.eq(dates[index])], signals[signals.date.eq(dates[index-1])], 0.)
    for name in ['daily_results', 'trades', 'completed_positions']:
        pd.testing.assert_frame_equal(book.frames()[name], result[name].query("strategy == 'analyst_sharpe'").reset_index(drop=True), check_dtype=False)
    for missing in ['ret', 'prc']:
        bad = daily.copy()
        bad.loc[mask, missing] = np.nan
        with pytest.raises(ValueError, match='Missing held return'):
            simulate_entries(signals, bad, bench, dates[0], dates[-1], config)
    bad = daily.copy()
    bad.loc[mask, 'is_terminal'] = False
    with pytest.raises(ValueError, match='Missing held return'):
        simulate_entries(signals, bad, bench, dates[0], dates[-1], config)
