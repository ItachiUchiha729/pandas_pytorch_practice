from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.analyst_sharpe import (
    AnalystSharpeConfig, analyst_scores, build_outcomes, build_targets,
    load_ibes_extract, make_schedule, normalize_events, prepare_market,
    rebalance_account, run_strategy, simulate, stock_signals,
)
from src.run_analyst_sharpe import demo_inputs, main


def market(n=100):
    dates = pd.bdate_range('2020-01-01', periods=n)
    stocks = pd.DataFrame({'permno': 1, 'date': dates, 'ret_total': .01,
                           'prc': 10., 'shrcd': 10, 'exchcd': 1, 'is_terminal': False})
    bench = pd.DataFrame({'date': dates, 'ret': .001})
    return stocks, bench


def event(date='2020-01-01', available='2020-01-02', analyst='a', rating=1, **kwargs):
    return dict(permno=1, analyst_id=analyst, recommendation_date=date,
                available_date=available, rating=rating, **kwargs)


def test_forward_horizon_excludes_start_day_and_respects_maturity():
    config = AnalystSharpeConfig(strategy_mode="scheduled")
    daily, bench = prepare_market(*market())
    events = normalize_events(pd.DataFrame([event(), event(analyst='b', rating=-1)]))
    outcomes = build_outcomes(events, daily, bench, config)
    expected = 1.01 ** 30 - 1.001 ** 30
    assert outcomes.analyst_return.tolist() == pytest.approx([expected, -expected])
    assert outcomes.start_date.iloc[0] == bench.date.iloc[1]
    assert outcomes.end_date.iloc[0] == bench.date.iloc[31]
    assert outcomes.outcome_available_date.iloc[0] == bench.date.iloc[31] + pd.Timedelta(1, unit="D").as_unit("ns")
    mature = pd.concat([outcomes.iloc[[0]].assign(analyst_return=x, event_id=str(x),
                         recommendation_date=pd.Timestamp('2019-01-01') + pd.Timedelta(i, unit="D").as_unit("ns"))
                       for i, x in enumerate(np.linspace(.01, .05, 25))], ignore_index=True)
    cutoff = outcomes.outcome_available_date.iloc[0]
    assert analyst_scores(mature, cutoff - pd.Timedelta(1, unit="D").as_unit("ns"), config).empty
    score = analyst_scores(mature, cutoff, config).iloc[0]
    assert score.n_calls == 25
    assert score.score == pytest.approx(np.mean(np.linspace(.01, .05, 25)) / np.std(np.linspace(.01, .05, 25), ddof=1))
    assert analyst_scores(mature.iloc[:24], cutoff, config).empty
    assert analyst_scores(mature.assign(analyst_return=.02), cutoff, config).empty


def test_neutral_missing_and_terminal_outcomes():
    config = AnalystSharpeConfig(strategy_mode="scheduled")
    daily, bench = market()
    events = normalize_events(pd.DataFrame([event(), event(analyst='neutral', rating=0)]))
    daily.loc[10, 'ret_total'] = np.nan
    stocks, bench = prepare_market(daily, bench)
    out = build_outcomes(events, stocks, bench, config)
    assert out.outcome_status.tolist() == ['missing_stock_outcome', 'not_directional']
    daily = daily.iloc[:11].copy()
    daily.loc[10, ['ret_total', 'is_terminal']] = [-.5, True]
    stocks, bench = prepare_market(daily, bench)
    out = build_outcomes(events, stocks, bench, config)
    assert out.stock_return.iloc[0] == pytest.approx(1.01 ** 8 * .5 - 1)


def test_state_withdrawal_expiry_conflict_and_distinct_analyst_count():
    daily, _ = prepare_market(*market())
    cutoff = pd.Timestamp('2020-01-15')
    quotes = daily[daily.date.eq(cutoff)]
    scores = pd.DataFrame({'analyst_id': ['a', 'b'], 'score': [.6, .2]})
    config = AnalystSharpeConfig(strategy_mode="scheduled")
    events = normalize_events(pd.DataFrame([event(), event(), event(analyst='b', rating=-1)]))
    assert len(events) == 2
    signal, _ = stock_signals(events, scores, quotes, cutoff, config)
    assert signal.eligible.iloc[0]
    assert signal.signal.iloc[0] == pytest.approx(.5)
    one, _ = stock_signals(events, scores.iloc[:1], quotes, cutoff, config)
    assert not one.eligible.iloc[0]
    withdraw = event('2020-01-10', '2020-01-11', 'b', np.nan, event_type='withdraw')
    newer = normalize_events(pd.DataFrame([event(), event(analyst='b'), withdraw]))
    signal, _ = stock_signals(newer, scores, quotes, cutoff, config)
    assert not signal.eligible.iloc[0]
    conflict = normalize_events(pd.DataFrame([event(), event(rating=-1), event(analyst='b')]))
    assert conflict.loc[conflict.analyst_id.eq('a'), 'event_type'].iloc[0] == 'invalid'
    signal, _ = stock_signals(conflict, scores, quotes, cutoff, config)
    assert not signal.eligible.iloc[0]
    signal, _ = stock_signals(events, scores, quotes, cutoff, replace(config, max_age_days=5))
    assert not signal.eligible.iloc[0]


def test_negative_policy_neutral_dilution_and_zero_denominator():
    config = AnalystSharpeConfig(strategy_mode="scheduled")
    daily, _ = prepare_market(*market())
    cutoff = daily.date.iloc[5]
    quotes = daily[daily.date.eq(cutoff)]
    events = normalize_events(pd.DataFrame([event(), event(analyst='b', rating=0)]))
    scores = pd.DataFrame({'analyst_id': ['a', 'b'], 'score': [.6, .2]})
    signal, _ = stock_signals(events, scores, quotes, cutoff, config)
    assert signal.signal.iloc[0] == pytest.approx(.75)
    scores['score'] = [-.6, -.2]
    signal, _ = stock_signals(events, scores, quotes, cutoff, config)
    assert not signal.eligible.iloc[0] and pd.isna(signal.signal.iloc[0])
    signal, _ = stock_signals(events, scores, quotes, cutoff, replace(config, negative_scores='signed'))
    assert signal.eligible.iloc[0] and signal.signal.iloc[0] == pytest.approx(-.75)


def test_costs_are_self_financing_and_use_drifted_positions():
    pos, cash, cost, turnover, _ = rebalance_account({}, 100, {1: .5, 2: .5}, .01)
    assert sum(pos.values()) == pytest.approx(100 / 1.01)
    assert cash == pytest.approx(0, abs=1e-12)
    assert cost == pytest.approx(100 - 100 / 1.01)
    # No change to target weights still requires trading after price drift.
    pos[1] *= 1.2
    before = sum(pos.values())
    new, cash, cost, turnover, _ = rebalance_account(pos, cash, {1: .5, 2: .5}, .01)
    assert turnover > 0
    assert sum(new.values()) + cash + cost == pytest.approx(before)
    assert new[1] == pytest.approx(new[2])
    new, cash, cost, _, _ = rebalance_account(new, cash, {}, .01)
    assert new == {} and cash > 0 and cost > 0


def test_schedule_next_session_and_two_calendar_weeks():
    calendar = pd.bdate_range('2020-01-01', '2020-02-28').difference(pd.DatetimeIndex(['2020-01-20']))
    schedule = make_schedule(calendar, '2020-01-04', '2020-02-28')
    assert schedule.cutoff.iloc[0] == pd.Timestamp('2020-01-06')
    assert schedule.execution_date.iloc[0] == pd.Timestamp('2020-01-07')
    assert schedule.cutoff.iloc[1] == pd.Timestamp('2020-01-21')
    assert (schedule.execution_date > schedule.cutoff).all()


def test_execution_lag_missing_returns_terminal_and_cash():
    config = AnalystSharpeConfig(cost_bps=0)
    daily, bench = prepare_market(*market(6))
    schedule = pd.DataFrame({'cutoff': [bench.date.iloc[0]], 'execution_date': [bench.date.iloc[1]]})
    targets = pd.DataFrame([{'strategy': 'analyst_sharpe', 'execution_date': bench.date.iloc[1], 'permno': 1, 'weight': .05}])
    results, _, _ = simulate(targets, schedule, daily, bench, bench.date.iloc[-1], config)
    result = results[results.strategy.eq('analyst_sharpe')]
    assert result.net_return.iloc[0] == pytest.approx(0)
    assert result.net_return.iloc[1] == pytest.approx(.0005)
    assert result.cash.iloc[0] == pytest.approx(config.initial_nav * .95)
    bad = daily.copy()
    bad.loc[2, 'ret_total'] = np.nan
    with pytest.raises(ValueError, match='Missing held return'):
        simulate(targets, schedule, bad, bench, bench.date.iloc[-1], config)
    terminal = daily.iloc[:3].copy()
    terminal.loc[2, ['is_terminal', 'ret_total']] = [True, -.5]
    results, _, _ = simulate(targets, schedule, terminal, bench, bench.date.iloc[-1], config)
    result = results[results.strategy.eq('analyst_sharpe')]
    assert result.nav.iloc[-1] == pytest.approx(config.initial_nav * .975)
    assert result.n_holdings.iloc[-1] == 0


def test_future_events_and_prices_cannot_change_prior_targets_or_returns():
    events, daily, bench, start, end = demo_inputs()
    config = AnalystSharpeConfig(strategy_mode="scheduled")
    boundary = bench.date.iloc[220]
    base = run_strategy(events, daily, bench, start, boundary, config)
    altered = events.copy()
    altered.loc[pd.to_datetime(altered.available_date).gt(boundary), 'rating'] *= -1
    changed_daily = daily.copy()
    changed_daily.loc[changed_daily.date.gt(boundary), 'ret_total'] = -.9
    changed = run_strategy(altered, changed_daily, bench, start, boundary, config)
    for key in ['targets', 'analyst_scores', 'daily_results']:
        pd.testing.assert_frame_equal(base[key], changed[key])
    # Actual data truncation also leaves pre-boundary signals unchanged.
    truncated = run_strategy(events[pd.to_datetime(events.available_date).le(boundary)],
                             daily[daily.date.le(boundary)], bench[bench.date.le(boundary)], start, boundary, config)
    for key in ['targets', 'analyst_scores', 'daily_results']:
        pd.testing.assert_frame_equal(base[key], truncated[key])


def test_cli_demo_writes_complete_reproducible_artifacts(tmp_path):
    main(['--demo', '--output-dir', str(tmp_path)])
    output = next(tmp_path.iterdir())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['synthetic']
    summary = pd.read_csv(output / 'summary.csv')
    assert set(summary.strategy) == {'analyst_sharpe', 'unweighted'}
    daily = pd.read_parquet(output / 'daily_results.parquet')
    assert daily.net_return.notna().all()
    assert (daily.cash >= -1e-8).all()
    assert 0 < daily.n_holdings.max() <= 20
    orders = pd.read_parquet(output / 'orders.parquet')
    assert (orders.execution_date > orders.cutoff).all()
    completed = pd.read_parquet(output / 'completed_positions.parquet')
    assert not completed.empty
    assert completed.held_sessions.eq(14).all()


def test_raw_extract_link_dates_mapping_and_availability(tmp_path):
    raw = pd.DataFrame({'amaskcd': [1., 2., 3., 4.], 'emaskcd': 10.,
        'anndats': pd.to_datetime(['2020-01-01'] * 4),
        'actdats': pd.to_datetime(['2020-01-02'] * 4),
        'revdats': pd.to_datetime(['2020-01-05'] * 4),
        'cusip': '12345678', 'ireccd': [1, 3, 5, 9], 'usfirm': 1})
    raw.to_parquet(tmp_path / 'calls.parquet')
    (tmp_path / 'manifest.json').write_text(json.dumps({'complete': True, 'files': [{'file': 'calls.parquet'}]}))
    names = pd.DataFrame({'permno': [1], 'ncusip': ['12345678'],
                          'namedt': ['2019-01-01'], 'nameendt': ['2021-01-01']})
    events, stats = load_ibes_extract(tmp_path, names, AnalystSharpeConfig(strategy_mode="scheduled"))
    assert events.rating.iloc[:3].tolist() == [2, 0, -2]
    assert events.event_type.iloc[-1] == 'invalid'
    assert events.available_date.eq(pd.Timestamp('2020-01-06')).all()
    assert stats['linked_rows'] == 4


def test_config_and_market_validation():
    with pytest.raises(ValueError):
        AnalystSharpeConfig(cost_bps=-1)
    with pytest.raises(ValueError):
        AnalystSharpeConfig(negative_scores='anything')
    daily, bench = market()
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_market(pd.concat([daily, daily.iloc[:1]]), bench)
    bench.loc[0, 'ret'] = np.nan
    with pytest.raises(ValueError, match='benchmark'):
        prepare_market(daily, bench)


def test_revised_versions_count_once_and_new_invalid_version_removes_score():
    config = AnalystSharpeConfig(min_calls=2)
    history = pd.DataFrame([
        dict(event('2020-01-01', '2020-01-02'), event_id='1', analyst_return=.1,
             outcome_status='complete', outcome_available_date=pd.Timestamp('2020-02-20')),
        dict(event('2020-01-01', '2020-01-03'), event_id='2', analyst_return=.2,
             outcome_status='complete', outcome_available_date=pd.Timestamp('2020-02-21')),
    ])
    for c in ['recommendation_date', 'available_date']:
        history[c] = pd.to_datetime(history[c])
    assert analyst_scores(history, pd.Timestamp('2020-03-01'), config).empty
    extra = history.iloc[[1]].copy()
    extra['recommendation_date'] = pd.Timestamp('2020-01-04')
    extra['available_date'] = pd.Timestamp('2020-01-05')
    extra['analyst_return'] = .3
    history = pd.concat([history, extra], ignore_index=True)
    assert analyst_scores(history, pd.Timestamp('2020-03-01'), config).n_calls.iloc[0] == 2
    invalid = extra.copy()
    invalid['available_date'] = pd.Timestamp('2020-03-02')
    invalid['outcome_status'] = 'not_directional'
    invalid['rating'] = np.nan
    history = pd.concat([history, invalid], ignore_index=True)
    assert analyst_scores(history, pd.Timestamp('2020-03-02'), config).empty


def test_unknown_delisting_return_is_not_ordinary_return():
    daily, bench = market()
    daily = daily.iloc[:3].drop(columns='ret_total')
    daily['ret'] = .01
    daily['dlret'] = np.nan
    daily.loc[2, 'is_terminal'] = True
    stocks, _ = prepare_market(daily, bench)
    assert pd.isna(stocks.ret_total.iloc[-1])


def test_real_file_cli_path_and_incomplete_run_manifest(tmp_path):
    events, daily, bench, start, _ = demo_inputs()
    end = bench.date.iloc[190]
    market_dir = tmp_path / 'market'
    market_dir.mkdir()
    daily.to_parquet(market_dir / 'daily.parquet', index=False)
    bench.to_parquet(market_dir / 'benchmark.parquet', index=False)
    (market_dir / 'manifest.json').write_text(json.dumps({'complete': True, 'files': [{'file': 'daily.parquet'}]}))
    events_path = tmp_path / 'events.parquet'
    events.to_parquet(events_path, index=False)
    output = tmp_path / 'results'
    argv = ['--events', str(events_path), '--market-dir', str(market_dir),
            '--start', str(start.date()), '--end', str(end.date()), '--output-dir', str(output)]
    main(argv)
    manifest = json.loads(next(output.glob('*/manifest.json')).read_text())
    assert manifest['complete'] and not manifest['synthetic']
    assert len(manifest['inputs']) == 4
    held = pd.read_parquet(next(output.glob('*/holdings.parquet')))
    first_entry = held.date.min()
    missing_date = bench.date.iloc[pd.Index(bench.date).get_loc(first_entry) + 1]
    daily.loc[daily.date.eq(missing_date), 'ret_total'] = np.nan
    daily.to_parquet(market_dir / 'daily.parquet', index=False)
    with pytest.raises(ValueError, match='Missing held return'):
        main(argv)
    manifests = [json.loads(p.read_text()) for p in output.glob('*/manifest.json')]
    assert sorted(m['complete'] for m in manifests) == [False, True]
    assert any('Missing held return' in m.get('error', '') for m in manifests)


def test_no_eligible_analysts_stays_in_cash():
    daily, bench = market()
    events = pd.DataFrame([event()])
    result = run_strategy(events, daily, bench, bench.date.iloc[0], bench.date.iloc[-1], AnalystSharpeConfig(strategy_mode='scheduled'))
    assert result['targets'].empty
    assert result['daily_results'].net_return.eq(0).all()
    assert result['daily_results'].n_holdings.eq(0).all()
