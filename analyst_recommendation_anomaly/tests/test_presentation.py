import json

import numpy as np
import pandas as pd
import pytest

from src.presentation import (performance_table, return_panel, drawdown_panel,
                              calendar_returns, closed_trade_returns, rolling_statistics,
                              max_underwater_sessions, load_run)


def results(values=(-.1, .1, .02)):
    dates = pd.bdate_range('2020-01-30', periods=len(values))
    return pd.DataFrame([dict(strategy=strategy, date=date, net_return=value,
                             benchmark_return=.001) for strategy in ['analyst_sharpe','unweighted']
                         for date,value in zip(dates, values)])


def test_stats_compound_returns_include_initial_loss_and_zero_rf_sharpe():
    daily = results()
    table = performance_table(daily)
    r = np.array([-.1, .1, .02])
    actual = table.loc['analyst_sharpe']
    assert actual.total_return == pytest.approx(np.prod(1+r)-1)
    assert actual.cagr_252 == pytest.approx(np.prod(1+r)**(252/3)-1)
    assert actual.max_drawdown == pytest.approx(-.1)
    assert actual.sharpe_zero_rf == pytest.approx(r.mean()/r.std(ddof=1)*np.sqrt(252))
    assert actual.sortino_zero_target == pytest.approx(r.mean()/np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(252))
    assert actual.expected_shortfall_return_05 == -.1
    assert actual.longest_underwater_sessions == 2


def test_missing_daily_returns_never_silently_removed():
    bad = results()
    bad.loc[0,'net_return'] = np.nan
    with pytest.raises(ValueError, match='Missing or invalid'):
        return_panel(bad)
    incomplete = results().iloc[1:]
    with pytest.raises(ValueError, match='Missing or invalid'):
        return_panel(incomplete)
    duplicate = pd.concat([results(),results().iloc[:1]])
    with pytest.raises(ValueError, match='Duplicate'):
        return_panel(duplicate)


def test_monthly_periods_compound_and_include_session_counts():
    monthly = calendar_returns(return_panel(results()), 'M')
    assert monthly.loc[pd.Period('2020-01'), 'analyst_sharpe'] == pytest.approx(-.01)
    assert monthly.loc[pd.Period('2020-02'), 'analyst_sharpe'] == pytest.approx(.02)
    assert monthly.sessions.tolist() == [2,1]


def test_closed_trade_returns_include_both_fees_and_reject_unmatched_trades():
    entries = pd.to_datetime(['2020-01-01'])
    exits = pd.to_datetime(['2020-01-22'])
    completed = pd.DataFrame(dict(strategy=['analyst_sharpe'],permno=[1],
                                   entry_date=entries,exit_date=exits,held_sessions=[14],exit_reason=['holding_period']))
    trades = pd.DataFrame(dict(strategy=['analyst_sharpe']*2,permno=[1,1],
                               date=[entries[0],exits[0]],action=['entry','exit'],
                               value=[100.,110.],cost_dollars=[1.,1.1]))
    table = closed_trade_returns({'completed_positions':completed,'trades':trades})
    assert table.net_pnl.iloc[0] == pytest.approx(7.9)
    assert table.net_trade_return.iloc[0] == pytest.approx(7.9/101)
    with pytest.raises(ValueError, match='reconcile'):
        closed_trade_returns({'completed_positions':completed,'trades':trades.iloc[:1]})


def test_empty_trade_sample_and_flat_rolling_metrics():
    empty = pd.DataFrame(columns=['strategy','permno','entry_date','exit_date'])
    assert closed_trade_returns({'completed_positions':empty,'trades':pd.DataFrame()}).empty
    r = pd.Series([0.,0.,0.])
    assert drawdown_panel(r).eq(0).all()
    assert max_underwater_sessions(r) == 0
    sharpe, vol = rolling_statistics(r.to_frame('cash'), window=2)
    assert sharpe.cash.isna().all()
    assert vol.cash.iloc[1:].eq(0).all()


def test_incomplete_and_legacy_run_rejected(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError,match='incomplete'):
        load_run(tmp_path)
    manifest.write_text(json.dumps({'complete':True,'config':{'strategy_mode':'scheduled'}}))
    with pytest.raises(ValueError,match='revision_zscore'):
        load_run(tmp_path)
