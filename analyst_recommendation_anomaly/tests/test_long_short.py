from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analyst_sharpe import AnalystSharpeConfig
from src.revision_zscore import simulate_entries, trailing_zscores
from src.streaming_strategy import SignalState, entry_candidates, make_book
from src.presentation import closed_trade_returns, trading_table


def inputs():
    dates = pd.bdate_range('2020-01-01', periods=20)
    stocks = pd.DataFrame([dict(date=d, permno=k, ret_total=0., prc=10.,
        is_terminal=False, shrcd=10, exchcd=1) for d in dates for k in [1,2,3]])
    bench = pd.DataFrame({'date':dates, 'ret':0.})
    signals = pd.DataFrame([dict(date=d, permno=k, eligible=True, signal=1. if k==1 else -1.,
        unweighted_score=1. if k==1 else -1., weighted_z=2. if k==1 else -3.,
        unweighted_z=2. if k==1 else -3., weighted_crossing=d==dates[1] and k==1,
        unweighted_crossing=d==dates[1] and k==1, weighted_short_crossing=d==dates[1] and k==2,
        unweighted_short_crossing=d==dates[1] and k==2) for d in dates for k in [1,2,3]])
    config = AnalystSharpeConfig(portfolio_mode='long_short', top_n=10, initial_nav=100000., cost_bps=0.)
    return dates, stocks, bench, signals, config


def test_signed_pnl_exact_hold_and_side_aware_reporting():
    dates, stocks, bench, signals, config = inputs()
    stocks.loc[stocks.date.eq(dates[3]) & stocks.permno.eq(1), 'ret_total'] = .1
    stocks.loc[stocks.date.eq(dates[3]) & stocks.permno.eq(2), 'ret_total'] = -.2
    result = simulate_entries(signals, stocks, bench, dates[0], dates[-1], config)
    daily = result['daily_results'].query("strategy == 'analyst_sharpe'")
    assert daily.nav.iloc[-1] == pytest.approx(103000.)
    assert result['completed_positions'].held_sessions.eq(14).all()
    pnl = closed_trade_returns(result).query("strategy == 'analyst_sharpe'").set_index('side')
    assert pnl.loc['long','net_pnl'] == pytest.approx(1000.)
    assert pnl.loc['short','net_pnl'] == pytest.approx(2000.)
    assert pnl.loc['short','net_trade_return'] == pytest.approx(.2)
    assert (daily.nav == daily.cash + result['holdings'].query("strategy == 'analyst_sharpe'")
            .groupby('date').value.sum().reindex(daily.date,fill_value=0).to_numpy()).all()
    stats = trading_table(result)
    assert stats.long_entries.eq(1).all() and stats.short_entries.eq(1).all()


def test_short_loss_can_exceed_initial_notional():
    dates, stocks, bench, signals, config = inputs()
    signals['weighted_crossing'] = signals['unweighted_crossing'] = False
    stocks.loc[stocks.date.eq(dates[3]) & stocks.permno.eq(2), 'ret_total'] = 1.5
    r = simulate_entries(signals, stocks, bench, dates[0], dates[-1], config)
    assert r['daily_results'].groupby('strategy').nav.last().eq(85000.).all()
    assert np.allclose(closed_trade_returns(r).net_trade_return, -1.5)


def test_short_proceeds_do_not_fund_leverage_and_fees_reconcile():
    dates, stocks, bench, signals, config = inputs()
    config = replace(config, top_n=2, cost_bps=10., short_borrow_bps_annual=252.)
    r = simulate_entries(signals, stocks, bench, dates[0], dates[-1], config)
    days = r['daily_results'].query("strategy == 'analyst_sharpe'")
    assert days.gross_exposure.max() >= 1.  # borrow expense can cause drift, no forced resize
    entry_day = days[days.date.eq(dates[2])].iloc[0]
    assert entry_day.gross_exposure == pytest.approx(1.)
    entries = r['trades'].query("strategy == 'analyst_sharpe' and action == 'entry'")
    assert entries.value.sum() == pytest.approx(100000/1.001)
    assert days.borrow_cost_dollars.sum() == pytest.approx(entries.value.iloc[0]*.0001*14)
    pnl = closed_trade_returns(r).query("strategy == 'analyst_sharpe'")
    assert pnl.net_pnl.sum() == pytest.approx(days.nav.iloc[-1]-config.initial_nav)
    assert days.gross_nav.iloc[-1] == pytest.approx(config.initial_nav)


def test_capacity_absolute_ranking_no_side_flip_and_long_only_default():
    dates, stocks, bench, signals, config = inputs()
    config = replace(config, top_n=1)
    mask = signals.date.eq(dates[3]) & signals.permno.eq(2)
    signals.loc[mask, ['weighted_crossing','unweighted_crossing']] = True
    r = simulate_entries(signals, stocks, bench, dates[0], dates[-1], config)
    assert r['trades'].query("action == 'entry'").side.eq('short').all()
    assert r['orders'].query("execution_date == @dates[4]").status.eq('already_held').all()
    long_only = simulate_entries(signals, stocks, bench, dates[0], dates[-1], replace(config, portfolio_mode='long_only'))
    assert long_only['trades'].query("action == 'entry'").permno.eq(1).all()
    assert AnalystSharpeConfig().portfolio_mode == 'long_only'


def test_short_crossing_requires_valid_prior_z_and_strict_negative_threshold():
    dates = pd.bdate_range('2020-01-01', periods=30)
    values = np.tile([-1., 1.], 15)
    values[23:26] = [-4., -5., -6.]
    panel = pd.DataFrame(dict(date=dates, permno=1, eligible=True, signal=values, unweighted_score=values))
    c = AnalystSharpeConfig(portfolio_mode='long_short')
    r = trailing_zscores(panel, dates, c)
    assert r.loc[23,'weighted_short_crossing']
    assert not r.loc[24,'weighted_short_crossing']
    assert not r.loc[:20,'weighted_short_crossing'].any()
    assert not entry_candidates(r.loc[[23]], replace(c,portfolio_mode='long_only')).shape[0]
    assert len(entry_candidates(r.loc[[23]], c)) == 1


@pytest.mark.parametrize('kwargs', [dict(portfolio_mode='bad'), dict(short_borrow_bps_annual=-1),
    dict(portfolio_mode='long_short', require_positive=True)])
def test_config_rejects_unsupported_short_settings(kwargs):
    with pytest.raises(ValueError):
        AnalystSharpeConfig(**kwargs)
