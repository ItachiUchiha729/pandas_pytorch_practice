from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.analyst_sharpe import AnalystSharpeConfig,normalize_events
from src.revision_zscore import revision_signals,raw_signal_triggers,simulate_entries
from src.streaming_strategy import SignalState


def test_positive_only_weights_and_two_positive_contributors():
    dates=pd.bdate_range('2020-01-01',periods=3)
    events=normalize_events(pd.DataFrame([dict(permno=1,analyst_id=a,recommendation_date=dates[i],available_date=dates[i],rating=rating)
       for a,ratings in [('a',[0,2]),('b',[0,-1]),('c',[2,-2]),('d',[2,-2])] for i,rating in enumerate(ratings)]))
    scores=pd.DataFrame(dict(analyst_id=['a','b','c','d'],score=[3.,1.,-100.,0.]))
    quotes=pd.DataFrame([dict(permno=1,prc=10.,shrcd=11,exchcd=1,is_terminal=False)])
    config=AnalystSharpeConfig(strategy_mode='revision_signal',portfolio_mode='long_short',top_n=200)
    signal,active=revision_signals(events,scores,quotes,dates,dates[-1],config)
    assert set(active.analyst_id)=={'a','b'}
    assert signal.signal.iloc[0]==pytest.approx(.75*2+.25*(-1))
    assert signal.n_active_scored.iloc[0]==2 and signal.eligible.iloc[0]
    assert signal.unweighted_score.iloc[0]==.5
    weights=active.influence/active.influence.sum()
    assert weights.sum()==pytest.approx(1.)
    scores.loc[scores.analyst_id.eq('b'),'score']=0.
    signal,_=revision_signals(events,scores,quotes,dates,dates[-1],config)
    assert signal.n_active_scored.iloc[0]==1 and not signal.eligible.iloc[0]


def test_strict_thresholds_without_zscore_or_crossing():
    config=AnalystSharpeConfig(strategy_mode='revision_signal',portfolio_mode='long_short')
    x=raw_signal_triggers(pd.DataFrame(dict(signal=[1.,1.01,-1.,-1.01,np.nan],unweighted_score=[1.,1.01,-1.,-1.01,np.nan])),config)
    assert x.weighted_crossing.tolist()==[False,True,False,False,False]
    assert x.weighted_short_crossing.tolist()==[False,False,False,True,False]
    assert x.weighted_z.isna().all()


def test_200_shared_slots_raw_strength_order_and_half_percent_target():
    dates=pd.bdate_range('2020-01-01',periods=18)
    config=AnalystSharpeConfig(strategy_mode='revision_signal',portfolio_mode='long_short',top_n=200,cost_bps=0.,initial_nav=100000.)
    stocks=pd.DataFrame([dict(date=d,permno=i,prc=10.,ret_total=0.,is_terminal=False) for d in dates for i in range(201)])
    signals=pd.DataFrame([dict(date=d,permno=i,eligible=True,signal=(1 if i%2 else -1)*(1.1+i/1000),unweighted_score=(1 if i%2 else -1)*(1.1+i/1000)) for d in dates for i in range(201)])
    signals=raw_signal_triggers(signals,config)
    result=simulate_entries(signals,stocks,pd.DataFrame(dict(date=dates,ret=0.)),dates[0],dates[-1],config)
    entries=result['trades'].query("strategy=='analyst_sharpe' and action=='entry'")
    first=entries[entries.date.eq(dates[1])]
    assert len(first)==200 and set(first.permno)==set(range(1,201))
    assert first.value.eq(500.).all()
    assert set(first.side)=={'long','short'}
    assert first.reason.eq('raw_signal_threshold').all()
    assert result['daily_results'].n_holdings.max()==200
    assert result['completed_positions'].held_sessions.eq(14).all()
    # Persistent levels may re-enter on the day after exit, never reset while held.
    assert entries.date.eq(dates[16]).any()
    assert set(entries.loc[entries.date.eq(dates[15]), 'permno']) == {0}  # New name may enter; exited names cannot.


def test_signed_weights_rejected_for_raw_mode():
    with pytest.raises(ValueError,match='positive Sharpe'):
        AnalystSharpeConfig(strategy_mode='revision_signal',negative_scores='signed')
