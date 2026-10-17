import numpy as np
import pandas as pd
import pytest
from src.analyst_sharpe import AnalystSharpeConfig
from src.revision_zscore import simulate_entries, _held_return
from src.streaming_strategy import make_book

@pytest.mark.parametrize('mode,side', [('long_only','long'),('long_short','long'),('long_short','short')])
@pytest.mark.parametrize('engine', ['reference','streaming'])
def test_approved_off_exchange_exit_charges_cost_and_preserves_prior_pnl(mode,side,engine):
    dates=pd.bdate_range('2005-04-05',periods=5)
    stocks=pd.DataFrame(dict(date=dates,permno=63079,prc=[10.,10.,11.,np.nan,np.nan],
        ret_total=[0.,0.,.1,np.nan,np.nan],exchcd=[1,1,1,0,0],is_terminal=False))
    signals=pd.DataFrame(dict(date=dates,permno=63079,eligible=True,signal=1.,unweighted_score=1.,
        weighted_z=2.,unweighted_z=2.))
    for prefix in ['weighted','unweighted']:
        signals[f'{prefix}_crossing']=[side=='long',False,False,False,False]
        signals[f'{prefix}_short_crossing']=[side=='short',False,False,False,False]
        if side=='short':signals[f'{prefix}_z']=-2.
    config=AnalystSharpeConfig(portfolio_mode=mode,top_n=1,initial_nav=100.,cost_bps=100.,missing_terminal_policy='last_close')
    if engine=='reference':
        result=simulate_entries(signals,stocks,pd.DataFrame(dict(date=dates,ret=0.)),dates[0],dates[-1],config)
    else:
        book=make_book('analyst_sharpe','weighted',dates,config)
        for i in range(1,len(dates)):book.step(i,stocks[stocks.date.eq(dates[i])],signals[signals.date.eq(dates[i-1])],0.)
        result=book.frames()
    daily=result['daily_results'].query("strategy=='analyst_sharpe'")
    entry=100/1.01
    expected=100+(1 if side=='long' else -1)*entry*.1-entry*.01-entry*1.1*.01
    assert daily.nav.iloc[-1]==pytest.approx(expected)
    assert result['completed_positions'].exit_reason.eq('off_exchange_close_proxy').all()
    assert result['completed_positions'].exit_date.eq(dates[3]).all()
    assert result['valuation_events'].event.eq('off_exchange_close_proxy').all()
    assert result['open_positions'].empty
    exits=result['trades'].query("action=='exit'")
    assert np.allclose(exits.value,entry*1.1)
    assert np.allclose(exits.cost_dollars,entry*1.1*.01)

@pytest.mark.parametrize('exchange,price,policy',[(1,np.nan,'last_close'),(np.nan,np.nan,'last_close'),(0,10.,'last_close'),(0,np.nan,'error')])
def test_no_unapproved_or_general_missing_return_fill(exchange,price,policy):
    q=pd.DataFrame([dict(permno=63079,prc=price,ret_total=np.nan,exchcd=exchange,is_terminal=False)]).set_index('permno')
    with pytest.raises(ValueError,match='Missing held return'):
        _held_return(q,63079,pd.Timestamp('2005-04-08'),'test',AnalystSharpeConfig(missing_terminal_policy=policy))
