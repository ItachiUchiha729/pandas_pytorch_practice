import numpy as np
import pandas as pd
import pytest
from src.analyst_sharpe import AnalystSharpeConfig
from src.revision_zscore import simulate_entries, _held_return, _can_execute
from src.streaming_strategy import make_book

@pytest.mark.parametrize('permno,start,price,reopened,ret', [(76841,'2020-11-04',328.9,236.26,-.281666), (10562,'2005-10-20',23.37,21.38,-.085152)])
@pytest.mark.parametrize('strategy_mode', ['revision_zscore', 'revision_signal'])
@pytest.mark.parametrize('mode,side', [('long_only','long'), ('long_short','long'), ('long_short','short')])
@pytest.mark.parametrize('engine', ['reference', 'streaming'])
def test_halt_carries_value_defers_exit_and_retains_reopening_loss(mode, side, engine, permno, start, price, reopened, ret, strategy_mode):
    dates = pd.bdate_range(start, periods=5)
    quotes = pd.DataFrame(dict(date=dates, permno=permno, prc=[price,price,np.nan,reopened,reopened],
        ret_total=[0.,0.,np.nan,ret,0.], is_terminal=False))
    signals = pd.DataFrame(dict(date=dates,permno=permno,eligible=True,signal=1.,unweighted_score=1.,
        weighted_z=2.,unweighted_z=2.,weighted_crossing=[True,False,False,False,False],
        unweighted_crossing=[True,False,False,False,False],weighted_short_crossing=False,unweighted_short_crossing=False))
    if side == 'short':
        for prefix in ['weighted','unweighted']:
            signals[f'{prefix}_short_crossing'] = signals[f'{prefix}_crossing']
            signals[f'{prefix}_crossing'] = False
            signals[f'{prefix}_z'] = -2.
    config=AnalystSharpeConfig(strategy_mode=strategy_mode,portfolio_mode=mode,top_n=1,holding_sessions=1,cost_bps=0.,initial_nav=100.)
    bench=pd.DataFrame(dict(date=dates,ret=0.))
    if engine=='reference':
        result=simulate_entries(signals,quotes,bench,dates[0],dates[-1],config)
    else:
        book=make_book('analyst_sharpe','weighted',dates,config)
        for i in range(1,len(dates)):
            book.step(i,quotes[quotes.date.eq(dates[i])],signals[signals.date.eq(dates[i-1])],0.)
        result=book.frames()
    daily=result['daily_results'].query("strategy == 'analyst_sharpe'").set_index('date')
    assert daily.loc[dates[2],'nav']==100.
    assert daily.loc[dates[3],'nav']==pytest.approx(100*(1+ret) if side == 'long' else 100*(1-ret))
    assert result['completed_positions'].exit_reason.eq('holding_period_after_halt').all()
    assert result['completed_positions'].exit_date.eq(dates[3]).all()
    assert result['valuation_events'].source.str.contains('elpais' if permno==10562 else 'uscourts').all()
    assert reopened/price-1 == pytest.approx(ret,abs=1e-6)
    assert len(result['valuation_events'])==(2 if engine=='reference' else 1)
    assert not _can_execute(quotes[quotes.date.eq(dates[2])].set_index('permno'),permno)


def test_arbitrary_missing_return_still_raises():
    q=pd.DataFrame([dict(permno=76841,ret_total=np.nan,prc=np.nan,is_terminal=False)]).set_index('permno')
    with pytest.raises(ValueError,match='Missing held return'):
        _held_return(q,76841,pd.Timestamp('2020-11-05'),'test',AnalystSharpeConfig())
    q['prc']=328.9
    with pytest.raises(ValueError,match='Missing held return'):
        _held_return(q,76841,pd.Timestamp('2020-11-06'),'test',AnalystSharpeConfig())
