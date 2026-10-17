"""Read-only strategy audit; run from project root with python scripts/audit_signals.py."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.signals import link_recommendations_to_crsp, make_recommendation_signals, attach_latest_available_signal
from src.backtest import prepare_crsp_monthly, run_long_short_backtest
from src.config import BacktestConfig

out = ROOT / 'outputs' / 'audit'
out.mkdir(parents=True, exist_ok=True)
r = pd.read_parquet(ROOT / 'data/raw/ibes_recommendations.parquet')
n = pd.read_parquet(ROOT / 'data/raw/crsp_names.parquet')
c = pd.read_parquet(ROOT / 'data/raw/crsp_monthly.parquet')
s = make_recommendation_signals(link_recommendations_to_crsp(r, n))
previous = s.groupby('permno').shift()
gap = (s.statpers - previous.statpers).dt.days
changed_ticker = previous.ticker.notna() & s.ticker.ne(previous.ticker)
checks = {
    'raw_rows': len(r), 'signal_rows': len(s),
    'invalid_meanrec': int((~s.meanrec.between(1, 5) & s.meanrec.notna()).sum()),
    'consensus_formula_errors': int((s.consensus_signal != -s.meanrec).fillna(False).sum()),
    'revision_formula_errors': int((s.revision_signal != previous.meanrec - s.meanrec).fillna(False).sum()),
    'revision_gaps_over_45_days': int((gap.gt(45) & s.revision_signal.notna()).sum()),
    'revision_ticker_switches': int((changed_ticker & s.revision_signal.notna()).sum()),
    'revision_coverage_changes': int((s.numrec.ne(previous.numrec) & s.revision_signal.ne(0) & s.revision_signal.notna()).sum()),
    'ibes_last_date': str(r.statpers.max()), 'crsp_last_date': str(c.date.max()),
}
s.loc[changed_ticker, ['permno','ticker','statpers','meanrec','revision_signal']].head(30).to_csv(out/'ticker_switch_examples.csv',index=False)
print('Signals audited', checks, flush=True)
# A single sorted as-of join reproduces the per-security implementation efficiently.
m = prepare_crsp_monthly(c)
p = pd.merge_asof(m.sort_values('date'), s.sort_values('available_date'),
                  by='permno', left_on='date', right_on='available_date', direction='backward')
# Independently compare against the production join on a deterministic sample.
ids = m.permno.drop_duplicates().iloc[::500]
reference = attach_latest_available_signal(m[m.permno.isin(ids)], s[s.permno.isin(ids)])
cols = ['permno','date','statpers','consensus_signal','revision_signal']
expected=reference[cols].sort_values(['permno','date']).reset_index(drop=True)
actual=p[p.permno.isin(ids)][cols].sort_values(['permno','date']).reset_index(drop=True)
pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
p=p[p.date.le(s.statpers.max()+pd.offsets.MonthEnd(0)) & p.shrcd.isin([10,11]) & p.exchcd.isin([1,2,3]) & p.price.ge(5) & p.numrec.ge(2)].copy()
p['signal_age_days']=(p.date-p.statpers).dt.days
outcomes=m[['permno','date','ret_total']].rename(columns={'date':'return_date','ret_total':'forward_return'})
outcomes['date']=outcomes.return_date-pd.offsets.MonthEnd(1)
p=p.merge(outcomes,on=['permno','date'],how='left',validate='many_to_one')
checks['future_signal_rows']=int((p.available_date>p.date).sum())
checks['eligible_rows']=len(p)
checks['eligible_signal_age_over_45_days']=int(p.signal_age_days.gt(45).sum())
checks['eligible_signal_age_over_365_days']=int(p.signal_age_days.gt(365).sum())
checks['eligible_signal_age_max_days']=int(p.signal_age_days.max())
reports=[]
costs=[]
for strategy in ['consensus','revision']:
    result,h=run_long_short_backtest(p, BacktestConfig(signal=strategy+'_signal'))
    saved=pd.read_csv(ROOT/f'outputs/{strategy}_monthly_results.csv',parse_dates=['date'])
    for col in ['gross_return','net_return','cost','turnover']:
        np.testing.assert_allclose(result[col].to_numpy(dtype=float,na_value=np.nan),saved[col].to_numpy(dtype=float),equal_nan=True,atol=1e-12)
    held=h[h.weight.ne(0)].copy()
    held['side']=np.where(held.weight.gt(0),'long','short')
    missing=held[held.forward_return.isna()]
    held.groupby('side').agg(rows=('permno','size'),mean_signal=(strategy+'_signal','mean'),mean_age_days=('signal_age_days','mean'),stale_over_45=('signal_age_days',lambda x:x.gt(45).mean())).to_csv(out/f'{strategy}_positions.csv')
    details={
        'strategy':strategy,'held_rows':len(held),
        'held_stale_over_45_days':int(held.signal_age_days.gt(45).sum()),
        'missing_held_rows':len(missing),'absent_outcome_rows':int(missing.return_date.isna().sum()),
        'missing_at_last_formation':int(missing.date.eq(m.date.max()).sum()),
        'long_nonpositive_revision':int((held.weight.gt(0)&held.revision_signal.le(0)).sum()) if strategy=='revision' else None,
        'short_nonnegative_revision':int((held.weight.lt(0)&held.revision_signal.ge(0)).sum()) if strategy=='revision' else None,
        'max_net_exposure_error':float(h.groupby('date').weight.sum().abs().max()),
        'mean_gross_exposure':float(held.assign(abs_weight=held.weight.abs()).groupby('date').abs_weight.sum().mean()),
    }
    valid=result[result.gross_return.notna()]
    details['break_even_cost_bps']=float(valid.gross_return.mean()/valid.turnover.mean()*10000)
    for bps in [0,1,2,5,10,20]:
        net=valid.gross_return-valid.turnover*bps/10000
        costs.append({'strategy':strategy,'cost_bps_per_dollar_traded':bps,'months':len(net),'annual_mean_pct':float(net.mean()*1200),'annual_vol_pct':float(net.std()*np.sqrt(12)*100),'sharpe':float(net.mean()/net.std()*np.sqrt(12))})
    held[['permno','date','statpers','meanrec','consensus_signal','revision_signal','signal_age_days','weight','forward_return']].head(100).to_csv(out/f'{strategy}_holding_examples.csv',index=False)
    reports.append(details)
    print(strategy,details,flush=True)
checks['strategies']=reports
(out/'signal_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
pd.DataFrame(costs).to_csv(out/'cost_sensitivity.csv',index=False)
print(pd.DataFrame(costs).to_string(index=False),flush=True)
