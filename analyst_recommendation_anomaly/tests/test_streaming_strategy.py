import json
import signal
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analyst_sharpe import AnalystSharpeConfig, run_strategy
from src.run_analyst_sharpe import demo_inputs
from src.streaming_strategy import run_streaming
from src.notebook_process import run_logged


def write_market(root, daily, benchmark):
    root.mkdir()
    files=[]
    split=benchmark.date.iloc[len(benchmark)//2]
    for i,frame in enumerate([daily[daily.date.lt(split)],daily[daily.date.ge(split)]]):
        name=f'part{i}.parquet'
        frame.to_parquet(root/name,index=False)
        files.append({'file':name})
    benchmark.to_parquet(root/'benchmark.parquet',index=False)
    (root/'manifest.json').write_text(json.dumps({'complete':True,'files':files}))


@pytest.mark.parametrize('lookback,freshness,mode,strategy_mode', [(None,None,'long_only','revision_zscore'),(120,None,'long_only','revision_zscore'),(None,14,'long_only','revision_zscore'),(None,None,'long_short','revision_zscore'),(None,None,'long_short','revision_signal')])
def test_streamed_and_reference_results_match(tmp_path,lookback,freshness,mode,strategy_mode):
    events,daily,benchmark,start,_=demo_inputs()
    end=benchmark.date.iloc[210]
    # Include same-date corrections and withdrawal/re-initiation cases.
    updated=events.iloc[[0]].copy()
    updated['available_date']=benchmark.date.iloc[100]
    updated['rating']=-1
    events=pd.concat([events,updated],ignore_index=True)
    config=replace(AnalystSharpeConfig(),lookback_days=lookback,revision_freshness_sessions=freshness,portfolio_mode=mode,strategy_mode=strategy_mode)
    reference=run_strategy(events,daily,benchmark,start,end,config)
    market=tmp_path/'market'
    write_market(market,daily,benchmark)
    output=tmp_path/'run'
    output.mkdir()
    run_streaming(events,market,start,end,config,output,progress=lambda message:None)
    keys={'daily_results':['strategy','date'],'orders':['strategy','execution_date','permno'],
          'trades':['strategy','date','permno','action'],'holdings':['strategy','date','permno'],
          'completed_positions':['strategy','entry_date','permno'],
          'signals':['date','permno'],'analyst_scores':['cutoff','analyst_id']}
    for name,sort in keys.items():
        actual=pd.read_parquet(output/f'{name}.parquet').sort_values(sort).reset_index(drop=True)
        expected=reference[name].sort_values(sort).reset_index(drop=True)
        actual=actual[expected.columns]
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-8,atol=1e-10)


def test_blank_child_failure_reports_code_signal_and_persistent_log(tmp_path):
    with pytest.raises(RuntimeError,match='return code 7') as error:
        run_logged([sys.executable,'-c','import sys; sys.exit(7)'],tmp_path)
    assert 'No output was produced' in str(error.value)
    assert list((tmp_path/'outputs/presentations/process_logs').glob('*.log'))
    with pytest.raises(RuntimeError,match='SIGKILL'):
        run_logged([sys.executable,'-c','import os,signal; os.kill(os.getpid(),signal.SIGKILL)'],tmp_path)


def test_child_success_must_have_complete_manifest(tmp_path):
    run=tmp_path/'run'
    run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'complete':True}))
    actual=run_logged([sys.executable,'-c',f'print("Artifacts: " + {str(run)!r})'],tmp_path)
    assert actual==run


def test_streamed_presentation_and_sensitivity_do_not_load_full_market(tmp_path):
    from dataclasses import asdict
    from src.presentation import load_run, audit_table, performance_table
    from src.streaming_strategy import replay_saved_sensitivity
    events,daily,benchmark,start,_=demo_inputs()
    end=benchmark.date.iloc[200]
    market=tmp_path/'market'
    write_market(market,daily,benchmark)
    output=tmp_path/'run'
    output.mkdir()
    config=AnalystSharpeConfig()
    run_streaming(events,market,start,end,config,output,progress=lambda _:None)
    manifest={'complete':True,'synthetic':False,'engine':'streaming','config':asdict(config),
              'start':str(start.date()),'end':str(end.date()),
              'inputs':[{'path':str(market/'benchmark.parquet')}]}
    (output/'manifest.json').write_text(json.dumps(manifest))
    _,frames=load_run(output)
    assert frames['signals'].permno.nunique()==1
    assert len(frames['daily_signal_coverage'])==201
    assert audit_table(frames,manifest['config']).passed.all()
    sensitivity=replay_saved_sensitivity(output,config,thresholds=(1.68,),costs=(10.,),progress=lambda _:None)
    expected=performance_table(frames['daily_results']).loc['analyst_sharpe','total_return']
    assert sensitivity.total_return.iloc[0]==pytest.approx(expected,abs=1e-10)
