"""Bounded-memory execution of the revision strategy on large daily extracts.

Market files are read one at a time; labels are built in security buckets. Only
current analyst state, a short signal ring buffer and open positions remain in
memory. Large diagnostic panels are appended to Parquet instead of accumulated.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd

from .revision_zscore import _verified_halt, _halt_event, HALT_COLUMNS, _off_exchange_proxy, _off_exchange_event
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .analyst_sharpe import normalize_events, prepare_market, build_outcomes, summarize
from .revision_zscore import _check_execution, _can_execute, _held_return, entry_strength_column, raw_signal_triggers


class PanelWriter:
    def __init__(self, path, empty, batch_size=50000):
        self.path = Path(path)
        self.schema = pa.Schema.from_pandas(empty, preserve_index=False)
        self.writer = pq.ParquetWriter(self.path, self.schema, compression='zstd')
        self.parts, self.pending, self.rows = [], 0, 0
        self.batch_size = batch_size

    def append(self, frame):
        if len(frame):
            self.parts.append(frame.reindex(columns=self.schema.names))
            self.pending += len(frame)
            self.rows += len(frame)
            if self.pending >= self.batch_size:
                self.flush()

    def flush(self):
        if self.parts:
            frame = pd.concat(self.parts, ignore_index=True)
            self.writer.write_table(pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False))
            self.parts.clear()
            self.pending = 0

    def close(self):
        self.flush()
        self.writer.close()


def empty_frame(spec):
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in spec.items()})


SIGNAL_SPEC = {**{k:'float64' for k in ['prc','shrcd','exchcd','n_active_scored','numerator','denominator','signal','unweighted_score',
                                     'weighted_mean','weighted_std','weighted_z','unweighted_mean','unweighted_std','unweighted_z']},
               'permno':'int64','date':'datetime64[ns]','is_terminal':'bool','eligible':'bool',
               'weighted_crossing':'bool','unweighted_crossing':'bool',
               'weighted_short_crossing':'bool','unweighted_short_crossing':'bool','exclusion_reason':'string'}
SCORE_SPEC = {'analyst_id':'string','n_calls':'int64','mean':'float64','std':'float64','score':'float64','cutoff':'datetime64[ns]'}
CONTRIB_SPEC = {'permno':'int64','analyst_id':'string','recommendation_date':'datetime64[ns]',
    'available_date':'datetime64[ns]','rating':'float64','event_type':'string','event_id':'string',
    'previous_rating':'float64','previous_event_type':'string','previous_recommendation_date':'datetime64[ns]',
    'revision':'float64','announcement_session':'int64','age_sessions':'int64','n_calls':'int64','mean':'float64',
    'std':'float64','score':'float64','influence':'float64','contribution':'float64','abs_influence':'float64','cutoff':'datetime64[ns]'}


def market_files(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Daily market extract is incomplete')
    return [root / item['file'] for item in manifest['files']]


def market_days(root, benchmark, end=None):
    """Normalize one nonoverlapping chronological file at a time, never all years."""
    prior_date, terminal_ids = None, set()
    for path in market_files(root):
        frame = pd.read_parquet(path)
        if end is not None:
            frame = frame[pd.to_datetime(frame.date).le(end)]
        if frame.empty:
            continue
        stocks, _ = prepare_market(frame, benchmark)
        del frame
        if prior_date is not None and stocks.date.min() <= prior_date:
            raise ValueError('Daily market files must have nonoverlapping chronological date ranges')
        repeated = stocks.permno.isin(terminal_ids)
        if repeated.any():
            ids = stocks.loc[repeated, 'permno'].unique()[:5]
            raise ValueError(f'Rows after terminal event across market files: {ids.tolist()}')
        terminal_ids.update(stocks.loc[stocks.is_terminal, 'permno'].tolist())
        prior_date = stocks.date.max()
        # iloc slices avoid materializing a dictionary of all daily DataFrames.
        dates = stocks.date.to_numpy()
        breaks = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1])+1, len(stocks)]
        for first, last in zip(breaks[:-1], breaks[1:]):
            yield pd.Timestamp(dates[first]), stocks.iloc[first:last]


def bucket_outcomes(events, root, benchmark, config, destination, progress, buckets=32):
    """Transpose only relevant stock rows to temporary buckets, then label calls."""
    security_ids = set()
    relevant = events.permno.unique()
    with tempfile.TemporaryDirectory(prefix='outcome_buckets_', dir=destination) as temporary:
        location = Path(temporary)
        for index, path in enumerate(market_files(root)):
            frame = pd.read_parquet(path)
            security_ids.update(frame.permno.unique().tolist())
            frame = frame[frame.permno.isin(relevant)].copy()
            if len(frame):
                frame['_bucket'] = frame.permno % buckets
                ds.write_dataset(pa.Table.from_pandas(frame, preserve_index=False), location,
                                 format='parquet', partitioning=['_bucket'],
                                 basename_template=f'part{index}-{{i}}.parquet', existing_data_behavior='overwrite_or_ignore')
            progress(f'Partitioning daily inputs: {index+1}/{len(market_files(root))} files')
        pieces = []
        for bucket in range(buckets):
            calls = events[events.permno.mod(buckets).eq(bucket)]
            if calls.empty:
                continue
            files = list((location / str(bucket)).glob('*.parquet'))
            if files:
                frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
                stocks, bench = prepare_market(frame, benchmark)
                del frame
            else:
                stocks = empty_frame({'permno':'int64','date':'datetime64[ns]','prc':'float64',
                                      'ret_total':'float64','is_terminal':'bool'})
                bench = benchmark
            pieces.append(build_outcomes(calls, stocks, bench, config))
            progress(f'Building complete historical labels: bucket {bucket+1}/{buckets}')
        outcomes = pd.concat(pieces).sort_index() if pieces else events.copy()
    return outcomes, np.array(sorted(security_ids | set(relevant)), dtype=np.int64)


class AnalystState:
    """Update only new versions, newly matured labels and rolling expirations."""
    def __init__(self, outcomes, calendar, config):
        self.frame, self.calendar, self.config = outcomes, calendar, config
        self.analyst_code, self.analyst_names = pd.factorize(outcomes.analyst_id, sort=True)
        self.call_code = outcomes.groupby(['analyst_id','permno','recommendation_date'], sort=False).ngroup().to_numpy()
        self.pair_code = outcomes.groupby(['analyst_id','permno'], sort=False).ngroup().to_numpy()
        self.call_latest = np.full(int(self.call_code.max())+1 if len(outcomes) else 0, -1, dtype=np.int64)
        pairs = int(self.pair_code.max())+1 if len(outcomes) else 0
        self.latest = np.full(pairs, -1, dtype=np.int64)
        self.previous = self.latest.copy()
        self.n = np.zeros(len(self.analyst_names), dtype=np.int64)
        self.mean, self.m2 = np.zeros(len(self.n)), np.zeros(len(self.n))
        self.included = np.zeros(len(outcomes), dtype=bool)
        self.value = outcomes.analyst_return.to_numpy()
        self.rating = outcomes.rating.to_numpy()
        self.valid_rating = outcomes.event_type.eq('rating').to_numpy()
        self.rec_date = outcomes.recommendation_date.to_numpy()
        self.rec_session = calendar.searchsorted(outcomes.recommendation_date)
        self.rec_in_calendar = outcomes.recommendation_date.ge(calendar[0]).to_numpy()
        self.available_session = calendar.searchsorted(outcomes.available_date)
        self.mature_session = calendar.searchsorted(outcomes.outcome_available_date)
        self.by_available, self.by_mature, self.by_expiry = defaultdict(list), defaultdict(list), defaultdict(list)
        self.expiry = np.full(len(outcomes), len(calendar)+1, dtype=np.int64)
        if config.lookback_days is not None:
            last = outcomes.available_date + pd.Timedelta(config.lookback_days+1, unit='D').as_unit('ns')
            self.expiry = calendar.searchsorted(last)
        for row in range(len(outcomes)):
            self.by_available[self.available_session[row]].append(row)
            if np.isfinite(self.value[row]):
                self.by_mature[self.mature_session[row]].append(row)
            if config.lookback_days is not None:
                self.by_expiry[self.expiry[row]].append(row)

    def remove(self, row):
        if row < 0 or not self.included[row]:
            return
        a, value = self.analyst_code[row], self.value[row]
        old_n = self.n[a]
        if old_n == 1:
            self.n[a], self.mean[a], self.m2[a] = 0, 0., 0.
        else:
            new_mean = (old_n*self.mean[a] - value)/(old_n-1)
            self.m2[a] = max(0., self.m2[a] - (value-self.mean[a])*(value-new_mean))
            self.mean[a], self.n[a] = new_mean, old_n-1
        self.included[row] = False

    def add(self, row, day):
        if (self.included[row] or self.call_latest[self.call_code[row]] != row or
            not np.isfinite(self.value[row]) or self.expiry[row] <= day or self.mature_session[row] > day):
            return
        a, value = self.analyst_code[row], self.value[row]
        self.n[a] += 1
        delta = value-self.mean[a]
        self.mean[a] += delta/self.n[a]
        self.m2[a] += delta*(value-self.mean[a])
        self.included[row] = True

    def advance(self, day):
        for row in self.by_available.get(day, []):
            call, pair = self.call_code[row], self.pair_code[row]
            self.remove(self.call_latest[call])
            self.call_latest[call] = row
            top, second = self.latest[pair], self.previous[pair]
            if top < 0 or self.rec_date[row] > self.rec_date[top]:
                self.latest[pair], self.previous[pair] = row, top
            elif self.rec_date[row] == self.rec_date[top]:
                self.latest[pair] = row
            elif second < 0 or self.rec_date[row] >= self.rec_date[second]:
                self.previous[pair] = row
            self.add(row, day)
        for row in self.by_mature.get(day, []):
            self.add(row, day)
        for row in self.by_expiry.get(day, []):
            self.remove(row)
        std = np.sqrt(np.divide(self.m2, self.n-1, out=np.full(len(self.n), np.nan), where=self.n>1))
        valid = (self.n >= self.config.min_calls) & (std > self.config.min_std)
        score = np.divide(self.mean, std, out=np.full(len(self.n), np.nan), where=valid)
        selected = np.flatnonzero(np.isfinite(score))
        scores = pd.DataFrame({'analyst_id':self.analyst_names[selected], 'n_calls':self.n[selected],
                               'mean':self.mean[selected], 'std':std[selected], 'score':score[selected],
                               'cutoff':self.calendar[day]})
        pairs = np.flatnonzero((self.latest >= 0) & (self.previous >= 0))
        top, previous = self.latest[pairs], self.previous[pairs]
        age = day-self.rec_session[top]
        valid = (self.valid_rating[top] & self.valid_rating[previous] &
                 (age >= 0) & np.isfinite(score[self.analyst_code[top]]))
        if self.config.revision_freshness_sessions is not None:
            valid &= self.rec_in_calendar[top] & (age < self.config.revision_freshness_sessions)
        top, previous, age = top[valid], previous[valid], age[valid]
        active = self.frame.iloc[top][['permno','analyst_id','recommendation_date','available_date','rating','event_type','event_id']].copy()
        a = self.analyst_code[top]
        active['previous_rating'] = self.rating[previous]
        active['previous_event_type'] = 'rating'
        active['previous_recommendation_date'] = self.rec_date[previous]
        active['revision'] = self.rating[top]-self.rating[previous]
        active['announcement_session'], active['age_sessions'] = self.rec_session[top], age
        active['n_calls'], active['mean'], active['std'], active['score'] = self.n[a], self.mean[a], std[a], score[a]
        if self.config.strategy_mode == 'revision_signal':
            active = active[active.score.gt(0)].copy()
        active['influence'] = active.score.clip(lower=0) if self.config.negative_scores == 'exclude' else active.score
        active['contribution'] = active.influence*active.revision
        active['abs_influence'] = active.influence.abs()
        active['cutoff'] = self.calendar[day]
        return scores, active


class SignalState:
    def __init__(self, ids, config):
        self.ids, self.config = ids, config
        self.buffers = {prefix: np.full((config.zscore_window, len(ids)), np.nan)
                        for prefix in ['weighted','unweighted']}
        self.previous_z = {prefix: np.full(len(ids), np.nan) for prefix in self.buffers}

    def step(self, day, date, quotes, active):
        signal = quotes[['permno','prc','shrcd','exchcd','is_terminal']].copy()
        grouped = active.groupby('permno').agg(n_active_scored=('analyst_id','nunique'),
            numerator=('contribution','sum'),denominator=('abs_influence','sum'),unweighted_score=('revision','mean'))
        grouped['signal'] = grouped.numerator/grouped.denominator.replace(0, np.nan)
        signal = signal.merge(grouped.reset_index(), on='permno', how='outer')
        signal['is_terminal'] = signal.is_terminal.eq(True)
        for column in ['n_active_scored','numerator','denominator','unweighted_score','signal']:
            signal[column] = signal[column].fillna(0.)
        known = signal.prc.notna() & signal.prc.gt(0) & ~signal.is_terminal
        signal.loc[~known,['signal','unweighted_score']] = np.nan
        universe = known & signal.prc.ge(self.config.min_price) & signal.shrcd.isin(self.config.allowed_share_codes) & signal.exchcd.isin(self.config.allowed_exchange_codes)
        signal['eligible'] = universe & signal.n_active_scored.ge(self.config.min_active_analysts) & signal.denominator.gt(0)
        signal['exclusion_reason'] = np.select([~known,~universe,~signal.n_active_scored.ge(self.config.min_active_analysts),~signal.denominator.gt(0)],
            ['missing_price_or_terminal','universe_screen','insufficient_scored_analysts','zero_influence'], default='')
        signal['date'] = date
        if self.config.strategy_mode == 'revision_signal':
            return raw_signal_triggers(signal, self.config)
        positions = np.searchsorted(self.ids, signal.permno.to_numpy())
        for prefix, column in [('weighted','signal'),('unweighted','unweighted_score')]:
            buffer = self.buffers[prefix]
            complete = np.isfinite(buffer).all(axis=0)
            mean = np.mean(buffer, axis=0)
            std = np.std(buffer, axis=0, ddof=1)
            mean[~complete], std[~complete] = np.nan, np.nan
            current = np.full(len(self.ids), np.nan)
            current[positions] = signal[column]
            z = np.divide(current-mean, std, out=np.full(len(self.ids), np.nan), where=std>self.config.min_std)
            crossing = (z > self.config.entry_zscore) & (self.previous_z[prefix] <= self.config.entry_zscore)
            signal[f'{prefix}_mean'], signal[f'{prefix}_std'] = mean[positions], std[positions]
            signal[f'{prefix}_z'], signal[f'{prefix}_crossing'] = z[positions], crossing[positions]
            signal[f'{prefix}_short_crossing'] = ((z < -self.config.entry_zscore) &
                (self.previous_z[prefix] >= -self.config.entry_zscore))[positions]
            self.previous_z[prefix] = z
            buffer[day % self.config.zscore_window] = current
        return signal


class PortfolioBook:
    """One strategy's positions and cash; identical policy to simulate_entries."""
    def __init__(self, strategy, prefix, calendar, config):
        self.strategy, self.prefix, self.calendar, self.config = strategy, prefix, calendar, config
        self.pos, self.gross_pos, self.metadata = {}, {}, {}
        self.cash = self.gross_cash = self.benchmark_nav = config.initial_nav
        self.daily, self.trades, self.holdings, self.orders, self.completed = [], [], [], [], []
        self.valuation_events = []

    def step(self, index, quotes, signals, benchmark_return):
        date, cutoff = self.calendar[index], self.calendar[index-1]
        strategy, prefix, config = self.strategy, self.prefix, self.config
        rate = config.cost_bps/10000
        pos, gross_pos, metadata = self.pos, self.gross_pos, self.metadata
        cash, gross_cash = self.cash, self.gross_cash
        before, gross_before = cash+sum(pos.values()), gross_cash+sum(gross_pos.values())
        held_before = set(pos)
        cost, traded = 0., 0.
        quotes = quotes.set_index('permno')
        for permno in list(pos):
            ret, proxy_exit = _held_return(quotes,permno,date,strategy,config)
            pos[permno] *= 1+ret
            gross_pos[permno] *= 1+ret
            if _verified_halt(quotes, permno, date):
                self.valuation_events.append(_halt_event(strategy, permno, date))
                continue
            off_exchange = proxy_exit and _off_exchange_proxy(quotes, permno, config)
            if off_exchange:
                self.valuation_events.append(_off_exchange_event(strategy, permno, date))
            terminal = bool(quotes.loc[permno,'is_terminal']) or off_exchange
            if terminal or index >= metadata[permno]['exit_index']:
                if not terminal:
                    _check_execution(quotes, permno, date)
                value = pos.pop(permno)
                fee = 0. if terminal and not proxy_exit else value*rate
                cash += value-fee
                gross_cash += gross_pos.pop(permno)
                cost += fee
                traded += 0. if terminal and not proxy_exit else value
                info = metadata.pop(permno)
                reason = 'off_exchange_close_proxy' if off_exchange else 'terminal_close_proxy' if proxy_exit else 'terminal' if terminal else ('holding_period_after_halt' if index > info['exit_index'] else 'holding_period')
                self.completed.append(dict(strategy=strategy,permno=permno,**info,exit_date=date,
                                           exit_reason=reason,held_sessions=index-info['entry_index']))
                self.trades.append(dict(strategy=strategy,date=date,permno=permno,action='exit',reason=reason,value=value,cost_dollars=fee))
        candidates = signals[signals.eligible & signals[f'{prefix}_crossing']].sort_values(
            [entry_strength_column(prefix, config),'permno'],ascending=[False,True])
        if config.require_positive:
            candidates = candidates[candidates['signal' if prefix=='weighted' else 'unweighted_score'].gt(0)]
        accepted, free = [], config.top_n-len(pos)
        for candidate in candidates.itertuples(index=False):
            permno = int(candidate.permno)
            reason = 'already_held' if permno in held_before else 'capacity' if len(accepted)>=free else 'entry'
            if reason == 'entry' and cash <= 1e-8:
                reason = 'no_cash'
            if reason == 'entry' and not _can_execute(quotes,permno):
                reason = 'untradable_at_execution'
            self.orders.append(dict(strategy=strategy,cutoff=cutoff,execution_date=date,permno=permno,
                                    zscore=getattr(candidate,f'{prefix}_z'),entry_signal=getattr(candidate,'signal' if prefix=='weighted' else 'unweighted_score'),status=reason))
            if reason == 'entry':
                _check_execution(quotes,permno,date)
                accepted.append(permno)
        nav_before_buys, gross_nav_before_buys = cash+sum(pos.values()), gross_cash+sum(gross_pos.values())
        if accepted:
            amount = min(nav_before_buys/config.top_n,cash/(len(accepted)*(1+rate)))
            gross_amount = min(gross_nav_before_buys/config.top_n,gross_cash/len(accepted))
            for permno in accepted:
                fee = amount*rate
                cash -= amount+fee
                gross_cash -= gross_amount
                pos[permno],gross_pos[permno] = amount,gross_amount
                exit_index = index+config.holding_sessions
                metadata[permno] = {'entry_date':date,'entry_index':index,'exit_index':exit_index,
                                   'planned_exit_date':self.calendar[exit_index] if exit_index<len(self.calendar) else pd.NaT}
                cost += fee
                traded += amount
                self.trades.append(dict(strategy=strategy,date=date,permno=permno,action='entry',
                                        reason='raw_signal_threshold' if config.strategy_mode=='revision_signal' else 'zscore_crossing',value=amount,cost_dollars=fee))
        if cash < -1e-7 or gross_cash < -1e-7:
            raise AssertionError('Negative cash')
        self.cash,self.gross_cash = max(0.,cash),max(0.,gross_cash)
        nav,gross_nav = self.cash+sum(pos.values()),self.gross_cash+sum(gross_pos.values())
        if not np.isfinite(nav) or nav<=0 or not np.isfinite(gross_nav) or gross_nav<=0:
            raise ValueError(f'Portfolio NAV exhausted or invalid on {date.date()}')
        self.benchmark_nav *= 1+benchmark_return
        self.daily.append(dict(strategy=strategy,date=date,net_return=nav/before-1,
            gross_return=gross_nav/gross_before-1,nav=nav,gross_nav=gross_nav,
            benchmark_return=benchmark_return,benchmark_nav=self.benchmark_nav,cost_dollars=cost,
            turnover=traded/(nav+cost),cash=self.cash,n_holdings=len(pos)))
        self.holdings.extend(dict(strategy=strategy,date=date,permno=k,value=v,weight=v/nav,
            entry_date=metadata[k]['entry_date'],planned_exit_date=metadata[k]['planned_exit_date']) for k,v in pos.items())

    def frames(self):
        open_positions = [dict(strategy=self.strategy,permno=k,value=v,**self.metadata[k]) for k,v in self.pos.items()]
        return {
            'valuation_events':pd.DataFrame(self.valuation_events, columns=HALT_COLUMNS),
            'daily_results':pd.DataFrame(self.daily),
            'trades':pd.DataFrame(self.trades,columns=['strategy','date','permno','action','reason','value','cost_dollars']),
            'holdings':pd.DataFrame(self.holdings,columns=['strategy','date','permno','value','weight','entry_date','planned_exit_date']),
            'orders':pd.DataFrame(self.orders,columns=['strategy','cutoff','execution_date','permno','zscore','entry_signal','status']),
            'completed_positions':pd.DataFrame(self.completed,columns=['strategy','permno','entry_date','entry_index','exit_index','planned_exit_date','exit_date','exit_reason','held_sessions']),
            'open_positions':pd.DataFrame(open_positions,columns=['strategy','permno','value','entry_date','entry_index','exit_index','planned_exit_date'])}


def run_streaming(events, market_root, start, end, config, destination, progress=print):
    if config.strategy_mode not in ('revision_zscore', 'revision_signal'):
        raise ValueError('Bounded-memory execution supports revision_zscore mode; use --demo for legacy experiments')
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    events = normalize_events(events)
    benchmark = pd.read_parquet(Path(market_root)/'benchmark.parquet').sort_values('date').reset_index(drop=True)
    # Validate the benchmark independently without allocating a market panel.
    benchmark['date'] = pd.to_datetime(benchmark.date).dt.normalize()
    benchmark['ret'] = pd.to_numeric(benchmark.ret,errors='coerce')
    if (benchmark.date.duplicated().any() or benchmark.date.isna().any() or
        not np.isfinite(benchmark.ret).all() or benchmark.ret.lt(-1).any() or
        benchmark.date.diff().gt(pd.Timedelta(7,unit='D').as_unit('ns')).any()):
        raise ValueError('Missing/invalid daily benchmark calendar or returns')
    calendar = pd.DatetimeIndex(benchmark.date)
    start,end = pd.Timestamp(start),pd.Timestamp(end)
    if start>end or start<calendar[0] or end>calendar[-1]:
        raise ValueError('Requested date range must lie inside the benchmark calendar')
    start_index = calendar.searchsorted(start)
    if start_index+1>=len(calendar) or calendar[start_index+1]>end:
        raise ValueError('Date range contains no next-session execution')
    progress(f'Preparing {len(events):,} events with bounded market-data buffers')
    outcomes, ids = bucket_outcomes(events,market_root,benchmark,config,destination,progress)
    events.to_parquet(destination/'events.parquet',index=False)
    outcomes.to_parquet(destination/'outcomes.parquet',index=False)
    state, signals_state = AnalystState(outcomes,calendar,config), SignalState(ids,config)
    writers = {name: PanelWriter(destination/f'{name}.parquet',empty_frame(spec)) for name,spec in
               [('signals',SIGNAL_SPEC),('analyst_scores',SCORE_SPEC),('contributions',CONTRIB_SPEC)]}
    books = [make_book('analyst_sharpe','weighted',calendar,config), make_book('unweighted','unweighted',calendar,config)]
    coverage, ages = [], np.zeros(len(calendar),dtype=np.int64)
    audit = {'scores_meet_minimum':True,'contributions_fresh':True,'contributions_available':True}
    last_scores = empty_frame(SCORE_SPEC)
    iterator = iter(market_days(market_root,benchmark,end))
    current = next(iterator,None)
    empty_quotes = empty_frame({'permno':'int64','date':'datetime64[ns]','prc':'float64','shrcd':'float64',
                               'exchcd':'float64','is_terminal':'bool','ret_total':'float64'})
    previous_signal = empty_frame(SIGNAL_SPEC)
    try:
        for day,date in enumerate(calendar):
            if date>end:
                break
            if current is not None and current[0]==date:
                quotes = current[1]
            else:
                quotes = empty_quotes
            if day>start_index:
                for book in books:
                    book.step(day,quotes,previous_signal,benchmark.ret.iloc[day])
            scores,active = state.advance(day)
            signal = signals_state.step(day,date,quotes,active)
            writers['signals'].append(signal)
            writers['analyst_scores'].append(scores)
            writers['contributions'].append(active)
            last_scores = scores.copy()
            coverage.append({'date':date,'observed_stocks':int(signal.prc.notna().sum()),
                             'eligible_stocks':int(signal.eligible.sum()),'valid_weighted_z':int(signal.weighted_z.notna().sum()),
                             'upward_crossings':int(signal.weighted_crossing.sum())})
            if len(active):
                ages += np.bincount(active.age_sessions.to_numpy(),minlength=len(ages))
                if config.revision_freshness_sessions is not None:
                    audit['contributions_fresh'] &= bool(active.age_sessions.between(0,config.revision_freshness_sessions-1).all())
                audit['contributions_available'] &= bool(active.available_date.le(date).all())
            audit['scores_meet_minimum'] &= bool(scores.n_calls.ge(config.min_calls).all())
            # Portfolio decisions need only eligible crossing rows from yesterday.
            previous_signal = entry_candidates(signal, config)
            if current is not None and current[0]==date:
                current = next(iterator,None)
            if day % 63 == 0 or date == end:
                progress(f'Session {day+1}/{calendar.searchsorted(end,side="right")}: {date.date()}, '
                         f'{len(scores):,} scored analysts, {len(active):,} active revisions')
    finally:
        for writer in writers.values():
            writer.close()
    frames_by_book = [book.frames() for book in books]
    frames = {name:pd.concat([f[name] for f in frames_by_book],ignore_index=True) for name in frames_by_book[0]}
    frames['summary'] = summarize(frames['daily_results'],config)
    for name,frame in frames.items():
        frame.to_parquet(destination/f'{name}.parquet',index=False)
    frames['summary'].to_csv(destination/'summary.csv',index=False)
    pd.DataFrame(coverage).to_parquet(destination/'daily_signal_coverage.parquet',index=False)
    pd.DataFrame({'age_sessions':np.arange(len(ages)),'contributions':ages}).to_parquet(destination/'revision_age_counts.parquet',index=False)
    last_scores.to_parquet(destination/'latest_analyst_scores.parquet',index=False)
    (destination/'streaming_audit.json').write_text(json.dumps(audit,indent=2))
    rows = {name:len(frame) for name,frame in frames.items()}
    rows.update({name:writer.rows for name,writer in writers.items()})
    rows.update(events=len(events),outcomes=len(outcomes))
    return frames['summary'],rows,outcomes.outcome_status.value_counts().to_dict()


def parquet_days(path, columns):
    """Yield complete dates, including dates split across row groups/batches."""
    carry = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=columns):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry,frame],ignore_index=True)
        last_date = frame.date.iloc[-1]
        complete = frame[frame.date.lt(last_date)]
        carry = frame[frame.date.eq(last_date)].copy()
        for date,group in complete.groupby('date',sort=True):
            yield date,group
    if carry is not None:
        yield carry.date.iloc[0],carry


def replay_saved_sensitivity(run_dir, config, thresholds=(1.28,1.68,2.), costs=(0.,5.,10.,20.), progress=print, baseline_output=None):
    """Replay scenarios using saved signals and yearly market files.

    baseline_output optionally saves the baseline portfolio frames only; callers
    retain signal provenance and assemble a complete run manifest separately.
    Signal-generation settings must match the source run.
    """
    from dataclasses import replace
    from .presentation import performance_table
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/'manifest.json').read_text())
    if (config.strategy_mode == 'revision_signal') != (manifest['config']['strategy_mode'] == 'revision_signal'):
        raise ValueError('Raw-signal mode requires newly generated positive-only contributor panels; do not reuse z-score panels')
    root = next(Path(item['path']).parent for item in manifest['inputs'] if Path(item['path']).name=='benchmark.parquet')
    benchmark = pd.read_parquet(root/'benchmark.parquet').sort_values('date').reset_index(drop=True)
    calendar = pd.DatetimeIndex(benchmark.date)
    start,end = pd.Timestamp(manifest['start']),pd.Timestamp(manifest['end'])
    start_index = calendar.searchsorted(start)
    threshold_field = 'entry_signal_threshold' if config.strategy_mode=='revision_signal' else 'entry_zscore'
    baseline_threshold = getattr(config, threshold_field)
    scenarios = [('baseline',baseline_threshold,config.cost_bps)]
    scenarios += [('threshold',x,config.cost_bps) for x in thresholds if x != baseline_threshold]
    scenarios += [('cost',baseline_threshold,x) for x in costs if x != config.cost_bps]
    settings = [replace(config,**{threshold_field:float(t)},cost_bps=float(c)) for _,t,c in scenarios]
    books = [[make_book('analyst_sharpe','weighted',calendar,c),make_book('unweighted','unweighted',calendar,c)] for c in settings]
    columns = ['date','permno','eligible','signal','unweighted_score','weighted_z','unweighted_z']
    signal_iterator = iter(parquet_days(run_dir/'signals.parquet',columns))
    signal_day = next(signal_iterator,None)
    market_iterator = iter(market_days(root,benchmark,end))
    market_day = next(market_iterator,None)
    previous_z = {'weighted':{},'unweighted':{}}
    prior = [empty_frame(SIGNAL_SPEC) for _ in settings]
    empty_quotes = empty_frame({'permno':'int64','prc':'float64','ret_total':'float64','is_terminal':'bool'})
    for index,date in enumerate(calendar):
        if date>end:
            break
        quotes = market_day[1] if market_day is not None and market_day[0]==date else empty_quotes
        if index>start_index:
            for pair,candidates in zip(books,prior):
                for book in pair:
                    book.step(index,quotes,candidates,benchmark.ret.iloc[index])
        signal = signal_day[1] if signal_day is not None and signal_day[0]==date else empty_frame(SIGNAL_SPEC)
        lagged = {prefix:signal.permno.map(previous_z[prefix]) for prefix in previous_z}
        next_prior = []
        for c in settings:
            current = signal.copy()
            for prefix in previous_z:
                current[f'{prefix}_crossing'] = current[f'{prefix}_z'].gt(c.entry_zscore) & lagged[prefix].le(c.entry_zscore)
                current[f'{prefix}_short_crossing'] = current[f'{prefix}_z'].lt(-c.entry_zscore) & lagged[prefix].ge(-c.entry_zscore)
            if c.strategy_mode == 'revision_signal':
                current = raw_signal_triggers(current, c)
            next_prior.append(entry_candidates(current, c))
        prior = next_prior
        previous_z = {prefix:dict(zip(signal.permno,signal[f'{prefix}_z'])) for prefix in previous_z}
        if signal_day is not None and signal_day[0]==date:
            signal_day = next(signal_iterator,None)
        if market_day is not None and market_day[0]==date:
            market_day = next(market_iterator,None)
        if index % 252 == 0:
            progress(f'Replaying {len(scenarios)} sensitivity scenarios: {date.date()}')
    if baseline_output is not None:
        destination = Path(baseline_output)
        destination.mkdir(parents=True, exist_ok=True)
        parts = [book.frames() for book in books[0]]
        frames = {name: pd.concat([part[name] for part in parts], ignore_index=True) for name in parts[0]}
        frames['summary'] = summarize(frames['daily_results'], config)
        for name, frame in frames.items():
            frame.to_parquet(destination / f'{name}.parquet', index=False)
        frames['summary'].to_csv(destination / 'summary.csv', index=False)
    rows = []
    for (kind,threshold,cost),pair in zip(scenarios,books):
        daily = pd.concat([pd.DataFrame(book.daily) for book in pair],ignore_index=True)
        stats = performance_table(daily).loc['analyst_sharpe']
        rows.append(dict(scenario=kind,threshold=threshold,cost_bps=cost,total_return=stats.total_return,
            cagr_252=stats.cagr_252,sharpe_zero_rf=stats.sharpe_zero_rf,max_drawdown=stats.max_drawdown,
            entries=sum(trade['action']=='entry' for trade in pair[0].trades)))
    return pd.DataFrame(rows)


def make_book(strategy, prefix, calendar, config):
    if config.portfolio_mode == 'long_short':
        from .long_short import LongShortBook
        return LongShortBook(strategy, prefix, calendar, config)
    return PortfolioBook(strategy, prefix, calendar, config)


def entry_candidates(signal, config):
    crossing = signal.weighted_crossing | signal.unweighted_crossing
    if config.portfolio_mode == 'long_short':
        crossing = crossing | signal.weighted_short_crossing | signal.unweighted_short_crossing
    return signal[signal.eligible & crossing].copy()
