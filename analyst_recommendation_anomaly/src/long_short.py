"""Optional signed total-return exposure model; the live default stays long-only.

Shorts are research total-return liabilities, not a share-level borrow ledger.
No locates, recalls, rebates, or broker margin liquidation are modeled. New
entries cannot increase gross exposure beyond NAV after entry costs. Market
moves may breach that limit; existing positions are not forcibly resized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .revision_zscore import _verified_halt, _halt_event, HALT_COLUMNS, _off_exchange_proxy, _off_exchange_event

from .revision_zscore import _held_return, _can_execute, _check_execution, entry_strength_column
from .streaming_strategy import PortfolioBook


class LongShortBook(PortfolioBook):
    def step(self, index, quotes, signals, benchmark_return):
        date, cutoff = self.calendar[index], self.calendar[index-1]
        config, strategy, prefix = self.config, self.strategy, self.prefix
        rate = config.cost_bps / 10000
        quotes = quotes.set_index('permno')
        before = self.cash + sum(self.pos.values())
        gross_before = self.gross_cash + sum(self.gross_pos.values())
        held_before = set(self.pos)
        cost, traded, borrow = 0., 0., 0.
        for permno in list(self.pos):
            info = self.metadata[permno]
            fee = max(-self.pos[permno], 0.) * config.short_borrow_bps_annual / 10000 / 252
            self.cash -= fee
            borrow += fee
            cost += fee
            info['borrow_cost_dollars'] += fee
            ret, proxy = _held_return(quotes, permno, date, strategy, config)
            self.pos[permno] *= 1 + ret
            self.gross_pos[permno] *= 1 + ret
            if _verified_halt(quotes, permno, date):
                self.valuation_events.append(_halt_event(strategy, permno, date))
                continue
            off_exchange = proxy and _off_exchange_proxy(quotes, permno, config)
            if off_exchange:
                self.valuation_events.append(_off_exchange_event(strategy, permno, date))
            terminal = bool(quotes.loc[permno, 'is_terminal']) or off_exchange
            if terminal or index >= info['exit_index']:
                if not terminal:
                    _check_execution(quotes, permno, date)
                value = self.pos.pop(permno)
                fee = 0. if terminal and not proxy else abs(value) * rate
                self.cash += value - fee
                self.gross_cash += self.gross_pos.pop(permno)
                cost += fee
                traded += 0. if terminal and not proxy else abs(value)
                self.metadata.pop(permno)
                reason = 'off_exchange_close_proxy' if off_exchange else 'terminal_close_proxy' if proxy else 'terminal' if terminal else ('holding_period_after_halt' if index > info['exit_index'] else 'holding_period')
                self.completed.append(dict(strategy=strategy, permno=permno, **info,
                    exit_date=date, exit_reason=reason, held_sessions=index-info['entry_index']))
                self.trades.append(dict(strategy=strategy, date=date, permno=permno,
                    action='exit', side=info['side'], reason=reason, value=abs(value), cost_dollars=fee))

        nav = self.cash + sum(self.pos.values())
        gross_nav = self.gross_cash + sum(self.gross_pos.values())
        if not np.isfinite(nav) or nav <= 0 or not np.isfinite(gross_nav) or gross_nav <= 0:
            raise ValueError(f'Portfolio NAV exhausted or invalid on {date.date()}')
        # Short-sale proceeds cannot be recycled into additional gross leverage.
        budget = max(0., nav - sum(abs(v) for v in self.pos.values()))
        gross_budget = max(0., gross_nav - sum(abs(v) for v in self.gross_pos.values()))
        selected = signals[signals.eligible &
            (signals[f'{prefix}_crossing'] | signals[f'{prefix}_short_crossing'])].copy()
        if (selected[f'{prefix}_crossing'] & selected[f'{prefix}_short_crossing']).any():
            raise ValueError('A stock cannot cross both entry thresholds on the same session')
        selected['strength'] = selected[entry_strength_column(prefix, config)].abs()
        selected = selected.sort_values(['strength', 'permno'], ascending=[False, True])
        accepted = []
        for row in selected.itertuples(index=False):
            permno = int(row.permno)
            side = 'long' if getattr(row, f'{prefix}_crossing') else 'short'
            reason = ('already_held' if permno in held_before else 'capacity'
                      if len(self.pos)+len(accepted) >= config.top_n else 'entry')
            if reason == 'entry' and budget <= 1e-8:
                reason = 'gross_exposure_limit'
            if reason == 'entry' and not _can_execute(quotes, permno):
                reason = 'untradable_at_execution'
            self.orders.append(dict(strategy=strategy, cutoff=cutoff, execution_date=date,
                permno=permno, zscore=getattr(row, f'{prefix}_z'), entry_signal=getattr(row, 'signal' if prefix=='weighted' else 'unweighted_score'), side=side, status=reason))
            if reason == 'entry':
                accepted.append((permno, side))
        if accepted:
            amount = min(nav/config.top_n, budget/(len(accepted)*(1+rate)))
            gross_amount = min(gross_nav/config.top_n, gross_budget/len(accepted))
            for permno, side in accepted:
                direction = 1 if side == 'long' else -1
                fee = amount * rate
                self.cash -= direction*amount + fee
                self.gross_cash -= direction*gross_amount
                self.pos[permno] = direction*amount
                self.gross_pos[permno] = direction*gross_amount
                end = index + config.holding_sessions
                self.metadata[permno] = dict(side=side, entry_date=date, entry_index=index,
                    exit_index=end, planned_exit_date=self.calendar[end] if end<len(self.calendar) else pd.NaT,
                    borrow_cost_dollars=0.)
                cost += fee
                traded += amount
                self.trades.append(dict(strategy=strategy, date=date, permno=permno, action='entry',
                    side=side, reason='raw_signal_threshold' if config.strategy_mode=='revision_signal' else 'zscore_crossing', value=amount, cost_dollars=fee))

        nav = self.cash + sum(self.pos.values())
        gross_nav = self.gross_cash + sum(self.gross_pos.values())
        if self.cash < -1e-7 or self.gross_cash < -1e-7:
            raise ValueError(f'Cash financing required on {date.date()}; model does not permit borrowing cash')
        if nav <= 0 or gross_nav <= 0:
            raise ValueError(f'Portfolio NAV exhausted on {date.date()}')
        self.cash, self.gross_cash = max(0., self.cash), max(0., self.gross_cash)
        long_value = sum(max(v, 0.) for v in self.pos.values())
        short_value = sum(max(-v, 0.) for v in self.pos.values())
        self.benchmark_nav *= 1 + benchmark_return
        self.daily.append(dict(strategy=strategy, date=date, net_return=nav/before-1,
            gross_return=gross_nav/gross_before-1, nav=nav, gross_nav=gross_nav,
            benchmark_return=benchmark_return, benchmark_nav=self.benchmark_nav,
            cost_dollars=cost, borrow_cost_dollars=borrow, turnover=traded/(nav+cost),
            cash=self.cash, n_holdings=len(self.pos), n_long=sum(v['side']=='long' for v in self.metadata.values()),
            n_short=sum(v['side']=='short' for v in self.metadata.values()),
            long_exposure=long_value/nav, short_exposure=short_value/nav,
            gross_exposure=(long_value+short_value)/nav, net_exposure=(long_value-short_value)/nav,
            free_collateral=self.cash-2*short_value))
        self.holdings.extend(dict(strategy=strategy, date=date, permno=k, value=v, weight=v/nav,
            side=self.metadata[k]['side'], entry_date=self.metadata[k]['entry_date'],
            planned_exit_date=self.metadata[k]['planned_exit_date']) for k,v in self.pos.items())

    def frames(self):
        frames = super().frames()
        for name, records, extra in [
            ('orders', self.orders, ['side']), ('trades', self.trades, ['side']),
            ('holdings', self.holdings, ['side']),
            ('completed_positions', self.completed, ['side', 'borrow_cost_dollars'])]:
            frames[name] = pd.DataFrame(records, columns=list(frames[name].columns)+extra)
        records = [dict(strategy=self.strategy, permno=k, value=v, **self.metadata[k]) for k,v in self.pos.items()]
        frames['open_positions'] = pd.DataFrame(records,
            columns=list(frames['open_positions'].columns)+['side', 'borrow_cost_dollars'])
        return frames


def simulate_long_short(signals, stocks, bench, start, end, config):
    calendar = pd.DatetimeIndex(bench.date)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start < calendar[0] or end > calendar[-1]:
        raise ValueError('Requested date range must lie inside the benchmark calendar')
    first = calendar.searchsorted(start)
    if first+1 >= len(calendar) or calendar[first+1] > end:
        raise ValueError('Date range contains no next-session execution')
    books = [LongShortBook('analyst_sharpe','weighted',calendar,config),
             LongShortBook('unweighted','unweighted',calendar,config)]
    market = dict(tuple(stocks.groupby('date')))
    days = dict(tuple(signals.groupby('date')))
    for index in range(first+1, len(calendar)):
        if calendar[index] > end:
            break
        for book in books:
            book.step(index, market.get(calendar[index], stocks.iloc[:0]),
                days.get(calendar[index-1], signals.iloc[:0]), bench.ret.iloc[index])
    results = [book.frames() for book in books]
    return {key:pd.concat([r[key] for r in results],ignore_index=True) for key in results[0]}
