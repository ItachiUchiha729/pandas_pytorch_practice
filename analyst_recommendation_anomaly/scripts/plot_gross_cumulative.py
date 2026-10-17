"""Plot observed-month gross compounding; missing outcomes are not recovered."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout='constrained')
exports = []
for ax, name, color in zip(axes, ['consensus', 'revision'], ['#2364aa', '#b45529']):
    frame = pd.read_csv(OUT / f'{name}_monthly_results.csv', parse_dates=['date']).sort_values('date')
    gross = frame['gross_return']
    observed_cumulative = ((1 + gross).cumprod(skipna=True) - 1) * 100
    dates = frame['date'] + pd.offsets.MonthEnd(1)
    observed = gross.notna()
    ax.plot(dates[observed], observed_cumulative[observed], color=color, linewidth=1.8,
            label='Gross cumulative return')
    ax.axhline(0, color='#555555', linewidth=.8)
    available = observed_cumulative.dropna()
    endpoint = f'{available.iloc[-1]:+.1f}%' if len(available) else 'unavailable'
    ax.set_title(f'{name.title()} | {endpoint}',
                 loc='left', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative return (%)')
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', fontsize=8)
    exports.append(pd.DataFrame({'strategy': name, 'return_month': dates,
                                 'gross_return': gross,
                                 'observed_month_cumulative_return': observed_cumulative / 100}))
axes[-1].set_xlabel('Return month')
fig.suptitle('Cumulative returns before trading costs', fontsize=17, fontweight='bold')
fig.supxlabel('Compounds available monthly returns only; lines connect observed points.', fontsize=9)
fig.savefig(OUT / 'gross_cumulative_returns.png', dpi=180)
fig.savefig(OUT / 'gross_cumulative_returns.pdf')
pd.concat(exports, ignore_index=True).to_csv(OUT / 'gross_cumulative_returns.csv', index=False)
print('Saved gross_cumulative_returns.png, .pdf, and .csv in outputs/')
