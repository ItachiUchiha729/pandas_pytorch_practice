"""Download legacy CRSP daily stocks and value-weighted market total returns.

Requires WRDS access to crsp.dsf, dsenames, dsedelist and dsi. Dates after
2024 require a separately validated CIZ adapter, not these legacy tables.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import pandas as pd
from .config import RAW_DIR
from .download_data import connect_wrds
from .download_analyst_data import date_windows

DAILY_SQL = """
with prices as (
 select permno, date, ret, prc from crsp.dsf
 where date between %(start)s and %(end)s
), delists as (
 select permno, dlstdt, dlret from crsp.dsedelist
 where dlstdt between %(start)s and %(end)s
), combined as (
 select coalesce(p.permno, d.permno) as permno,
        coalesce(p.date, d.dlstdt) as date,
        p.ret, p.prc, d.dlret, (d.permno is not null) as is_terminal
 from prices p full outer join delists d
 on p.permno = d.permno and p.date = d.dlstdt
)
select c.*, n.shrcd, n.exchcd
from combined c left join crsp.dsenames n
 on c.permno = n.permno and n.namedt <= c.date and c.date <= n.nameendt
order by c.date, c.permno
"""
BENCHMARK_SQL = """
select date, vwretd as ret from crsp.dsi
where date between %(start)s and %(end)s order by date
"""


def download(start, end, output_dir=RAW_DIR):
    windows = list(date_windows(start, end))
    if pd.Timestamp(end) > pd.Timestamp('2024-12-31'):
        raise ValueError('Legacy downloader ends at 2024-12-31. Use validated daily CIZ inputs for later dates.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db = connect_wrds()
    try:
        for table, required in {
            'dsf': {'permno', 'date', 'ret', 'prc'},
            'dsenames': {'permno', 'namedt', 'nameendt', 'shrcd', 'exchcd'},
            'dsedelist': {'permno', 'dlstdt', 'dlret'},
            'dsi': {'date', 'vwretd'},
        }.items():
            schema = db.describe_table(library='crsp', table=table)
            if not required.issubset(set(schema['name'].str.lower())):
                raise ValueError(f'Unexpected crsp.{table} schema; inspect WRDS tables')
        destination = Path(tempfile.mkdtemp(prefix='analyst_daily_', dir=output_dir))
        manifest = {'complete': False, 'start': start, 'end': end, 'files': [],
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'benchmark': 'crsp.dsi.vwretd (USD value-weighted market total return)',
                    'source': ['crsp.dsf', 'crsp.dsenames', 'crsp.dsedelist'],
                    'notes': 'Full outer stock/delisting join; ambiguous joins and off-calendar events fail validation.'}
        def save():
            temp = destination / 'manifest.json.tmp'
            temp.write_text(json.dumps(manifest, indent=2) + '\n')
            temp.replace(destination / 'manifest.json')
        save()
        benchmark = db.raw_sql(BENCHMARK_SQL, params={'start': start, 'end': end}, date_cols=['date'])
        if benchmark.empty or benchmark.date.duplicated().any() or benchmark.ret.isna().any():
            raise ValueError('Missing/duplicate benchmark observations')
        if pd.Timestamp(end) - pd.Timestamp(benchmark.date.max()) > pd.Timedelta(7, unit="D").as_unit("ns"):
            raise ValueError('Benchmark does not cover requested end date')
        benchmark.to_parquet(destination / 'benchmark.parquet', index=False)
        for first, last in windows:
            print(f'Downloading daily CRSP: {first} through {last}', flush=True)
            frame = db.raw_sql(DAILY_SQL, params={'start': first, 'end': last}, date_cols=['date'])
            if frame.duplicated(['permno', 'date']).any():
                raise ValueError('Duplicate daily security records; resolve historical joins')
            if not pd.to_datetime(frame.date).isin(pd.to_datetime(benchmark.date)).all():
                raise ValueError('Off-calendar delisting events; resolve event timing')
            name = f'daily_{first}_{last}.parquet'
            temp = destination / (name + '.tmp')
            frame.to_parquet(temp, index=False)
            temp.replace(destination / name)
            manifest['files'].append({'file': name, 'rows': len(frame)})
            save()
        manifest['complete'] = True
        save()
        print(f'Completed: {destination.resolve()}')
        return destination
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2000-01-01', help='Include analyst warmup history')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--output-dir', type=Path, default=RAW_DIR)
    download(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
