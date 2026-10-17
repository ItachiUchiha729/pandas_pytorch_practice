"""Download individual IBES recommendations, preserving original vendor fields.

Run: python -m src.download_analyst_data --start 2000-01-01 --end 2025-12-31
Each extraction gets its own directory with annual Parquet files and a manifest.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import tempfile

import pandas as pd

from .config import RAW_DIR
from .download_data import connect_wrds, _iso_date


def identifier(value: str) -> str:
    if not re.fullmatch(r'[a-z_][a-z0-9_]*', value):
        raise ValueError(f'Invalid SQL identifier: {value!r}')
    return value


def date_windows(start: str, end: str):
    first, last = pd.Timestamp(_iso_date(start)), pd.Timestamp(_iso_date(end))
    if first > last:
        raise ValueError('start must not be after end')
    for year in range(first.year, last.year + 1):
        yield max(first, pd.Timestamp(year, 1, 1)).date().isoformat(), min(
            last, pd.Timestamp(year, 12, 31)
        ).date().isoformat()


def check_schema(schema: pd.DataFrame) -> list[str]:
    columns = set(schema['name'].str.lower())
    required = {'ticker', 'amaskcd', 'emaskcd', 'ireccd', 'anndats'}
    missing = required - columns
    if missing:
        raise ValueError(
            f'Expected analyst recommendation fields missing: {sorted(missing)}. '
            'Run --inspect and check the WRDS dictionary; no data was downloaded.'
        )
    # Preserve actual vendor timestamps; never substitute the announcement date
    # for the date the observation entered the database.
    return [name for name in ('anndats', 'actdats', 'revdats') if name in columns]


def download(start: str, end: str, output_dir: Path = RAW_DIR,
             library: str = 'ibes', table: str | None = None, inspect: bool = False) -> Path | None:
    library = identifier(library)
    if table is not None:
        table = identifier(table)
    windows = list(date_windows(start, end))  # Validate before authentication.
    db = connect_wrds()
    try:
        tables = db.list_tables(library=library)
        if table is None:
            table = next((name for name in ('recddet', 'recdet') if name in tables), 'recddet')
        if table not in tables:
            candidates = sorted(t for t in tables if 'rec' in t.lower())
            raise ValueError(
                f'{library}.{table} is not visible to your account. '
                f'Recommendation-related tables: {candidates}. '
                'Confirm Detail Recommendations access; use --table for its actual name.'
            )
        schema = db.describe_table(library=library, table=table)
        if inspect:
            print(schema.to_string(index=False))
            return None
        date_cols = check_schema(schema)
        if 'actdats' not in date_cols:
            print('Note: ACTDATS is absent. Verify availability timing before backtesting.')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        destination = Path(tempfile.mkdtemp(prefix=f'ibes_analyst_{stamp}_', dir=output_dir))
        manifest = {
            'source': f'{library}.{table}', 'extracted_at_utc': stamp,
            'start': windows[0][0], 'end': windows[-1][1],
            'filter_date': 'anndats', 'complete': False, 'files': [],
            'columns': schema['name'].tolist(),
            'notes': 'Raw records, no deduplication or rating transformations. '
                     'Do not combine masked analyst IDs across different extracts. '
                     'Pre-start recommendations are not included; download earlier history '
                     'if an initial rating is needed for revisions.',
        }
        manifest_path = destination / 'manifest.json'
        def save_manifest():
            temp = destination / 'manifest.json.tmp'
            temp.write_text(json.dumps(manifest, indent=2) + '\n')
            temp.replace(manifest_path)
        save_manifest()
        print(f'Writing extraction to {destination}', flush=True)
        for first, last in windows:
            print(f'Downloading announcements {first} through {last} ...', flush=True)
            frame = db.raw_sql(
                f'SELECT * FROM {library}.{table} '
                'WHERE anndats >= %(start)s AND anndats < %(stop)s',
                params={'start': first, 'stop': (date.fromisoformat(last) + timedelta(days=1)).isoformat()},
                date_cols=date_cols,
            )
            filename = f'recommendations_{first}_{last}.parquet'
            temp = destination / (filename + '.tmp')
            frame.to_parquet(temp, index=False)
            temp.replace(destination / filename)
            manifest['files'].append({'file': filename, 'rows': len(frame)})
            save_manifest()
            print(f'Wrote {len(frame):,} rows', flush=True)
        manifest['complete'] = True
        manifest['total_rows'] = sum(item['rows'] for item in manifest['files'])
        save_manifest()
        print(f'Finished: {manifest["total_rows"]:,} rows in {destination}')
        return destination
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2000-01-01')
    parser.add_argument('--end', default='2025-12-31')
    parser.add_argument('--library', default='ibes')
    parser.add_argument('--table', help='Default: detect recddet or recdet in the selected library.')
    parser.add_argument('--output-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--inspect', action='store_true', help='Show table fields without downloading.')
    args = parser.parse_args()
    download(**vars(args))


if __name__ == '__main__':
    main()
