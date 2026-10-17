"""Run the daily analyst Sharpe strategy; --demo requires no WRDS account."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .analyst_sharpe import AnalystSharpeConfig, load_ibes_extract, run_strategy
from .config import RAW_DIR, OUTPUT_DIR


def demo_inputs():
    """Artificial data exercises full warmup and trading; no empirical claims."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2017-01-02', periods=560)
    benchmark = pd.DataFrame({'date': dates, 'ret': rng.normal(.00015, .004, len(dates))})
    daily, events = [], []
    for stock in range(1, 25):
        ret = benchmark.ret.to_numpy() + .00035 + rng.normal(0, .008, len(dates))
        daily.append(pd.DataFrame({'date': dates, 'permno': stock, 'ret_total': ret,
                                   'prc': 30 * np.cumprod(1 + ret), 'shrcd': 10,
                                   'exchcd': 1, 'is_terminal': False}))
        for analyst in range(3):
            for i in range(0, len(dates) - 2, 12):
                events.append({'permno': stock, 'analyst_id': str(analyst),
                               'recommendation_date': dates[i],
                               'available_date': dates[i] + pd.Timedelta(1, unit="D").as_unit("ns"),
                               'rating': int(rng.choice([-2, -1, 0, 1, 2], p=[.02, .03, .10, .40, .45]))})
    return pd.DataFrame(events), pd.concat(daily, ignore_index=True), benchmark, dates[160], dates[-1]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(Path(path).resolve()), 'sha256': digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--extract', type=Path, help='One completed raw IBES detail extract')
    parser.add_argument('--events', type=Path, help='Alternatively, normalized linked event Parquet')
    parser.add_argument('--market-dir', type=Path, help='Daily downloader directory including manifest')
    parser.add_argument('--names', type=Path, default=RAW_DIR / 'crsp_names.parquet')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--config', type=Path, help='JSON overrides for AnalystSharpeConfig')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'analyst_sharpe')
    args = parser.parse_args(argv)
    config = AnalystSharpeConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    if args.demo and (args.extract or args.events or args.market_dir):
        parser.error('--demo cannot be combined with real input paths')
    if not args.demo and (bool(args.extract) == bool(args.events) or not args.market_dir or not args.start or not args.end):
        parser.error('Supply exactly one of --extract/--events, plus --market-dir, --start and --end')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix='synthetic_' if args.demo else 'research_', dir=args.output_dir))
    manifest = {'complete': False, 'synthetic': args.demo,
                'engine': 'reference' if args.demo else 'streaming',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'start': args.start, 'end': args.end, 'config': asdict(config), 'inputs': [], 'input_diagnostics': {},
                'code': [fingerprint(Path(__file__)), fingerprint(Path(__file__).with_name('analyst_sharpe.py')),
                         fingerprint(Path(__file__).with_name('revision_zscore.py')),
                         fingerprint(Path(__file__).with_name('streaming_strategy.py')),
                         fingerprint(Path(__file__).with_name('long_short.py'))],
                'notes': 'No terminal liquidation; cash and risk-free return are zero. '
                         'Strict missing-held-return and execution-price validation.'}
    def save_manifest():
        temporary = destination / 'manifest.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(destination / 'manifest.json')
    def progress(message):
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / (1024**2 if sys.platform == 'darwin' else 1024)
        line = f'{datetime.now(timezone.utc).isoformat()} | {message} | peak RSS {rss_mb:.0f} MiB'
        manifest['stage'] = message
        manifest['peak_rss_mib'] = round(rss_mb, 1)
        save_manifest()
        with (destination / 'progress.log').open('a') as stream:
            stream.write(line + '\n')
        print(line, flush=True)
    save_manifest()
    print(f'Artifacts: {destination.resolve()}', flush=True)
    try:
        if args.demo:
            progress('Generating synthetic demonstration')
            events, daily, benchmark, default_start, default_end = demo_inputs()
            start, end = args.start or default_start, args.end or default_end
        else:
            market_manifest = args.market_dir / 'manifest.json'
            if not market_manifest.exists():
                raise ValueError('Daily inputs missing. Run python -m src.download_analyst_market first.')
            metadata = json.loads(market_manifest.read_text())
            if not metadata.get('complete'):
                raise ValueError('Daily market extract is incomplete')
            benchmark_path = args.market_dir / 'benchmark.parquet'
            manifest['inputs'] = [fingerprint(market_manifest), fingerprint(benchmark_path)]
            manifest['inputs'] += [fingerprint(args.market_dir / f['file']) for f in metadata['files']]
            progress('Loading individual analyst events; daily market panel stays on disk')
            if args.events:
                events = pd.read_parquet(args.events)
                manifest['inputs'].append(fingerprint(args.events))
            else:
                events, diagnostics = load_ibes_extract(args.extract, pd.read_parquet(args.names), config)
                manifest['input_diagnostics'] = diagnostics
                manifest['inputs'] += [fingerprint(args.extract / 'manifest.json'), fingerprint(args.names)]
                im = json.loads((args.extract / 'manifest.json').read_text())
                manifest['inputs'] += [fingerprint(args.extract / item['file']) for item in im['files']]
            start, end = args.start, args.end
        manifest['start'], manifest['end'] = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
        if args.demo:
            progress('Running synthetic strategy')
            artifacts = run_strategy(events, daily, benchmark, start, end, config)
            for name, frame in artifacts.items():
                frame.to_parquet(destination / f'{name}.parquet', index=False)
            summary = artifacts['summary']
            summary.to_csv(destination / 'summary.csv', index=False)
            manifest['rows'] = {name: len(frame) for name, frame in artifacts.items()}
            manifest['outcome_status_counts'] = artifacts['outcomes'].outcome_status.value_counts().to_dict()
        else:
            from .streaming_strategy import run_streaming
            summary, rows, counts = run_streaming(events, args.market_dir, start, end, config, destination, progress)
            manifest['rows'], manifest['outcome_status_counts'] = rows, counts
        manifest['complete'] = True
        progress('Complete')
    except Exception as error:
        manifest['error'] = f'{type(error).__name__}: {error}'
        progress('Failed: ' + manifest['error'])
        raise
    print('SYNTHETIC DEMONSTRATION — NOT EMPIRICAL PERFORMANCE' if args.demo else 'RESEARCH BACKTEST — assumptions recorded in manifest')
    print(summary.to_string(index=False))
    print(f'Artifacts: {destination.resolve()}', flush=True)


if __name__ == '__main__':
    main()
