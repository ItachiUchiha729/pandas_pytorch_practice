import json
import pandas as pd
import pytest
from src import download_analyst_market as module


class FakeWRDS:
    def __init__(self, duplicate=False):
        self.closed = False
        self.duplicate = duplicate

    def describe_table(self, library, table):
        fields = {'dsf': ['permno','date','ret','prc'],
                  'dsenames': ['permno','namedt','nameendt','shrcd','exchcd'],
                  'dsedelist': ['permno','dlstdt','dlret'], 'dsi': ['date','vwretd']}
        return pd.DataFrame({'name': fields[table]})

    def raw_sql(self, query, params, date_cols):
        dates = pd.bdate_range(params['start'], params['end'])
        if 'vwretd' in query:
            return pd.DataFrame({'date': dates, 'ret': 0.})
        assert 'full outer join delists' in query
        frame = pd.DataFrame({'date': dates, 'permno': 1, 'ret': .01, 'prc': 10.,
                              'dlret': float('nan'), 'shrcd': 10, 'exchcd': 1, 'is_terminal': False})
        return pd.concat([frame, frame.iloc[:1]]) if self.duplicate else frame

    def close(self):
        self.closed = True


def test_daily_download_manifest_and_failure(monkeypatch, tmp_path):
    db = FakeWRDS()
    monkeypatch.setattr(module, 'connect_wrds', lambda: db)
    destination = module.download('2020-01-01', '2020-01-31', tmp_path)
    manifest = json.loads((destination/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['files']) == 1 and db.closed
    assert (destination/'benchmark.parquet').exists()
    bad = FakeWRDS(duplicate=True)
    monkeypatch.setattr(module, 'connect_wrds', lambda: bad)
    with pytest.raises(ValueError, match='Duplicate'):
        module.download('2020-01-01', '2020-01-31', tmp_path)
    manifests = [json.loads(p.read_text()) for p in tmp_path.glob('*/manifest.json')]
    assert sorted(m['complete'] for m in manifests) == [False, True]
    assert bad.closed


def test_post_legacy_dates_rejected_before_connection(monkeypatch, tmp_path):
    def unexpected_connection():
        raise AssertionError('Should not authenticate')
    monkeypatch.setattr(module, 'connect_wrds', unexpected_connection)
    with pytest.raises(ValueError, match='2024-12-31'):
        module.download('2020-01-01', '2025-01-01', tmp_path)
