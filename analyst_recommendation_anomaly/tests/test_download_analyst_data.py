import json
import pandas as pd
import pytest
from src import download_analyst_data as module


class FakeWRDS:
    def __init__(self, fail=False):
        self.closed = False
        self.calls = []
        self.fail = fail

    def list_tables(self, library):
        return ['recdet', 'recdsum']

    def describe_table(self, **kwargs):
        return pd.DataFrame({'name': ['ticker', 'amaskcd', 'emaskcd', 'ireccd', 'anndats', 'actdats']})

    def raw_sql(self, query, params, date_cols):
        self.calls.append(params)
        if self.fail and len(self.calls) == 2:
            raise RuntimeError('connection lost')
        # Preserve duplicate vendor records and masked IDs, including null IDs.
        return pd.DataFrame({'ticker': ['A','A'], 'amaskcd': ['001',None],
                             'emaskcd': ['002','002'], 'ireccd': [2,2],
                             'anndats': pd.to_datetime([params['start']]*2)})

    def close(self):
        self.closed = True


def test_download_preserves_records_and_bounds(tmp_path, monkeypatch):
    db = FakeWRDS()
    monkeypatch.setattr(module, 'connect_wrds', lambda: db)
    path = module.download('2020-12-31', '2021-01-02', tmp_path)
    manifest = json.loads((path/'manifest.json').read_text())
    assert manifest['complete'] and manifest['total_rows'] == 4
    assert db.calls == [{'start':'2020-12-31','stop':'2021-01-01'},
                        {'start':'2021-01-01','stop':'2021-01-03'}]
    frame = pd.read_parquet(path/manifest['files'][0]['file'])
    assert frame.amaskcd.iloc[0] == '001'
    assert pd.isna(frame.amaskcd.iloc[1])
    assert len(frame) == 2 and db.closed


def test_interrupted_extract_is_not_marked_complete(tmp_path, monkeypatch):
    db = FakeWRDS(fail=True)
    monkeypatch.setattr(module, 'connect_wrds', lambda: db)
    with pytest.raises(RuntimeError, match='connection lost'):
        module.download('2020-12-31', '2021-01-02', tmp_path)
    manifest = json.loads(next(tmp_path.glob('*/manifest.json')).read_text())
    assert manifest['complete'] is False
    assert len(manifest['files']) == 1 and db.closed


def test_inspect_writes_nothing(tmp_path, monkeypatch):
    db = FakeWRDS()
    monkeypatch.setattr(module, 'connect_wrds', lambda: db)
    module.download('2020-01-01','2020-12-31',tmp_path,inspect=True)
    assert not list(tmp_path.iterdir()) and not db.calls and db.closed


def test_bad_inputs_and_summary_schema_are_rejected():
    with pytest.raises(ValueError):
        list(module.date_windows('2021-01-01','2020-01-01'))
    with pytest.raises(ValueError):
        module.identifier('recdet; DROP TABLE anything')
    with pytest.raises(ValueError, match='missing'):
        module.check_schema(pd.DataFrame({'name':['ticker','meanrec','statpers']}))
