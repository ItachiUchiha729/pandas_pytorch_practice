import numpy as np
import pandas as pd

from src.backtest import (
    build_formation_panel,
    combine_crsp_returns,
    run_long_short_backtest,
)
from src.config import BacktestConfig
from src.signals import attach_latest_available_signal, make_recommendation_signals


def test_signal_is_not_used_before_available_date():
    formation = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
        }
    )
    signals = pd.DataFrame(
        {
            "permno": [1],
            "available_date": pd.to_datetime(["2020-02-01"]),
            "consensus_signal": [-1.0],
        }
    )
    joined = attach_latest_available_signal(formation, signals)
    assert np.isnan(joined.loc[joined["date"] == "2020-01-31", "consensus_signal"]).all()
    assert joined.loc[joined["date"] == "2020-02-29", "consensus_signal"].iloc[0] == -1.0


def test_month_end_snapshot_gets_one_day_lag():
    linked = pd.DataFrame(
        {"permno": [1], "statpers": ["2020-01-31"], "meanrec": [1.5]}
    )
    signal = make_recommendation_signals(linked, publication_lag_days=1)
    assert signal["available_date"].iloc[0] == pd.Timestamp("2020-02-01")


def test_revision_uses_only_previous_snapshot():
    linked = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "statpers": ["2020-01-15", "2020-02-15", "2020-03-15"],
            "meanrec": [3.0, 2.0, 2.5],
        }
    )
    signal = make_recommendation_signals(linked)
    assert np.isnan(signal["revision_signal"].iloc[0])
    assert signal["revision_signal"].iloc[1] == 1.0
    assert signal["revision_signal"].iloc[2] == -0.5


def test_formation_month_uses_next_calendar_month_return():
    crsp = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": ["2020-01-31", "2020-02-29"],
            "ret": [0.99, 0.05],
            "dlret": [np.nan, np.nan],
            "prc": [10.0, 10.5],
            "shrout": [100.0, 100.0],
            "shrcd": [10, 10],
            "exchcd": [1, 1],
        }
    )
    signals = pd.DataFrame(
        {
            "permno": [1],
            "available_date": pd.to_datetime(["2020-01-01"]),
            "numrec": [3],
            "consensus_signal": [-1.0],
        }
    )
    panel = build_formation_panel(crsp, signals, BacktestConfig(min_price=0.0))
    january = panel.loc[panel["date"] == "2020-01-31"].iloc[0]
    assert np.isclose(january["forward_return"], 0.05)
    assert not np.isclose(january["forward_return"], 0.99)


def test_delisting_return_is_compounded():
    result = combine_crsp_returns(pd.Series([-0.20]), pd.Series([-0.50]))
    assert np.isclose(result.iloc[0], -0.60)


def test_future_recommendation_does_not_change_past_signal():
    base = pd.DataFrame(
        {"permno": [1, 1], "statpers": ["2020-01-15", "2020-02-15"], "meanrec": [3.0, 2.0]}
    )
    changed = base.copy()
    changed.loc[1, "meanrec"] = 5.0
    base_signal = make_recommendation_signals(base)
    changed_signal = make_recommendation_signals(changed)
    assert base_signal["consensus_signal"].iloc[0] == changed_signal["consensus_signal"].iloc[0]


def test_missing_held_outcome_is_not_treated_as_zero():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 4),
            "permno": [1, 2, 3, 4],
            "consensus_signal": [-4.0, -3.0, -2.0, -1.0],
            "forward_return": [0.01, 0.02, 0.03, np.nan],
        }
    )
    config = BacktestConfig(signal="consensus_signal", quantile=0.25)
    results, holdings = run_long_short_backtest(panel, config)
    assert holdings.loc[holdings["permno"] == 4, "weight"].iloc[0] > 0
    assert results["held_return_missing"].iloc[0]
    assert np.isnan(results["gross_return"].iloc[0])
