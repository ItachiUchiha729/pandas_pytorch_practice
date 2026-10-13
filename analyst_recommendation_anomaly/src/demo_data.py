"""Reproducible synthetic data for exercising the notebook without WRDS."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_data(
    n_stocks: int = 100,
    n_months: int = 60,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return synthetic IBES recommendations, CRSP names, and CRSP returns.

    The simulated recommendation level has modest predictive content solely so
    the complete research pipeline produces an interpretable demonstration.
    These data are not empirical evidence for the anomaly.
    """

    rng = np.random.default_rng(seed)
    # The extra CRSP month supplies the return earned by the final formation.
    month_ends = pd.date_range("2018-01-31", periods=n_months + 1, freq="ME")
    permnos = np.arange(10_000, 10_000 + n_stocks)
    cusips = [f"{i:08d}" for i in range(n_stocks)]
    names = pd.DataFrame(
        {
            "permno": permnos,
            "ncusip": cusips,
            "ticker": [f"S{i:03d}" for i in range(n_stocks)],
            "namedt": pd.Timestamp("1900-01-01"),
            "nameendt": pd.Timestamp("2099-12-31"),
            "shrcd": 10,
            "exchcd": 1,
        }
    )

    latent = rng.normal(size=(n_months + 1, n_stocks))
    meanrec = np.clip(
        3.0 - 0.7 * latent[:n_months] + rng.normal(0, 0.35, (n_months, n_stocks)),
        1,
        5,
    )
    rec_rows: list[dict] = []
    crsp_rows: list[dict] = []
    for t, date in enumerate(month_ends):
        statpers = date - pd.Timedelta(days=10)
        for j, permno in enumerate(permnos):
            if t < n_months:
                rec_rows.append(
                    {
                        "ticker": f"S{j:03d}",
                        "cusip": cusips[j],
                        "oftic": f"S{j:03d}",
                        "statpers": statpers,
                        "meanrec": meanrec[t, j],
                        "medrec": np.round(meanrec[t, j]),
                        "numrec": int(rng.integers(2, 12)),
                    }
                )
            predictive_component = 0.008 * latent[t - 1, j] if t > 0 else 0.0
            monthly_return = 0.005 + predictive_component + rng.normal(0, 0.07)
            crsp_rows.append(
                {
                    "permno": permno,
                    "date": date,
                    "ret": monthly_return,
                    "dlret": np.nan,
                    "prc": 10.0 + abs(rng.normal(20, 10)),
                    "shrout": rng.uniform(10_000, 1_000_000),
                    "ncusip": cusips[j],
                    "ticker": f"S{j:03d}",
                    "shrcd": 10,
                    "exchcd": 1,
                }
            )
    return pd.DataFrame(rec_rows), names, pd.DataFrame(crsp_rows)
