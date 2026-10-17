"""Point-in-time identifier linking and recommendation signal construction."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def normalize_cusip(values: pd.Series) -> pd.Series:
    """Normalize identifiers to uppercase eight-character CUSIPs."""

    return values.astype("string").str.strip().str.upper().str[:8]


def link_recommendations_to_crsp(
    recommendations: pd.DataFrame,
    crsp_names: pd.DataFrame,
) -> pd.DataFrame:
    """Link IBES rows to PERMNO using CUSIP and historical name-date validity.

    Ambiguous links are removed instead of selecting a match using future data.
    This assumes the downloaded IBES CUSIP is historically valid; users should
    audit a sample against the WRDS IBES-CRSP linking resource when available.
    """

    rec = recommendations.copy()
    names = crsp_names.copy()
    rec["statpers"] = pd.to_datetime(rec["statpers"])
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"])
    rec["cusip8"] = normalize_cusip(rec["cusip"])
    names["cusip8"] = normalize_cusip(names["ncusip"])

    rec = rec.reset_index(drop=True)
    rec["_rec_id"] = np.arange(len(rec))
    candidates = rec.merge(
        names[["permno", "cusip8", "namedt", "nameendt"]],
        on="cusip8",
        how="left",
        validate="many_to_many",
    )
    valid = candidates[
        candidates["statpers"].between(candidates["namedt"], candidates["nameendt"])
    ].copy()

    counts = valid.groupby("_rec_id")["permno"].nunique()
    ambiguous_ids = counts[counts > 1].index
    if len(ambiguous_ids):
        warnings.warn(
            f"Dropping {len(ambiguous_ids)} recommendation rows with ambiguous PERMNO links.",
            stacklevel=2,
        )
        valid = valid[~valid["_rec_id"].isin(ambiguous_ids)]

    valid = valid.sort_values(["_rec_id", "namedt"]).drop_duplicates("_rec_id", keep="last")
    return valid.drop(columns=["_rec_id", "namedt", "nameendt"])


def make_recommendation_signals(
    linked_recommendations: pd.DataFrame,
    publication_lag_days: int = 1,
) -> pd.DataFrame:
    """Create consensus and revisions from unambiguous historical snapshots.

    Exact repeated rows are collapsed. Conflicting security/date snapshots are
    excluded with a warning; revisions use the previous retained snapshot.
    """

    if publication_lag_days < 0:
        raise ValueError("publication_lag_days must be nonnegative")

    frame = linked_recommendations.copy()
    frame["statpers"] = pd.to_datetime(frame["statpers"])
    frame["meanrec"] = pd.to_numeric(frame["meanrec"], errors="coerce")
    if "numrec" in frame:
        frame["numrec"] = pd.to_numeric(frame["numrec"], errors="coerce")
    frame["available_date"] = frame["statpers"] + pd.to_timedelta(
        publication_lag_days, unit="D"
    )
    frame = frame.drop_duplicates()
    duplicates = frame.duplicated(["permno", "statpers"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, ["permno", "statpers"]].drop_duplicates()
        warnings.warn(
            f"Dropping {int(duplicates.sum())} ambiguous recommendation rows across "
            f"{len(keys)} PERMNO/statpers pairs. Multiple IBES records map to the "
            "same security/date; revisions use the previous retained snapshot. "
            f"Examples: {keys.head().to_dict('records')}",
            stacklevel=2,
        )
        frame = frame.loc[~duplicates].copy()
    frame = frame.sort_values(["permno", "statpers"])
    frame["consensus_signal"] = -frame["meanrec"]
    prior_meanrec = frame.groupby("permno", sort=False)["meanrec"].shift(1)
    frame["revision_signal"] = prior_meanrec - frame["meanrec"]
    return frame


def attach_latest_available_signal(
    formation_rows: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """As-of join the latest signal available on each formation date by PERMNO."""

    left = formation_rows.copy()
    right = signals.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["available_date"] = pd.to_datetime(right["available_date"])

    pieces: list[pd.DataFrame] = []
    signal_columns = [c for c in right.columns if c != "permno"]
    for permno, group in left.groupby("permno", sort=False):
        history = right[right["permno"] == permno].sort_values("available_date")
        group = group.sort_values("date")
        if history.empty:
            for column in signal_columns:
                if column not in group:
                    group[column] = np.nan
            pieces.append(group)
            continue
        pieces.append(
            pd.merge_asof(
                group,
                history.drop(columns="permno"),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )

    if not pieces:
        return left.assign(available_date=pd.NaT)
    return pd.concat(pieces, ignore_index=True)
