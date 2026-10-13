"""Download point-in-time IBES recommendations and CRSP monthly returns.

Run from the project root:
    python -m src.download_data --start 2000-01-01 --end 2025-12-31

The module imports ``wrds`` lazily, so all other project code and synthetic
tests work without WRDS credentials or the proprietary data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, ensure_directories


IBES_RECOMMENDATIONS_SQL = """
    select ticker, cusip, oftic, statpers,
           meanrec, medrec, numrec
    from ibes.recdsum
    where statpers between %(start)s and %(end)s
    order by statpers, ticker
"""

CRSP_MONTHLY_SQL = """
    select m.permno, m.date, m.ret, m.prc, m.shrout,
           n.ncusip, n.ticker, n.shrcd, n.exchcd,
           d.dlret
    from crsp.msf as m
    left join crsp.msenames as n
      on m.permno = n.permno
     and n.namedt <= m.date
     and m.date <= n.nameendt
    left join crsp.msedelist as d
      on m.permno = d.permno
     and date_trunc('month', d.dlstdt) = date_trunc('month', m.date)
    where m.date between %(start)s and %(end)s
    order by m.date, m.permno
"""

CRSP_NAMES_SQL = """
    select distinct permno, ncusip, ticker, namedt, nameendt, shrcd, exchcd
    from crsp.msenames
    where nameendt >= %(start)s and namedt <= %(end)s
    order by permno, namedt
"""


def _iso_date(value: str) -> str:
    """Validate and normalize a command-line ISO date."""

    return pd.Timestamp(value).date().isoformat()


def connect_wrds():
    """Open a WRDS connection using standard WRDS authentication."""

    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - depends on local setup
        raise RuntimeError("Install the WRDS client with: pip install wrds") from exc

    username = os.environ.get("WRDS_USERNAME")
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def list_library(connection, library: str) -> None:
    """Print tables visible in a WRDS library to diagnose schema differences."""

    tables = connection.list_tables(library=library)
    print("\n".join(sorted(tables)))


def download_all(
    start: str,
    end: str,
    output_dir: Path = RAW_DIR,
    force: bool = False,
) -> None:
    """Download missing raw inputs and cache them as Parquet files."""

    start = _iso_date(start)
    end = _iso_date(end)
    if start > end:
        raise ValueError("start must not be after end")

    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendation_params = {"start": start, "end": end}
    # A portfolio formed at the final requested month-end needs the following
    # calendar month's CRSP return. This extension uses no future information
    # at formation; it only supplies the subsequent realized outcome.
    crsp_end = (pd.Timestamp(end) + pd.offsets.MonthEnd(1)).date().isoformat()
    crsp_params = {"start": start, "end": crsp_end}
    jobs = [
        (
            "ibes_recommendations.parquet",
            IBES_RECOMMENDATIONS_SQL,
            recommendation_params,
            ["statpers"],
        ),
        ("crsp_monthly.parquet", CRSP_MONTHLY_SQL, crsp_params, ["date"]),
        (
            "crsp_names.parquet",
            CRSP_NAMES_SQL,
            crsp_params,
            ["namedt", "nameendt"],
        ),
    ]
    pending = [job for job in jobs if force or not (output_dir / job[0]).exists()]
    for filename, _, _, _ in jobs:
        path = output_dir / filename
        if path.exists() and not force:
            print(f"Using cached file: {path}")
    if not pending:
        print("All required Parquet files are already cached; nothing to download.")
        return

    db = connect_wrds()
    try:
        for filename, query, query_params, date_cols in pending:
            print(f"Downloading {filename} ...")
            frame = db.raw_sql(query, params=query_params, date_cols=date_cols)
            path = output_dir / filename
            frame.to_parquet(path, index=False)
            print(f"Wrote {len(frame):,} rows to {path}")
            del frame
    finally:
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().date().isoformat())
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--list-library",
        choices=("ibes", "crsp"),
        help="List visible tables instead of downloading data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing Parquet cache files instead of reusing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_library:
        db = connect_wrds()
        try:
            list_library(db, args.list_library)
        finally:
            db.close()
        return
    download_all(args.start, args.end, args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
