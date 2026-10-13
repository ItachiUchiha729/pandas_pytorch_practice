"""Paths and default research settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class BacktestConfig:
    """Conservative defaults for the monthly recommendation strategy."""

    signal: str = "consensus_signal"
    quantile: float = 0.10
    min_analysts: int = 2
    publication_lag_days: int = 1
    transaction_cost_bps: float = 10.0
    min_price: float = 5.0
    allowed_share_codes: tuple[int, ...] = (10, 11)
    allowed_exchange_codes: tuple[int, ...] = (1, 2, 3)


def ensure_directories() -> None:
    """Create project output directories if they do not exist."""

    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)

