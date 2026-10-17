"""Leakage-resistant monthly portfolio construction and performance summaries."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .signals import attach_latest_available_signal


def combine_crsp_returns(ret: pd.Series, dlret: pd.Series) -> pd.Series:
    """Combine ordinary and delisting returns without replacing both-missing rows."""

    ret = pd.to_numeric(ret, errors="coerce")
    dlret = pd.to_numeric(dlret, errors="coerce")
    combined = (1.0 + ret.fillna(0.0)) * (1.0 + dlret.fillna(0.0)) - 1.0
    return combined.mask(ret.isna() & dlret.isna())


def prepare_crsp_monthly(crsp: pd.DataFrame) -> pd.DataFrame:
    """Clean CRSP monthly data and calculate investability fields."""

    frame = crsp.copy()
    frame["date"] = pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0)
    frame["ret_total"] = combine_crsp_returns(frame["ret"], frame["dlret"])
    frame["price"] = pd.to_numeric(frame["prc"], errors="coerce").abs()
    frame["market_cap"] = frame["price"] * pd.to_numeric(
        frame["shrout"], errors="coerce"
    )

    duplicate = frame.duplicated(["permno", "date"], keep=False)
    if duplicate.any():
        raise ValueError(
            "CRSP input contains duplicate PERMNO-month rows; inspect name/delist joins."
        )
    return frame.sort_values(["date", "permno"]).reset_index(drop=True)


def build_formation_panel(
    crsp: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Attach available signals at month t and returns from calendar month t+1."""

    monthly = prepare_crsp_monthly(crsp)
    formation = attach_latest_available_signal(monthly, signals)
    # CRSP contains one extra outcome month. Do not accidentally treat that
    # extra month as a new formation month with a carried-forward signal.
    if "statpers" in signals and signals["statpers"].notna().any():
        last_signal_month = pd.to_datetime(signals["statpers"]).max() + pd.offsets.MonthEnd(0)
        formation = formation[formation["date"] <= last_signal_month].copy()

    investable = (
        formation["shrcd"].isin(config.allowed_share_codes)
        & formation["exchcd"].isin(config.allowed_exchange_codes)
        & formation["price"].ge(config.min_price)
        & formation["numrec"].ge(config.min_analysts)
    )
    formation = formation[investable].copy()
    formation["formation_month"] = formation["date"].dt.to_period("M")
    formation["return_month"] = formation["formation_month"] + 1

    outcomes = monthly[["permno", "date", "ret_total"]].copy()
    outcomes["return_month"] = outcomes["date"].dt.to_period("M")
    outcomes = outcomes.rename(
        columns={"date": "return_date", "ret_total": "forward_return"}
    )
    panel = formation.merge(
        outcomes,
        on=["permno", "return_month"],
        how="left",
        validate="many_to_one",
    )

    timing_violation = panel["available_date"] > panel["date"]
    if timing_violation.any():
        raise AssertionError("Look-ahead detected: signal was unavailable at formation.")
    return panel


def _monthly_weights(group: pd.DataFrame, signal: str, quantile: float) -> pd.Series:
    """Create equal-weighted long and short legs for one cross-section."""

    weights = pd.Series(0.0, index=group.index)
    # Selection cannot depend on whether next month's return happens to exist.
    # Looking at forward_return here would be a subtle form of look-ahead bias.
    candidates = group.loc[group[signal].notna(), signal]
    if len(candidates) < 2 or candidates.nunique() < 2:
        return weights

    leg_size = max(1, int(np.floor(len(candidates) * quantile)))
    # Stable PERMNO tie-break prevents accidental dependence on input row order.
    ranked = group.loc[candidates.index].sort_values(
        [signal, "permno"], kind="mergesort"
    )
    short_ids = ranked.index[:leg_size]
    long_ids = ranked.index[-leg_size:]
    weights.loc[long_ids] = 1.0 / leg_size
    weights.loc[short_ids] = -1.0 / leg_size
    return weights


def run_long_short_backtest(
    panel: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct portfolios and return monthly results plus security weights."""

    if not 0 < config.quantile <= 0.5:
        raise ValueError("quantile must lie in (0, 0.5]")
    if config.signal not in panel:
        raise KeyError(f"Missing signal column: {config.signal}")

    holdings = panel.copy()
    holdings["weight"] = 0.0
    for _, month in holdings.groupby("date", sort=True):
        holdings.loc[month.index, "weight"] = _monthly_weights(
            month, config.signal, config.quantile
        )
    holdings["contribution"] = holdings["weight"] * holdings["forward_return"]
    holdings["held_return_missing"] = (
        holdings["weight"].ne(0.0) & holdings["forward_return"].isna()
    )

    # Use the union of current and prior holdings so complete exits count toward
    # turnover even when a security leaves the next month's investable universe.
    current = holdings.loc[holdings["weight"].ne(0), ["date", "permno", "weight"]]
    previous = current.copy()
    previous["date"] = previous["date"] + pd.offsets.MonthEnd(1)
    previous = previous.rename(columns={"weight": "previous_weight"})
    turnover_rows = current.merge(previous, on=["date", "permno"], how="outer")
    turnover_rows["turnover_piece"] = (
        turnover_rows["weight"].fillna(0.0)
        - turnover_rows["previous_weight"].fillna(0.0)
    ).abs()
    valid_dates = pd.Index(holdings["date"].unique())
    turnover = (
        turnover_rows[turnover_rows["date"].isin(valid_dates)]
        .groupby("date")["turnover_piece"]
        .sum()
    )

    results = holdings.groupby("date", as_index=False).agg(
        gross_return=("contribution", lambda x: x.sum(min_count=1)),
        n_securities=("weight", lambda x: int((x != 0).sum())),
        outcome_coverage=("forward_return", lambda x: float(x.notna().mean())),
        held_return_missing=("held_return_missing", "any"),
    )
    results["turnover"] = results["date"].map(turnover).fillna(0.0)
    # Do not turn an unavailable held-stock return into an assumed zero return.
    results.loc[results["held_return_missing"], "gross_return"] = np.nan
    results["cost"] = results["turnover"] * config.transaction_cost_bps / 10_000.0
    results["net_return"] = results["gross_return"] - results["cost"]
    results["wealth"] = (1.0 + results["net_return"]).cumprod(skipna=False)
    return results, holdings


def performance_summary(
    monthly_results: pd.DataFrame,
    return_column: str = "net_return",
    *,
    strict: bool = False,
) -> pd.Series:
    """Report coverage and performance without imputing missing outcomes.

    With gaps, moment statistics describe only observed months and CAGR and
    drawdown are unavailable. Set strict=True to reject missing held outcomes.
    """

    missing_held_month = monthly_results[return_column].isna() & monthly_results[
        "n_securities"
    ].gt(0)
    if missing_held_month.any():
        dates = pd.to_datetime(monthly_results.loc[missing_held_month, "date"]).dt.strftime("%Y-%m-%d").tolist()
        message = (
            f"Missing held-security outcomes in {int(missing_held_month.sum())} months. "
            f"Affected formation months: {dates[:5]}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(
            message + " Available-month statistics exclude missing months and may be biased; "
            "CAGR and max drawdown are unavailable for an incomplete return series.",
            stacklevel=2,
        )
    returns = monthly_results[return_column].dropna()
    if returns.empty and strict:
        raise ValueError("No returns available for performance summary")
    complete = len(returns) == len(monthly_results) and not returns.empty
    cagr = (1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0 if complete else np.nan
    annualized_mean = returns.mean() * 12.0
    annual_volatility = returns.std(ddof=1) * np.sqrt(12.0)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(12.0)
        if returns.std(ddof=1) > 0
        else np.nan
    )
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1.0) - 1.0
    return pd.Series(
        {
            "months": len(returns),
            "total_months": len(monthly_results),
            "missing_return_months": int(monthly_results[return_column].isna().sum()),
            "months_with_missing_held_return": int(missing_held_month.sum()),
            "return_coverage": len(returns) / len(monthly_results) if len(monthly_results) else np.nan,
            "cagr": cagr,
            "annualized_mean_return": annualized_mean,
            "annual_volatility": annual_volatility,
            "sharpe_zero_rf": sharpe,
            "max_drawdown": drawdown.min() if complete else np.nan,
            "average_monthly_turnover": monthly_results.loc[returns.index, "turnover"].mean(),
        }
    )
