"""
Deterministic financial analysis (Phase 0). Pure pandas/numpy -- no LLM,
no I/O. Every function here takes DataFrames/Series in and returns
DataFrames/Series/dicts out, which makes them trivial to unit test with
synthetic data instead of real market data.

Conventions:
  - Input price DataFrames are expected to have a 'Close' column (as
    returned by tools.data.fetch_stock_data) and a DatetimeIndex.
  - All returned Series/DataFrames are indexed by date, ascending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.tools.errors import ToolError

TRADING_DAYS_PER_YEAR = 252


def _close(df: pd.DataFrame) -> pd.Series:
    if "Close" not in df.columns:
        raise ToolError(
            f"Expected a 'Close' column, got columns: {list(df.columns)}",
            code="bad_input_frame",
        )
    return df["Close"].astype(float)


def daily_returns(df: pd.DataFrame) -> pd.Series:
    """Simple daily percentage returns, e.g. 0.01 == +1%."""
    close = _close(df)
    return close.pct_change().dropna()


def cumulative_returns(df: pd.DataFrame) -> pd.Series:
    """Cumulative return relative to the first day in the series (0.0 == flat)."""
    close = _close(df)
    if close.empty:
        raise ToolError(
            "Cannot compute cumulative returns on empty data.", code="empty_series"
        )
    return (close / close.iloc[0]) - 1.0


def normalized_prices(df: pd.DataFrame, base: float = 100.0) -> pd.Series:
    """Rebase the close-price series so it starts at `base` (default 100).
    This is what makes cross-ticker comparison charts visually meaningful
    when the raw share prices are wildly different magnitudes.
    """
    close = _close(df)
    if close.empty:
        raise ToolError("Cannot normalize an empty price series.", code="empty_series")
    return (close / close.iloc[0]) * base


def annualized_volatility(df: pd.DataFrame) -> float:
    """Annualized volatility (std of daily returns * sqrt(trading days/year))."""
    rets = daily_returns(df)
    if len(rets) < 2:
        raise ToolError(
            "Need at least 2 daily returns to compute volatility.",
            code="insufficient_data",
        )
    return float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(df: pd.DataFrame) -> float:
    """Maximum peak-to-trough decline over the period, as a negative fraction
    (e.g. -0.23 == a 23% drawdown at the worst point).
    """
    close = _close(df)
    if close.empty:
        raise ToolError("Cannot compute drawdown on empty data.", code="empty_series")
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(drawdown.min())


def total_return(df: pd.DataFrame) -> float:
    close = _close(df)
    if len(close) < 2:
        raise ToolError(
            "Need at least 2 price points to compute total return.",
            code="insufficient_data",
        )
    return float((close.iloc[-1] / close.iloc[0]) - 1.0)


def cagr(df: pd.DataFrame) -> float:
    """Compound annual growth rate over the period covered by df."""
    close = _close(df)
    if len(close) < 2:
        raise ToolError(
            "Need at least 2 price points to compute CAGR.", code="insufficient_data"
        )
    days = (close.index[-1] - close.index[0]).days
    if days <= 0:
        raise ToolError(
            "Cannot compute CAGR over a zero-length period.", code="insufficient_data"
        )
    years = days / 365.25
    growth = close.iloc[-1] / close.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (1 / years) - 1)


def basic_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Bundle of the headline metrics used in the API response / chart caption."""
    return {
        "total_return": total_return(df),
        "cagr": cagr(df),
        "annualized_volatility": annualized_volatility(df),
        "max_drawdown": max_drawdown(df),
    }


def compute_comparison(
    price_data: dict[str, pd.DataFrame], metric: str = "normalized_returns"
) -> pd.DataFrame:
    """Align multiple tickers' series onto a shared date index for comparison.

    `metric` controls what each column represents:
      - "normalized_returns" / "normalized_price": rebased to start at 100
      - "cumulative_returns": cumulative % return from day 0
      - "price": raw close price
    """
    if not price_data:
        raise ToolError("No price data provided for comparison.", code="empty_input")

    series_map: dict[str, pd.Series] = {}
    for ticker, df in price_data.items():
        if metric in ("normalized_returns", "normalized_price"):
            series_map[ticker] = normalized_prices(df)
        elif metric == "cumulative_returns":
            series_map[ticker] = cumulative_returns(df)
        elif metric == "price":
            series_map[ticker] = _close(df)
        else:
            raise ToolError(f"Unknown comparison metric: {metric!r}", code="bad_metric")

    combined = pd.DataFrame(series_map)
    # Inner-join semantics: only dates where *all* tickers have data. This
    # keeps a comparison chart honest (no partially-missing series) at the
    # cost of trimming to the shortest overlapping trading history.
    combined = combined.dropna(how="any")
    if combined.empty:
        raise ToolError(
            "No overlapping trading dates across the requested tickers.",
            code="no_overlap",
        )
    return combined
