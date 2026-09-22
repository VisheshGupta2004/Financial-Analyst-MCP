"""
Deterministic data-access layer (Phase 0).

Everything here is plain Python + pandas. No LLM, no MCP, no network-call
decision-making happens in this module -- it just fetches and validates.

`yfinance` is imported lazily inside `fetch_stock_data` (not at module import
time) for two reasons:
  1. It lets the rest of this module (ticker/date validation, retry logic)
     be imported and unit-tested in environments where yfinance isn't
     installed.
  2. It allows dependency injection of a fake "yfinance-like" object in
     tests via the `yf_module` parameter, so fetch logic (retries, empty-
     result handling, exception translation) can be fully unit tested
     without hitting the network.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd

from backend.tools.errors import (
    DataFetchError,
    InvalidDateRangeError,
    InvalidTickerError,
    NoDataError,
)

# Conservative but permissive ticker pattern: 1-10 chars, letters/digits plus
# the punctuation yfinance/most exchanges actually use (e.g. BRK-B, RDS.A).
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def validate_ticker(raw_ticker: str) -> str:
    """Normalize + syntactically validate a ticker symbol.

    This only checks *shape* (looks like a plausible ticker). Whether the
    ticker actually exists is determined later by whether fetch_stock_data
    returns any rows -- that's an existence check, not a syntax check, and
    conflating them would mean a network call every time we want to reject
    obviously-malformed input like "the stock market" or an empty string.
    """
    if raw_ticker is None:
        raise InvalidTickerError("Ticker must not be empty.", details={"ticker": raw_ticker})
    ticker = raw_ticker.strip().upper()
    if not ticker:
        raise InvalidTickerError("Ticker must not be empty.", details={"ticker": raw_ticker})

    # Share-class tickers are sometimes written with a dot (BRK.B, the older
    # NYSE ticker-tape notation) instead of a hyphen (BRK-B, what Yahoo
    # Finance / yfinance actually expects). LLMs frequently emit the dotted
    # form since it's the more commonly seen notation in prose. Normalize it
    # rather than reject a perfectly valid ticker the user clearly meant.
    if re.match(r"^[A-Z]{1,5}\.[A-Z]{1,2}$", ticker):
        ticker = ticker.replace(".", "-")

    if not _TICKER_RE.match(ticker):
        raise InvalidTickerError(
            f"'{raw_ticker}' does not look like a valid stock ticker symbol.",
            details={"ticker": raw_ticker},
        )
    return ticker


def validate_date_range(
    start: date, end: date, *, reference_date: date | None = None
) -> None:
    """Validate that a date range is sane: start <= end, end not absurdly in
    the future, and the range isn't longer than yfinance can reasonably serve.
    """
    reference_date = reference_date or date.today()
    if start > end:
        raise InvalidDateRangeError(
            f"Start date {start.isoformat()} is after end date {end.isoformat()}.",
            details={"start": start.isoformat(), "end": end.isoformat()},
        )
    if start > reference_date:
        raise InvalidDateRangeError(
            f"Start date {start.isoformat()} is in the future.",
            details={"start": start.isoformat()},
        )
    max_span = timedelta(days=366 * 50)
    if end - start > max_span:
        raise InvalidDateRangeError(
            "Requested date range spans more than 50 years, which is not supported.",
            details={"start": start.isoformat(), "end": end.isoformat()},
        )


def _default_yf_module():
    import yfinance as yf  # local import: optional dependency

    return yf


def fetch_stock_data(
    ticker: str,
    start: date,
    end: date,
    *,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    yf_module: Any = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for `ticker` between `start` and `end` (inclusive).

    Raises:
        InvalidTickerError: ticker fails syntax validation.
        InvalidDateRangeError: date range is invalid.
        NoDataError: request succeeded but returned zero rows (almost always
            means the ticker doesn't exist, or has no trading data in range).
        DataFetchError: transient/network/library failure, after retries.
    """
    ticker = validate_ticker(ticker)
    validate_date_range(start, end)

    yf = yf_module if yf_module is not None else _default_yf_module()

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            # yfinance's `end` is exclusive, so add a day to make our own
            # contract ("end is inclusive") consistent regardless of backend.
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                raise NoDataError(
                    f"No trading data found for ticker '{ticker}' between "
                    f"{start.isoformat()} and {end.isoformat()}. The ticker may not "
                    "exist, may be delisted, or may not have traded during this window.",
                    details={
                        "ticker": ticker,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                    },
                )
            # yfinance sometimes returns a MultiIndex column frame even for a
            # single ticker (varies by version) -- normalize to flat columns.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.sort_index()
            return df
        except NoDataError:
            raise
        except InvalidTickerError:
            raise
        except (
            Exception
        ) as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_fn(retry_backoff_seconds * attempt)
                continue
            raise DataFetchError(
                f"Failed to fetch data for '{ticker}' after {max_retries} attempts: {exc}",
                details={"ticker": ticker, "attempts": max_retries},
            ) from exc

    # Unreachable, but keeps type-checkers happy.
    raise DataFetchError(f"Unknown failure fetching '{ticker}'.") from last_exc


def fetch_multiple(
    tickers: list[str],
    start: date,
    end: date,
    *,
    max_retries: int = 3,
    retry_backoff_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    yf_module: Any = None,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers. Raises on the first failure -- callers
    that want partial results should catch per-ticker in a loop themselves;
    for this application's use case (a comparison chart), a single bad ticker
    should fail the whole request with a clear message rather than silently
    dropping it from the chart.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        normalized = validate_ticker(ticker)
        result[normalized] = fetch_stock_data(
            ticker,
            start,
            end,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep_fn=sleep_fn,
            yf_module=yf_module,
        )
    return result
