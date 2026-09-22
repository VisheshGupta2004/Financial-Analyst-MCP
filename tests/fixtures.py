"""Shared test fixtures: synthetic price data generation, no network needed."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def make_price_frame(
    start: date,
    n_days: int,
    start_price: float = 100.0,
    daily_drift: float = 0.0005,
    daily_vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame with a DatetimeIndex of business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    daily_returns = rng.normal(loc=daily_drift, scale=daily_vol, size=len(idx))
    close = start_price * np.cumprod(1 + daily_returns)
    df = pd.DataFrame(
        {
            "Open": close * (1 - 0.001),
            "High": close * (1 + 0.005),
            "Low": close * (1 - 0.005),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=len(idx)),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


class FakeYFinanceModule:
    """Stand-in for the `yfinance` module used in tests, so tools.data can be
    fully exercised (including retry / empty-result / exception paths)
    without the real yfinance package or any network access.
    """

    def __init__(
        self,
        frames: dict[str, pd.DataFrame] | None = None,
        raise_for: dict[str, Exception] | None = None,
    ):
        self.frames = frames or {}
        self.raise_for = raise_for or {}
        self.calls: list[dict] = []

    def download(self, ticker, start=None, end=None, progress=False, auto_adjust=True):
        self.calls.append({"ticker": ticker, "start": start, "end": end})
        if ticker in self.raise_for:
            raise self.raise_for[ticker]
        return self.frames.get(ticker, pd.DataFrame())
