import unittest
from datetime import date

import numpy as np
import pandas as pd

from backend.tools.analysis import (
    annualized_volatility,
    basic_metrics,
    cagr,
    compute_comparison,
    cumulative_returns,
    daily_returns,
    max_drawdown,
    normalized_prices,
    total_return,
)
from backend.tools.errors import ToolError
from tests.fixtures import make_price_frame


def _flat_frame(price: float, n: int, start=date(2026, 1, 1)) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame({"Close": [price] * n}, index=idx)


def _monotonic_frame(prices: list[float], start=date(2026, 1, 1)) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"Close": prices}, index=idx)


class DailyReturnsTests(unittest.TestCase):
    def test_flat_price_has_zero_returns(self):
        df = _flat_frame(100.0, 10)
        rets = daily_returns(df)
        self.assertTrue(np.allclose(rets.values, 0.0))

    def test_known_return_sequence(self):
        df = _monotonic_frame([100.0, 110.0, 99.0])
        rets = daily_returns(df)
        self.assertAlmostEqual(rets.iloc[0], 0.10, places=8)
        self.assertAlmostEqual(rets.iloc[1], -0.10, places=8)


class CumulativeReturnsTests(unittest.TestCase):
    def test_starts_at_zero(self):
        df = _monotonic_frame([100.0, 110.0, 120.0])
        cum = cumulative_returns(df)
        self.assertAlmostEqual(cum.iloc[0], 0.0)

    def test_final_value_matches_total_return(self):
        df = _monotonic_frame([50.0, 55.0, 60.0, 75.0])
        cum = cumulative_returns(df)
        self.assertAlmostEqual(cum.iloc[-1], total_return(df))

    def test_raises_on_empty(self):
        empty = pd.DataFrame({"Close": []})
        with self.assertRaises(ToolError):
            cumulative_returns(empty)


class NormalizedPricesTests(unittest.TestCase):
    def test_starts_at_base(self):
        df = _monotonic_frame([250.0, 260.0, 240.0])
        norm = normalized_prices(df, base=100.0)
        self.assertAlmostEqual(norm.iloc[0], 100.0)

    def test_proportional_to_price(self):
        df = _monotonic_frame([200.0, 400.0])
        norm = normalized_prices(df, base=100.0)
        self.assertAlmostEqual(norm.iloc[1], 200.0)  # doubled -> normalized doubles too


class VolatilityAndDrawdownTests(unittest.TestCase):
    def test_zero_volatility_for_flat_series(self):
        df = _flat_frame(50.0, 30)
        self.assertAlmostEqual(annualized_volatility(df), 0.0, places=8)

    def test_max_drawdown_is_negative_or_zero(self):
        df = _monotonic_frame([100, 120, 80, 90, 130])
        dd = max_drawdown(df)
        self.assertLessEqual(dd, 0.0)
        # peak 120 -> trough 80 => -33.33%
        self.assertAlmostEqual(dd, (80 - 120) / 120, places=4)

    def test_monotonic_increase_has_zero_drawdown(self):
        df = _monotonic_frame([10, 20, 30, 40])
        self.assertAlmostEqual(max_drawdown(df), 0.0)

    def test_volatility_raises_on_insufficient_data(self):
        df = _monotonic_frame([100.0])
        with self.assertRaises(ToolError):
            annualized_volatility(df)


class CagrTests(unittest.TestCase):
    def test_doubling_over_one_year(self):
        idx = [date(2025, 1, 1), date(2026, 1, 1)]
        df = pd.DataFrame({"Close": [100.0, 200.0]}, index=pd.DatetimeIndex(idx))
        self.assertAlmostEqual(cagr(df), 1.0, places=2)

    def test_total_loss_returns_negative_one(self):
        idx = [date(2025, 1, 1), date(2026, 1, 1)]
        df = pd.DataFrame({"Close": [100.0, 0.0]}, index=pd.DatetimeIndex(idx))
        self.assertEqual(cagr(df), -1.0)


class BasicMetricsTests(unittest.TestCase):
    def test_returns_all_expected_keys(self):
        df = make_price_frame(date(2026, 1, 1), 60)
        metrics = basic_metrics(df)
        self.assertEqual(
            set(metrics.keys()),
            {"total_return", "cagr", "annualized_volatility", "max_drawdown"},
        )
        for v in metrics.values():
            self.assertIsInstance(v, float)


class ComputeComparisonTests(unittest.TestCase):
    def test_normalized_comparison_aligns_and_rebases(self):
        a = _monotonic_frame([100.0, 110.0, 121.0])
        b = _monotonic_frame([50.0, 60.0, 55.0])
        combined = compute_comparison({"A": a, "B": b}, metric="normalized_returns")
        self.assertEqual(list(combined.columns), ["A", "B"])
        self.assertAlmostEqual(combined["A"].iloc[0], 100.0)
        self.assertAlmostEqual(combined["B"].iloc[0], 100.0)

    def test_raises_on_empty_input(self):
        with self.assertRaises(ToolError):
            compute_comparison({})

    def test_raises_on_unknown_metric(self):
        a = _monotonic_frame([1.0, 2.0])
        with self.assertRaises(ToolError):
            compute_comparison({"A": a}, metric="not_a_real_metric")

    def test_raises_when_no_date_overlap(self):
        a = _monotonic_frame([100.0, 110.0], start=date(2020, 1, 1))
        b = _monotonic_frame([50.0, 60.0], start=date(2026, 1, 1))
        with self.assertRaises(ToolError):
            compute_comparison({"A": a, "B": b}, metric="price")

    def test_price_metric_uses_raw_values(self):
        a = _monotonic_frame([100.0, 110.0])
        combined = compute_comparison({"A": a}, metric="price")
        self.assertAlmostEqual(combined["A"].iloc[0], 100.0)


if __name__ == "__main__":
    unittest.main()
