import unittest
from datetime import date

from backend.tools.data import (
    fetch_multiple,
    fetch_stock_data,
    validate_date_range,
    validate_ticker,
)
from backend.tools.errors import (
    DataFetchError,
    InvalidDateRangeError,
    InvalidTickerError,
    NoDataError,
)
from tests.fixtures import FakeYFinanceModule, make_price_frame


class ValidateTickerTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self):
        self.assertEqual(validate_ticker("  aapl "), "AAPL")

    def test_accepts_dotted_and_hyphenated_tickers(self):
        self.assertEqual(validate_ticker("brk-b"), "BRK-B")
        # Yahoo/yfinance uses hyphens for share classes, not dots -- input in
        # the dotted form (BRK.B, RDS.A) should be normalized to the hyphenated
        # form yfinance actually expects, not passed through as-is.
        self.assertEqual(validate_ticker("rds.a"), "RDS-A")
        self.assertEqual(validate_ticker("BRK.B"), "BRK-B")

    def test_rejects_empty(self):
        with self.assertRaises(InvalidTickerError):
            validate_ticker("")
        with self.assertRaises(InvalidTickerError):
            validate_ticker("   ")

    def test_rejects_none(self):
        with self.assertRaises(InvalidTickerError):
            validate_ticker(None)

    def test_rejects_sentence_like_input(self):
        with self.assertRaises(InvalidTickerError):
            validate_ticker("the stock market")

    def test_rejects_too_long(self):
        with self.assertRaises(InvalidTickerError):
            validate_ticker("THISISNOTATICKER")


class ValidateDateRangeTests(unittest.TestCase):
    def test_accepts_valid_range(self):
        validate_date_range(
            date(2024, 1, 1), date(2024, 6, 1), reference_date=date(2026, 1, 1)
        )

    def test_rejects_start_after_end(self):
        with self.assertRaises(InvalidDateRangeError):
            validate_date_range(
                date(2024, 6, 1), date(2024, 1, 1), reference_date=date(2026, 1, 1)
            )

    def test_rejects_future_start(self):
        with self.assertRaises(InvalidDateRangeError):
            validate_date_range(
                date(2030, 1, 1), date(2030, 6, 1), reference_date=date(2026, 1, 1)
            )

    def test_rejects_absurdly_long_range(self):
        with self.assertRaises(InvalidDateRangeError):
            validate_date_range(
                date(1900, 1, 1), date(2026, 1, 1), reference_date=date(2026, 1, 1)
            )


class FetchStockDataTests(unittest.TestCase):
    def test_returns_dataframe_for_valid_ticker(self):
        frame = make_price_frame(date(2026, 1, 1), 40)
        fake_yf = FakeYFinanceModule(frames={"AAPL": frame})
        df = fetch_stock_data(
            "aapl",
            date(2026, 1, 1),
            date(2026, 3, 1),
            yf_module=fake_yf,
            sleep_fn=lambda s: None,
        )
        self.assertFalse(df.empty)
        self.assertIn("Close", df.columns)

    def test_raises_no_data_for_empty_result(self):
        fake_yf = FakeYFinanceModule(frames={})  # returns empty df for anything
        with self.assertRaises(NoDataError):
            fetch_stock_data(
                "ZZZQQQ1",
                date(2026, 1, 1),
                date(2026, 3, 1),
                yf_module=fake_yf,
                sleep_fn=lambda s: None,
            )

    def test_raises_invalid_ticker_before_hitting_network(self):
        fake_yf = FakeYFinanceModule()
        with self.assertRaises(InvalidTickerError):
            fetch_stock_data("", date(2026, 1, 1), date(2026, 3, 1), yf_module=fake_yf)
        self.assertEqual(fake_yf.calls, [])  # never called download()

    def test_retries_on_transient_failure_then_succeeds(self):
        frame = make_price_frame(date(2026, 1, 1), 20)

        class FlakyModule(FakeYFinanceModule):
            def download(self, ticker, **kwargs):
                self.calls.append({"ticker": ticker})
                if len(self.calls) < 3:
                    raise ConnectionError("simulated transient network failure")
                return frame

        flaky = FlakyModule()
        df = fetch_stock_data(
            "AAPL",
            date(2026, 1, 1),
            date(2026, 2, 1),
            yf_module=flaky,
            max_retries=5,
            sleep_fn=lambda s: None,
        )
        self.assertFalse(df.empty)
        self.assertEqual(len(flaky.calls), 3)

    def test_gives_up_after_max_retries(self):
        fake_yf = FakeYFinanceModule(raise_for={"AAPL": ConnectionError("down")})
        with self.assertRaises(DataFetchError):
            fetch_stock_data(
                "AAPL",
                date(2026, 1, 1),
                date(2026, 2, 1),
                yf_module=fake_yf,
                max_retries=2,
                sleep_fn=lambda s: None,
            )
        self.assertEqual(len(fake_yf.calls), 2)

    def test_flattens_multiindex_columns(self):
        import pandas as pd

        frame = make_price_frame(date(2026, 1, 1), 10)
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]])
        fake_yf = FakeYFinanceModule(frames={"AAPL": frame})
        df = fetch_stock_data(
            "AAPL",
            date(2026, 1, 1),
            date(2026, 1, 20),
            yf_module=fake_yf,
            sleep_fn=lambda s: None,
        )
        self.assertIn("Close", df.columns)


class FetchMultipleTests(unittest.TestCase):
    def test_fetches_all_tickers(self):
        frames = {
            "AAPL": make_price_frame(date(2026, 1, 1), 30, seed=1),
            "MSFT": make_price_frame(date(2026, 1, 1), 30, seed=2),
        }
        fake_yf = FakeYFinanceModule(frames=frames)
        result = fetch_multiple(
            ["aapl", "msft"],
            date(2026, 1, 1),
            date(2026, 2, 1),
            yf_module=fake_yf,
            sleep_fn=lambda s: None,
        )
        self.assertEqual(set(result.keys()), {"AAPL", "MSFT"})

    def test_fails_fast_on_one_bad_ticker(self):
        frames = {"AAPL": make_price_frame(date(2026, 1, 1), 30)}
        fake_yf = FakeYFinanceModule(
            frames=frames
        )  # MSFT missing -> empty -> NoDataError
        with self.assertRaises(NoDataError):
            fetch_multiple(
                ["aapl", "msft"],
                date(2026, 1, 1),
                date(2026, 2, 1),
                yf_module=fake_yf,
                sleep_fn=lambda s: None,
            )


if __name__ == "__main__":
    unittest.main()
