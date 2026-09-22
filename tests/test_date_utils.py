import unittest
from datetime import date

from backend.date_utils import TimeframeError, resolve_pattern, sniff_timeframe


class ResolvePatternTests(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 8, 21)

    def test_last_n_months(self):
        r = resolve_pattern("last_6_months", self.today)
        self.assertEqual(r.start, date(2026, 2, 21))
        self.assertEqual(r.end, self.today)

    def test_last_n_days(self):
        r = resolve_pattern("last_30_days", self.today)
        self.assertEqual(r.start, date(2026, 7, 22))

    def test_last_n_years(self):
        r = resolve_pattern("last_1_years", self.today)
        self.assertEqual(r.start, date(2025, 8, 21))

    def test_ytd(self):
        r = resolve_pattern("ytd", self.today)
        self.assertEqual(r.start, date(2026, 1, 1))
        self.assertEqual(r.end, self.today)

    def test_year_full_past_year(self):
        r = resolve_pattern("year:2025", self.today)
        self.assertEqual(r.start, date(2025, 1, 1))
        self.assertEqual(r.end, date(2025, 12, 31))

    def test_year_current_year_clips_to_today(self):
        r = resolve_pattern("year:2026", self.today)
        self.assertEqual(r.start, date(2026, 1, 1))
        self.assertEqual(r.end, self.today)

    def test_since_date(self):
        r = resolve_pattern("since:2024-01-01", self.today)
        self.assertEqual(r.start, date(2024, 1, 1))
        self.assertEqual(r.end, self.today)

    def test_explicit_range(self):
        r = resolve_pattern("2024-01-01:2024-06-30", self.today)
        self.assertEqual(r.start, date(2024, 1, 1))
        self.assertEqual(r.end, date(2024, 6, 30))

    def test_month_end_clamping(self):
        # Jan 31 minus 1 month should clamp to Feb 28/29, not overflow.
        r = resolve_pattern("last_1_months", date(2026, 3, 31))
        self.assertEqual(r.start, date(2026, 2, 28))

    def test_invalid_pattern_raises(self):
        with self.assertRaises(TimeframeError):
            resolve_pattern("sometime last week probably", self.today)

    def test_zero_count_raises(self):
        with self.assertRaises(TimeframeError):
            resolve_pattern("last_0_months", self.today)

    def test_inverted_range_raises(self):
        with self.assertRaises(TimeframeError):
            resolve_pattern("2024-06-30:2024-01-01", self.today)


class SniffTimeframeTests(unittest.TestCase):
    def test_last_n_months_phrase(self):
        self.assertEqual(sniff_timeframe("over the last 6 months"), "last_6_months")

    def test_past_year_phrase(self):
        self.assertEqual(sniff_timeframe("over the past year"), "last_1_years")

    def test_singular_last_unit(self):
        self.assertEqual(sniff_timeframe("during the last month"), "last_1_months")

    def test_during_year_phrase(self):
        self.assertEqual(sniff_timeframe("performance during 2025"), "year:2025")

    def test_bare_year_phrase(self):
        self.assertEqual(sniff_timeframe("Tesla's performance in 2025"), "year:2025")

    def test_since_month_year_phrase(self):
        self.assertEqual(sniff_timeframe("since January 2024"), "since:2024-01-01")

    def test_since_iso_phrase(self):
        self.assertEqual(sniff_timeframe("since 2024-03-15"), "since:2024-03-15")

    def test_ytd_phrase(self):
        self.assertEqual(sniff_timeframe("show me AAPL YTD"), "ytd")
        self.assertEqual(sniff_timeframe("year to date performance"), "ytd")

    def test_no_timeframe_returns_none(self):
        self.assertIsNone(sniff_timeframe("compare apple and microsoft"))

    def test_past_6_months_variant(self):
        self.assertEqual(sniff_timeframe("compare over past 6 months"), "last_6_months")


if __name__ == "__main__":
    unittest.main()
