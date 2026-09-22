import unittest

from backend.llm_client import MockLLMClient

# A minimal schema shape sufficient to trigger the mock's AnalysisRequest branch.
_ANALYSIS_SCHEMA = {
    "properties": {
        "tickers": {},
        "timeframe_pattern": {},
        "metric": {},
        "analysis_type": {},
    }
}


class MockLLMExtractionTests(unittest.TestCase):
    def setUp(self):
        self.client = MockLLMClient()

    def _extract(self, prompt: str) -> dict:
        return self.client.complete_json(
            system_prompt="extract intent",
            user_prompt=prompt,
            json_schema=_ANALYSIS_SCHEMA,
        )

    def test_two_company_names_and_relative_timeframe(self):
        result = self._extract("Compare Apple and Microsoft over the last 6 months.")
        self.assertEqual(set(result["tickers"]), {"AAPL", "MSFT"})
        self.assertEqual(result["timeframe_pattern"], "last_6_months")
        self.assertEqual(result["analysis_type"], "comparison")
        self.assertFalse(result["needs_clarification"])

    def test_single_company_and_year(self):
        result = self._extract("Show me Tesla's performance during 2025.")
        self.assertEqual(result["tickers"], ["TSLA"])
        self.assertEqual(result["timeframe_pattern"], "year:2025")
        self.assertEqual(result["analysis_type"], "single")

    def test_bare_ticker_symbols(self):
        result = self._extract("Compare AAPL, MSFT and NVDA over the past 6 months.")
        self.assertEqual(set(result["tickers"]), {"AAPL", "MSFT", "NVDA"})

    def test_normalized_returns_keyword_detected(self):
        result = self._extract(
            "Plot normalized returns for Tesla and Apple since January 2024."
        )
        self.assertEqual(result["metric"], "normalized_returns")
        self.assertEqual(result["timeframe_pattern"], "since:2024-01-01")
        self.assertEqual(set(result["tickers"]), {"TSLA", "AAPL"})

    def test_no_ticker_flags_clarification(self):
        result = self._extract("What's the weather like today?")
        self.assertEqual(result["tickers"], [])
        self.assertTrue(result["needs_clarification"])

    def test_unrecognized_schema_returns_empty_dict(self):
        result = self.client.complete_json(
            system_prompt="x",
            user_prompt="y",
            json_schema={"properties": {"something_else": {}}},
        )
        self.assertEqual(result, {})

    def test_ytd_keyword_not_picked_up_as_ticker(self):
        result = self._extract("Compare TSLA and AAPL normalized returns YTD")
        self.assertEqual(set(result["tickers"]), {"TSLA", "AAPL"})
        self.assertEqual(result["timeframe_pattern"], "ytd")

    def test_sp_ampersand_does_not_produce_single_letter_tickers(self):
        result = self._extract("Show me the S&P 500 this year")
        self.assertNotIn("S", result["tickers"])
        self.assertNotIn("P", result["tickers"])

    def test_todays_date_prefix_does_not_leak_a_fake_timeframe(self):
        # Regression test: planner.py wraps the real prompt as
        # "Today's date is YYYY-MM-DD.\nUser request: <prompt>" before
        # calling the LLM. The date stamp itself must never be scanned for
        # tickers/timeframe -- only the actual "User request: ..." portion.
        wrapped = "Today's date is 2026-08-22.\nUser request: Compare AAPL and MSFT"
        result = self._extract(wrapped)
        self.assertEqual(set(result["tickers"]), {"AAPL", "MSFT"})
        self.assertIsNone(result["timeframe_pattern"])

    def test_todays_date_prefix_with_real_timeframe_still_works(self):
        wrapped = "Today's date is 2026-08-22.\nUser request: Compare Apple and Microsoft over the last 6 months."
        result = self._extract(wrapped)
        self.assertEqual(result["timeframe_pattern"], "last_6_months")

    def test_critic_schema_returns_approval_shape(self):
        result = self.client.complete_json(
            system_prompt="x",
            user_prompt="y",
            json_schema={"properties": {"approved": {}, "issues": {}}},
        )
        self.assertIn("approved", result)


if __name__ == "__main__":
    unittest.main()
