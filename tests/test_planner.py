"""
Planner integration tests.

REQUIRES: pydantic (backend.schemas). Not executable in the sandbox that
produced this repository -- see README "What Was and Wasn't Executed". The
underlying extraction logic exercised here is separately verified, without
pydantic, in tests/test_llm_client.py and tests/test_date_utils.py.
"""

import unittest
from datetime import date

from backend.llm_client import MockLLMClient

try:
    from backend.planner import build_analysis_request

    _IMPORT_ERROR = None
except (
    ImportError
) as exc:  # pragma: no cover - exercised only without pydantic installed
    _IMPORT_ERROR = exc


@unittest.skipIf(_IMPORT_ERROR is not None, f"pydantic not installed: {_IMPORT_ERROR}")
class PlannerWithMockLLMTests(unittest.TestCase):
    def setUp(self):
        self.llm = MockLLMClient()

    def test_two_ticker_comparison_resolves_dates(self):
        req = build_analysis_request(
            "Compare Apple and Microsoft over the last 6 months.", self.llm
        )
        self.assertEqual(set(req.tickers), {"AAPL", "MSFT"})
        self.assertEqual(req.analysis_type, "comparison")
        self.assertFalse(req.needs_clarification)
        self.assertIsInstance(req.start_date, date)
        self.assertIsInstance(req.end_date, date)
        self.assertLess(req.start_date, req.end_date)

    def test_missing_timeframe_triggers_clarification(self):
        req = build_analysis_request("Compare AAPL and MSFT", self.llm)
        self.assertTrue(req.needs_clarification)
        self.assertIsNotNone(req.clarification_message)

    def test_empty_prompt_triggers_clarification_without_calling_llm(self):
        req = build_analysis_request("   ", self.llm)
        self.assertTrue(req.needs_clarification)
        self.assertEqual(req.tickers, [])

    def test_unrecognized_company_triggers_clarification(self):
        req = build_analysis_request("How is the market doing?", self.llm)
        self.assertTrue(req.needs_clarification)
        self.assertEqual(req.tickers, [])

    def test_raw_prompt_is_preserved_for_logging(self):
        prompt = "Show NVDA year to date."
        req = build_analysis_request(prompt, self.llm)
        self.assertEqual(req.raw_prompt, prompt)


if __name__ == "__main__":
    unittest.main()
