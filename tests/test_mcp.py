"""
MCP server tool tests.

REQUIRES: the `mcp` package (backend.mcp_server imports mcp.server.fastmcp).
Not executable in the sandbox that produced this repository -- see README
"What Was and Wasn't Executed". The deterministic logic each tool wraps
(backend.tools.data / analysis / plotting) *is* fully unit tested without
`mcp` in test_data.py, test_analysis.py, and test_plotting.py -- these tests
only add a thin layer checking the tool functions marshal arguments/results
correctly (str dates in, JSON-serializable dicts out) and translate
ToolError -> {"success": False, ...} rather than raising.

FastMCP's @mcp.tool() decorator registers the function but (per the
documented API) still returns a plain callable, so these tests invoke the
tool functions directly as Python functions rather than going through the
MCP protocol/transport layer -- protocol-level behavior (tool discovery,
JSON-RPC framing) is instead covered by an end-to-end manual check
documented in README ("How to start the MCP server").
"""

import unittest
from datetime import date
from unittest import mock

from tests.fixtures import FakeYFinanceModule, make_price_frame


class MCPToolTests(unittest.TestCase):
    def setUp(self):
        # Imported lazily inside each test (not at module scope) so this
        # whole file safely no-ops with a clear skip if `mcp` isn't
        # installed, rather than breaking collection of the other test files.
        try:
            from backend import mcp_server
        except SystemExit as exc:
            self.skipTest(f"mcp package not installed: {exc}")
        self.mcp_server = mcp_server

    def test_fetch_stock_data_success(self):
        frame = make_price_frame(date(2026, 1, 1), 40)
        fake_yf = FakeYFinanceModule(frames={"AAPL": frame})
        with mock.patch("backend.tools.data._default_yf_module", return_value=fake_yf):
            result = self.mcp_server.fetch_stock_data(
                "AAPL", "2026-01-01", "2026-03-01"
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["ticker"], "AAPL")
        self.assertGreater(len(result["records"]), 0)
        self.assertIn("total_return", result["metrics"])

    def test_fetch_stock_data_invalid_ticker_returns_structured_error(self):
        result = self.mcp_server.fetch_stock_data("", "2026-01-01", "2026-03-01")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "invalid_ticker")

    def test_fetch_stock_data_bad_date_format(self):
        result = self.mcp_server.fetch_stock_data("AAPL", "not-a-date", "2026-03-01")
        self.assertFalse(result["success"])

    def test_plot_comparison_success_returns_chart_path(self):
        frames = {
            "AAPL": make_price_frame(date(2026, 1, 1), 40, seed=1),
            "MSFT": make_price_frame(date(2026, 1, 1), 40, seed=2),
        }
        fake_yf = FakeYFinanceModule(frames=frames)
        with mock.patch("backend.tools.data._default_yf_module", return_value=fake_yf):
            result = self.mcp_server.plot_comparison(
                ["AAPL", "MSFT"], "2026-01-01", "2026-03-01"
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["chart_path"].endswith(".png"))
        self.assertIn("AAPL", result["metrics"])
        self.assertIn("MSFT", result["metrics"])

    def test_plot_comparison_no_data_returns_structured_error(self):
        fake_yf = FakeYFinanceModule(frames={})
        with mock.patch("backend.tools.data._default_yf_module", return_value=fake_yf):
            result = self.mcp_server.plot_comparison(
                ["ZZZQQQ1"], "2026-01-01", "2026-03-01"
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "no_data")


if __name__ == "__main__":
    unittest.main()
