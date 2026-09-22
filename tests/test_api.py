"""
FastAPI endpoint tests.

REQUIRES: fastapi, pydantic, httpx (TestClient's transport). Not executable
in the sandbox that produced this repository -- see README "What Was and
Wasn't Executed". backend.orchestrator.run_pipeline (the actual pipeline
logic /analyze delegates to) is exercised indirectly through
tests/test_planner.py + tests/test_llm_client.py + tests/test_mcp.py; these
tests focus on the HTTP layer itself: request/response shape, status codes,
and error mapping.
"""

import unittest
from contextlib import ExitStack
from unittest import mock


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient

            import backend.main as main_module
        except ImportError as exc:
            self.skipTest(f"fastapi/httpx not installed: {exc}")

        self.main_module = main_module
        # TestClient only runs FastAPI's lifespan (startup/shutdown --
        # this is what populates app.state.llm_client) when entered as a
        # context manager. Using it "bare" silently skips lifespan, which
        # then surfaces later as a confusing AttributeError on
        # request.app.state.llm_client instead of a clear setup error.
        self._stack = ExitStack()
        self.client = self._stack.enter_context(TestClient(main_module.app))
        self.addCleanup(self._stack.close)

    def test_health_endpoint_returns_200_even_when_mcp_unreachable(self):
        # No real MCP server subprocess is running in this test environment,
        # so /health should report degraded, not 500.
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["status"], {"ok", "degraded"})
        self.assertIn("llm_provider", body)

    def test_analyze_rejects_empty_prompt(self):
        response = self.client.post("/analyze", json={"prompt": ""})
        self.assertEqual(response.status_code, 422)  # pydantic min_length violation

    def test_analyze_success_path_with_mocked_pipeline(self):
        from backend.schemas import AnalysisRequest

        fake_request = AnalysisRequest(
            tickers=["AAPL", "MSFT"],
            timeframe_pattern="last_6_months",
            metric="normalized_returns",
        )
        fake_result = {
            "success": True,
            "request": fake_request,
            "warnings": [],
            "errors": [],
            "chart_path": "output/comparison_test.png",
            "metrics": {
                "AAPL": {
                    "total_return": 0.1,
                    "cagr": 0.2,
                    "annualized_volatility": 0.3,
                    "max_drawdown": -0.05,
                },
                "MSFT": {
                    "total_return": 0.05,
                    "cagr": 0.1,
                    "annualized_volatility": 0.25,
                    "max_drawdown": -0.03,
                },
            },
        }

        with mock.patch("backend.main.MCPClient") as mock_mcp_client_cls, mock.patch(
            "backend.main.run_pipeline", new=mock.AsyncMock(return_value=fake_result)
        ):
            mock_mcp_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock.MagicMock()
            )
            mock_mcp_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            response = self.client.post(
                "/analyze",
                json={"prompt": "Compare Apple and Microsoft over the last 6 months."},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["chart_path"], "output/comparison_test.png")
        self.assertIn("AAPL", body["summary"])

    def test_analyze_clarification_needed_returns_200_with_errors(self):
        from backend.schemas import AnalysisRequest

        fake_request = AnalysisRequest(
            tickers=[], needs_clarification=True, clarification_message="Which stocks?"
        )
        fake_result = {
            "success": False,
            "request": fake_request,
            "warnings": [],
            "errors": ["Which stocks?"],
        }

        with mock.patch("backend.main.MCPClient") as mock_mcp_client_cls, mock.patch(
            "backend.main.run_pipeline", new=mock.AsyncMock(return_value=fake_result)
        ):
            mock_mcp_client_cls.return_value.__aenter__ = mock.AsyncMock(
                return_value=mock.MagicMock()
            )
            mock_mcp_client_cls.return_value.__aexit__ = mock.AsyncMock(
                return_value=False
            )

            response = self.client.post(
                "/analyze", json={"prompt": "How is the market doing?"}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertIn("Which stocks?", body["errors"])


if __name__ == "__main__":
    unittest.main()
