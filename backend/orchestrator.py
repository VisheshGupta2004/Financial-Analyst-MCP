"""
MCP client + end-to-end orchestration (Phase 1 client half, Phase 3 pipeline).

This is *this project's own* MCP client -- it connects to backend/mcp_server.py
as a subprocess over stdio, discovers its tools, and invokes them. It does
not depend on Cursor, or on any IDE, in any way: it is a small async Python
class usable from FastAPI, a script, or a test.

NOTE ON THIS DELIVERY: requires the `mcp` package (not installed in the
sandbox that produced this codebase -- see README). Written against the
documented `mcp.ClientSession` / stdio transport API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from backend.config import MCPConfig
from backend.critic import review as critic_review
from backend.llm_client import LLMClient
from backend.planner import PlannerError, build_analysis_request
from backend.schemas import AnalysisRequest

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without `mcp` installed
    _MCP_AVAILABLE = False


class OrchestratorError(Exception):
    """Raised for MCP connection failures, tool-invocation failures, or
    timeouts. Carries a `code` so main.py can map it to a clean HTTP error
    rather than a raw traceback.
    """

    def __init__(self, message: str, *, code: str = "orchestrator_error"):
        super().__init__(message)
        self.message = message
        self.code = code


class MCPClient:
    """Async context-managed MCP client connected to the financial-analyst
    MCP server over stdio.

    Usage:
        async with MCPClient(config) as client:
            tools = await client.list_tools()
            result = await client.call_tool("plot_comparison", {...})
    """

    def __init__(self, config: MCPConfig):
        if not _MCP_AVAILABLE:
            raise OrchestratorError(
                "The 'mcp' package is not installed. Install project dependencies "
                "first: pip install -r requirements.txt",
                code="mcp_not_installed",
            )
        self._config = config
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def connect(self) -> MCPClient:
        """Start the MCP server subprocess and initialize a session. Safe to
        call again after disconnect() (e.g. to reconnect after a crash) --
        each call builds a fresh AsyncExitStack.
        """
        self._stack = AsyncExitStack()
        try:
            server_params = StdioServerParameters(
                command=self._config.server_command,
                args=self._config.server_args_list(),
            )
            read, write = await self._stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            return self
        except Exception as exc:
            await self._stack.aclose()
            self._session = None
            raise OrchestratorError(
                f"Failed to start/connect to the MCP server: {exc}",
                code="mcp_connection_failed",
            ) from exc

    async def disconnect(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None
        self._stack = None

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def __aenter__(self) -> MCPClient:
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.disconnect()

    async def list_tools(self) -> list[str]:
        if self._session is None:
            raise OrchestratorError(
                "MCP client is not connected.", code="not_connected"
            )
        result = await self._session.list_tools()
        return [t.name for t in result.tools]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout_seconds: int = 30
    ) -> dict[str, Any]:
        if self._session is None:
            raise OrchestratorError(
                "MCP client is not connected.", code="not_connected"
            )
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments=arguments),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise OrchestratorError(
                f"MCP tool '{name}' timed out after {timeout_seconds}s.",
                code="mcp_tool_timeout",
            ) from exc
        except Exception as exc:
            raise OrchestratorError(
                f"MCP tool '{name}' invocation failed: {exc}", code="mcp_tool_failed"
            ) from exc

        return self._parse_tool_result(name, result)

    @staticmethod
    def _parse_tool_result(name: str, result: Any) -> dict[str, Any]:
        # Prefer structured content if the SDK/server version provides it.
        structured = getattr(result, "structuredContent", None)
        if structured:
            return dict(structured)

        if getattr(result, "isError", False):
            text = MCPClient._extract_text(result)
            raise OrchestratorError(
                f"MCP tool '{name}' returned an error: {text}", code="mcp_tool_error"
            )

        text = MCPClient._extract_text(result)
        if not text:
            raise OrchestratorError(
                f"MCP tool '{name}' returned no content.", code="mcp_tool_empty_result"
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise OrchestratorError(
                f"MCP tool '{name}' returned non-JSON content: {text!r}",
                code="mcp_tool_bad_output",
            ) from exc

    @staticmethod
    def _extract_text(result: Any) -> str:
        content = getattr(result, "content", None) or []
        parts = [getattr(c, "text", "") for c in content if getattr(c, "text", None)]
        return "\n".join(parts)


async def run_pipeline(
    prompt: str,
    *,
    llm_client: LLMClient,
    mcp_client: MCPClient,
    critic_enabled: bool = True,
    tool_timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Full pipeline: prompt -> Planner -> (optional) Critic -> MCP tool call.

    Returns a plain dict with keys: success, request (AnalysisRequest),
    warnings (list[str]), and either chart_path/metrics (on success) or
    errors (list[str]) on failure. Never raises for expected failure modes
    (ambiguous prompt, invalid ticker, data-source failure) -- those are
    reported in the returned dict so the API layer can turn them into a
    clean structured response instead of a 500.
    """
    try:
        request: AnalysisRequest = build_analysis_request(prompt, llm_client)
    except PlannerError as exc:
        return {
            "success": False,
            "request": None,
            "warnings": [],
            "errors": [f"Planning failed: {exc}"],
        }

    request, warnings = critic_review(request, llm_client, enabled=critic_enabled)

    if request.needs_clarification:
        return {
            "success": False,
            "request": request,
            "warnings": warnings,
            "errors": [
                request.clarification_message or "The request needs clarification."
            ],
        }

    tool_args = {
        "tickers": request.tickers,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "metric": request.metric,
        "chart_title": None,
    }

    try:
        tool_result = await mcp_client.call_tool(
            "plot_comparison", tool_args, timeout_seconds=tool_timeout_seconds
        )
    except OrchestratorError as exc:
        return {
            "success": False,
            "request": request,
            "warnings": warnings,
            "errors": [exc.message],
        }

    if not tool_result.get("success"):
        return {
            "success": False,
            "request": request,
            "warnings": warnings,
            "errors": [
                tool_result.get("message", "The analysis tool reported a failure.")
            ],
        }

    return {
        "success": True,
        "request": request,
        "warnings": warnings,
        "errors": [],
        "chart_path": tool_result.get("chart_path"),
        "metrics": tool_result.get("metrics"),
    }
