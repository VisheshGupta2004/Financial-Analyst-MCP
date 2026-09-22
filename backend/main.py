"""
FastAPI backend (Phase 4).

Endpoints:
  POST /analyze  -- natural-language prompt in, chart + analysis out.
  GET  /health   -- liveness/readiness, including MCP connectivity.

NOTE ON THIS DELIVERY: requires fastapi, uvicorn, pydantic (not installed in
the sandbox that produced this codebase -- see README). Written against the
documented FastAPI API and exercised only via `python -c` import-shape
review, not a live server -- see README "What Was and Wasn't Executed".
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # must happen before backend.config reads os.environ

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import AppConfig, load_config
from backend.llm_client import LLMError, get_llm_client
from backend.logging_config import configure_logging, set_request_id
from backend.orchestrator import MCPClient, OrchestratorError, run_pipeline
from backend.schemas import AnalyzeApiRequest, AnalyzeApiResponse, HealthResponse

CFG: AppConfig = load_config()
configure_logging(CFG.logging)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(CFG.output_dir, exist_ok=True)
    try:
        app.state.llm_client = get_llm_client(CFG.llm)
    except LLMError as exc:
        logger.error("Failed to initialize LLM client: %s", exc)
        raise

    # A single, long-lived MCP client/subprocess shared across all requests,
    # instead of spawning a fresh subprocess per request (which is correct
    # but adds several seconds of subprocess-startup latency to every call).
    # The lock serializes tool calls, since a single stdio session is not
    # safe for concurrent use.
    app.state.mcp_client = MCPClient(CFG.mcp)
    app.state.mcp_lock = asyncio.Lock()
    try:
        await app.state.mcp_client.connect()
        logger.info("Persistent MCP client connected at startup.")
    except OrchestratorError as exc:
        # Don't crash the whole backend if the MCP server can't start --
        # /health will correctly report degraded, and /analyze will attempt
        # a reconnect on first use (see below).
        logger.error("MCP client failed to connect at startup: %s", exc)

    logger.info(
        "Backend starting with LLM provider=%s, critic_enabled=%s",
        CFG.llm.provider,
        CFG.critic.enabled,
    )
    yield

    await app.state.mcp_client.disconnect()
    logger.info("Backend shutting down.")


app = FastAPI(title="financial-analyst-mcp", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.backend.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(CFG.output_dir, exist_ok=True)
app.mount("/output", StaticFiles(directory=CFG.output_dir), name="output")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = set_request_id(request.headers.get("x-request-id"))
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["x-request-id"] = rid
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    mcp_connected = False
    client: MCPClient = request.app.state.mcp_client
    async with request.app.state.mcp_lock:
        try:
            if not client.is_connected:
                await client.connect()
            await client.list_tools()
            mcp_connected = True
        except OrchestratorError as exc:
            logger.warning("Health check: MCP server not reachable: %s", exc)

    return HealthResponse(
        status="ok" if mcp_connected else "degraded",
        mcp_connected=mcp_connected,
        llm_provider=CFG.llm.provider,
    )


@app.post("/analyze", response_model=AnalyzeApiResponse)
async def analyze(payload: AnalyzeApiRequest, request: Request) -> AnalyzeApiResponse:
    rid = set_request_id(request.headers.get("x-request-id"))
    llm_client = request.app.state.llm_client
    mcp_client: MCPClient = request.app.state.mcp_client

    async with request.app.state.mcp_lock:
        try:
            if not mcp_client.is_connected:
                await mcp_client.connect()
            result = await run_pipeline(
                payload.prompt,
                llm_client=llm_client,
                mcp_client=mcp_client,
                critic_enabled=CFG.critic.enabled,
                tool_timeout_seconds=CFG.mcp.tool_call_timeout_seconds,
            )
        except OrchestratorError as exc:
            # The persistent subprocess may have crashed between requests --
            # try exactly one reconnect-and-retry before giving up, so a
            # single flaky MCP process doesn't take down every request after it.
            logger.warning(
                "MCP call failed for request %s, attempting one reconnect: %s", rid, exc
            )
            try:
                await mcp_client.connect()
                result = await run_pipeline(
                    payload.prompt,
                    llm_client=llm_client,
                    mcp_client=mcp_client,
                    critic_enabled=CFG.critic.enabled,
                    tool_timeout_seconds=CFG.mcp.tool_call_timeout_seconds,
                )
            except OrchestratorError as retry_exc:
                logger.error(
                    "MCP connection failure for request %s after retry: %s",
                    rid,
                    retry_exc,
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Analysis backend unavailable: {retry_exc.message}",
                ) from retry_exc

    if not result["success"]:
        return AnalyzeApiResponse(
            success=False,
            request_id=rid,
            extracted_request=result.get("request"),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
        )

    request_obj = result["request"]
    summary = _build_summary(request_obj, result.get("metrics") or {})

    return AnalyzeApiResponse(
        success=True,
        summary=summary,
        chart_path=result.get("chart_path"),
        request_id=rid,
        extracted_request=request_obj,
        metrics=result.get("metrics"),
        errors=[],
        warnings=result.get("warnings", []),
    )


def _build_summary(request_obj, metrics: dict) -> str:
    tickers = ", ".join(request_obj.tickers)
    lines = [
        f"Analyzed {tickers} from {request_obj.start_date} to {request_obj.end_date} ({request_obj.metric})."
    ]
    for ticker, m in metrics.items():
        lines.append(
            f"{ticker}: total return {m['total_return']:+.1%}, "
            f"CAGR {m['cagr']:+.1%}, "
            f"volatility {m['annualized_volatility']:.1%}, "
            f"max drawdown {m['max_drawdown']:.1%}."
        )
    return " ".join(lines)
