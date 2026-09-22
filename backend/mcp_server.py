"""
MCP Server (Phase 1).

Exposes the deterministic Phase-0 tools (backend/tools/*.py) over the Model
Context Protocol using the official `mcp` Python SDK's FastMCP high-level
API. This server has no knowledge of LLMs, prompts, or planning -- it is a
pure tool-execution surface, callable by *any* MCP client (this project's
own orchestrator.py, Claude Desktop, Cursor, or anything else that speaks
MCP), which is the actual value MCP adds over a hand-rolled function-calling
layer: the tool implementation is decoupled from any particular host.

Run standalone for manual testing:
    python -m backend.mcp_server
or via the official inspector:
    mcp dev backend/mcp_server.py

NOTE ON THIS DELIVERY: this file requires the `mcp` package, which could not
be installed in the sandbox that produced this codebase (no network access).
It is written against the documented FastMCP API. See README "What Was and
Wasn't Executed" for what *was* verified, and how to verify this file
yourself once you have network access.
"""

from __future__ import annotations

import logging
from datetime import date as date_cls

from backend.config import load_config
from backend.logging_config import configure_logging
from backend.tools import analysis, data, plotting
from backend.tools.errors import ToolError

configure_logging()
logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without `mcp` installed
    raise SystemExit(
        "The 'mcp' package is not installed. Install project dependencies first:\n"
        "    pip install -r requirements.txt\n"
        f"(original import error: {exc})"
    )

mcp = FastMCP("financial-analyst-mcp")

_CFG = load_config()


def _parse_date(value: str, field_name: str) -> date_cls:
    try:
        return date_cls.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(
            f"Invalid {field_name}: {value!r} is not an ISO date (YYYY-MM-DD)."
        ) from exc


@mcp.tool()
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> dict:
    """Fetch daily OHLCV price data for a single ticker between two ISO dates
    (inclusive). Returns the raw records plus headline metrics (total return,
    CAGR, annualized volatility, max drawdown) for that ticker over the range.
    """
    try:
        start, end = _parse_date(start_date, "start_date"), _parse_date(
            end_date, "end_date"
        )
        df = data.fetch_stock_data(
            ticker,
            start,
            end,
            max_retries=_CFG.data.max_retries,
            retry_backoff_seconds=_CFG.data.retry_backoff_seconds,
        )
        metrics = analysis.basic_metrics(df)
        records = [
            {"date": idx.date().isoformat(), "close": float(row["Close"])}
            for idx, row in df.iterrows()
        ]
        return {
            "success": True,
            "ticker": data.validate_ticker(ticker),
            "records": records,
            "metrics": metrics,
        }
    except ToolError as exc:
        logger.info("fetch_stock_data tool error: %s", exc.to_dict())
        return {"success": False, **exc.to_dict()}


@mcp.tool()
def calculate_returns(
    ticker: str, start_date: str, end_date: str, kind: str = "cumulative"
) -> dict:
    """Calculate a return series for a single ticker.

    `kind` is one of "daily" (day-over-day % change) or "cumulative"
    (cumulative % return since the start of the range).
    """
    try:
        start, end = _parse_date(start_date, "start_date"), _parse_date(
            end_date, "end_date"
        )
        df = data.fetch_stock_data(
            ticker,
            start,
            end,
            max_retries=_CFG.data.max_retries,
            retry_backoff_seconds=_CFG.data.retry_backoff_seconds,
        )
        if kind == "daily":
            series = analysis.daily_returns(df)
        elif kind == "cumulative":
            series = analysis.cumulative_returns(df)
        else:
            raise ToolError(
                f"Unknown return kind: {kind!r}. Use 'daily' or 'cumulative'.",
                code="bad_input",
            )
        records = [
            {"date": idx.date().isoformat(), "value": float(v)}
            for idx, v in series.items()
        ]
        return {
            "success": True,
            "ticker": data.validate_ticker(ticker),
            "kind": kind,
            "records": records,
        }
    except ToolError as exc:
        logger.info("calculate_returns tool error: %s", exc.to_dict())
        return {"success": False, **exc.to_dict()}


@mcp.tool()
def compute_comparison(
    tickers: list[str],
    start_date: str,
    end_date: str,
    metric: str = "normalized_returns",
) -> dict:
    """Fetch and align multiple tickers onto a shared date index for
    comparison, without producing a chart. `metric` is one of
    "normalized_returns", "cumulative_returns", or "price".
    """
    try:
        start, end = _parse_date(start_date, "start_date"), _parse_date(
            end_date, "end_date"
        )
        frames = data.fetch_multiple(
            tickers,
            start,
            end,
            max_retries=_CFG.data.max_retries,
            retry_backoff_seconds=_CFG.data.retry_backoff_seconds,
        )
        combined = analysis.compute_comparison(frames, metric=metric)
        records = [
            {
                "date": idx.date().isoformat(),
                **{col: float(val) for col, val in row.items()},
            }
            for idx, row in combined.iterrows()
        ]
        metrics = {ticker: analysis.basic_metrics(df) for ticker, df in frames.items()}
        return {
            "success": True,
            "metric": metric,
            "records": records,
            "metrics": metrics,
        }
    except ToolError as exc:
        logger.info("compute_comparison tool error: %s", exc.to_dict())
        return {"success": False, **exc.to_dict()}


@mcp.tool()
def plot_comparison(
    tickers: list[str],
    start_date: str,
    end_date: str,
    metric: str = "normalized_returns",
    chart_title: str | None = None,
) -> dict:
    """Fetch, compute, and plot a comparison chart for one or more tickers.
    This is the primary end-to-end tool: a single call fetches all requested
    tickers, aligns them, renders a PNG chart, and returns the chart's file
    path plus headline metrics for each ticker -- everything the frontend
    needs for one user request.
    """
    try:
        start, end = _parse_date(start_date, "start_date"), _parse_date(
            end_date, "end_date"
        )
        frames = data.fetch_multiple(
            tickers,
            start,
            end,
            max_retries=_CFG.data.max_retries,
            retry_backoff_seconds=_CFG.data.retry_backoff_seconds,
        )
        combined = analysis.compute_comparison(frames, metric=metric)
        chart_path = plotting.plot_comparison(
            combined, metric=metric, title=chart_title, output_dir=_CFG.output_dir
        )
        metrics = {ticker: analysis.basic_metrics(df) for ticker, df in frames.items()}
        return {
            "success": True,
            "chart_path": chart_path,
            "metric": metric,
            "tickers": list(frames.keys()),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "metrics": metrics,
        }
    except ToolError as exc:
        logger.info("plot_comparison tool error: %s", exc.to_dict())
        return {"success": False, **exc.to_dict()}


if __name__ == "__main__":
    mcp.run()
