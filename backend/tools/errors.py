"""Typed errors for the deterministic tools layer.

Kept dependency-free (stdlib only) so both the tools and their tests can run
without pydantic/FastAPI/MCP installed. These are caught and translated into
structured MCP tool error results in mcp_server.py, and into structured HTTP
error responses in main.py -- callers should never see a raw traceback.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for all deterministic-tool failures. Carries a machine
    readable `code` (stable, safe to branch on) plus a human readable message.
    """

    code: str = "tool_error"

    def __init__(
        self, message: str, *, code: str | None = None, details: dict | None = None
    ):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
        }


class InvalidTickerError(ToolError):
    code = "invalid_ticker"


class InvalidDateRangeError(ToolError):
    code = "invalid_date_range"


class DataFetchError(ToolError):
    code = "data_fetch_error"


class NoDataError(ToolError):
    code = "no_data"


class PlottingError(ToolError):
    code = "plotting_error"
