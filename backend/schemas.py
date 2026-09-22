"""
Pydantic (v2) models -- the structured boundary between the LLM's free-text
interpretation and every deterministic component downstream.

Design note: the LLM does NOT emit raw start_date/end_date directly for
relative phrases ("last 6 months"). It emits `timeframe_pattern`, one of a
small closed vocabulary of normalized strings (see backend/date_utils.py),
OR explicit ISO dates if the user gave explicit dates. `AnalysisRequest`'s
model_validator then deterministically resolves `timeframe_pattern` into
concrete `start_date`/`end_date` using backend.date_utils.resolve_pattern --
date arithmetic is never left to the LLM.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.date_utils import DateRange, TimeframeError, resolve_pattern
from backend.tools.data import validate_ticker
from backend.tools.errors import InvalidTickerError

Metric = Literal["price", "normalized_returns", "cumulative_returns"]
AnalysisType = Literal["single", "comparison"]
ChartType = Literal["line"]


class AnalysisRequest(BaseModel):
    """Structured intent extracted from a user's natural-language prompt."""

    tickers: list[str] = Field(default_factory=list, max_length=8)
    timeframe_pattern: str | None = Field(
        default=None,
        description=(
            "Normalized timeframe pattern, e.g. 'last_6_months', 'year:2025', "
            "'since:2024-01-01', 'ytd'. Mutually resolvable with explicit dates."
        ),
    )
    start_date: date | None = None
    end_date: date | None = None
    metric: Metric = "normalized_returns"
    analysis_type: AnalysisType = "single"
    chart_type: ChartType = "line"

    needs_clarification: bool = False
    clarification_message: str | None = None

    # Free-text field kept for logging/debugging/eval -- never consumed by
    # downstream deterministic code, only humans/logs.
    raw_prompt: str | None = None

    @field_validator("tickers")
    @classmethod
    def _normalize_tickers(cls, tickers: list[str]) -> list[str]:
        normalized = []
        for t in tickers:
            try:
                normalized.append(validate_ticker(t))
            except InvalidTickerError as exc:
                raise ValueError(str(exc)) from exc
        # de-duplicate while preserving order
        seen = set()
        deduped = []
        for t in normalized:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    @model_validator(mode="after")
    def _resolve_timeframe_and_infer_analysis_type(self) -> AnalysisRequest:
        if not self.tickers:
            self.needs_clarification = True
            self.clarification_message = self.clarification_message or (
                "Could not identify any stock tickers or company names in the request."
            )
            return self

        # If explicit dates were already given (e.g. user said an ISO range),
        # trust them and skip pattern resolution.
        if self.start_date is None or self.end_date is None:
            if self.timeframe_pattern:
                try:
                    resolved: DateRange = resolve_pattern(self.timeframe_pattern, date.today())
                except TimeframeError as exc:
                    # Don't hard-fail the whole request -- surface as a
                    # clarification instead, which the API can act on cleanly.
                    self.needs_clarification = True
                    self.clarification_message = (
                        f"Could not resolve timeframe '{self.timeframe_pattern}': {exc}"
                    )
                    return self
                self.start_date, self.end_date = resolved.start, resolved.end
            elif self.start_date is not None and self.end_date is None:
                # An explicit start date with no end date and no pattern is a
                # common, unambiguous case ("since 2024-05-10", "from March
                # onward") -- an open-ended range naturally means "through
                # today", so default it rather than asking for clarification
                # on something that was never actually ambiguous.
                self.end_date = date.today()
            elif not self.needs_clarification:
                self.needs_clarification = True
                self.clarification_message = (
                    "No timeframe was specified or recognized. Please include a "
                    "date range, e.g. 'last 6 months', 'in 2025', or explicit dates."
                )
                return self

        if self.start_date and self.end_date and self.start_date > self.end_date:
            self.needs_clarification = True
            self.clarification_message = (
                f"Start date {self.start_date} is after end date {self.end_date}."
            )

        # Infer analysis_type from ticker count if caller didn't set it explicitly
        # (Pydantic already applied the "single" default, so this only overrides
        # when there's more than one ticker but the default was left as-is).
        if len(self.tickers) > 1:
            object.__setattr__(self, "analysis_type", "comparison")
        return self


class AnalyzeApiRequest(BaseModel):
    """Request body for POST /analyze."""

    prompt: str = Field(..., min_length=1, max_length=500)


class AnalyzeApiResponse(BaseModel):
    """Response body for POST /analyze."""

    success: bool
    summary: str | None = None
    chart_path: str | None = None
    request_id: str
    extracted_request: AnalysisRequest | None = None
    metrics: dict | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mcp_connected: bool
    llm_provider: str
