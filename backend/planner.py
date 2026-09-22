"""
Planner Agent (Phase 2).

Responsibility, and *only* responsibility: turn a natural-language prompt
into a validated `AnalysisRequest`. It does not fetch data, does not decide
retry policy, does not render charts, and never generates or executes
arbitrary Python -- it emits structured data that downstream deterministic
code consumes.

The LLM is given today's date and asked to either:
  (a) emit `timeframe_pattern` as one of the normalized patterns described in
      backend/date_utils.py, or
  (b) emit explicit `start_date`/`end_date` if the user gave explicit dates.
Either way, date arithmetic itself happens in date_utils, not in the model.
"""

from __future__ import annotations

import logging
from datetime import date

from pydantic import ValidationError

from backend.llm_client import LLMClient, LLMError
from backend.schemas import AnalysisRequest

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a financial-analysis request parser. Extract structured intent \
from a user's natural-language question about stock performance.

Rules:
- Extract stock ticker symbols. If the user names a company, map it to its ticker \
  (e.g. "Apple" -> "AAPL", "Microsoft" -> "MSFT"). For share-class tickers, use a \
  hyphen, not a dot (e.g. "BRK-B", not "BRK.B").
- For the timeframe, prefer emitting `timeframe_pattern` as one of: \
  "last_N_days" / "last_N_weeks" / "last_N_months" / "last_N_years", "ytd", \
  "year:YYYY", "since:YYYY-MM-DD". Only emit explicit start_date/end_date if the \
  user gave explicit calendar dates.
- metric: "normalized_returns" for comparisons of relative performance, \
  "cumulative_returns" for total-return-over-time questions, "price" for raw price.
- analysis_type: "comparison" if 2+ tickers, else "single".
- If you cannot confidently identify at least one ticker, or the request is \
  ambiguous, set needs_clarification=true and explain why in clarification_message. \
  Do not guess a ticker you are not reasonably confident about.
- Never fabricate a timeframe if none was given or implied; leave timeframe_pattern \
  null and set needs_clarification instead.
"""


class PlannerError(Exception):
    pass


def build_analysis_request(prompt: str, llm_client: LLMClient) -> AnalysisRequest:
    """Run the Planner Agent: prompt -> validated AnalysisRequest.

    Never raises on "the LLM was unsure" -- that becomes a
    needs_clarification=True response, which is a normal, expected outcome.
    Raises PlannerError only for genuine infrastructure failures (LLM call
    failed, or returned something so malformed it can't even be interpreted
    as a clarification request).
    """
    if not prompt or not prompt.strip():
        return AnalysisRequest(
            tickers=[],
            needs_clarification=True,
            clarification_message="Prompt was empty.",
            raw_prompt=prompt,
        )

    user_prompt = (
        f"Today's date is {date.today().isoformat()}.\nUser request: {prompt.strip()}"
    )
    schema = AnalysisRequest.model_json_schema()

    try:
        raw = llm_client.complete_json(
            system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, json_schema=schema
        )
    except LLMError as exc:
        logger.error("Planner LLM call failed: %s", exc)
        raise PlannerError(f"LLM call failed: {exc}") from exc

    raw = dict(raw or {})
    raw["raw_prompt"] = prompt.strip()

    try:
        return AnalysisRequest(**raw)
    except ValidationError as exc:
        logger.warning("Planner LLM output failed schema validation: %s", exc)
        # Degrade to a clarification response rather than crashing the request
        # -- a malformed LLM response should surface as "I couldn't understand
        # that", not as a 500 error.
        return AnalysisRequest(
            tickers=[],
            needs_clarification=True,
            clarification_message=(
                "The request could not be parsed into a valid analysis request. "
                "Please rephrase, naming specific tickers/companies and a timeframe."
            ),
            raw_prompt=prompt.strip(),
        )
