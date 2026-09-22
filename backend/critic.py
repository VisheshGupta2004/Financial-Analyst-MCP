"""
Optional Critic/Validator Agent (Phase 2).

This is the one place in the redesigned architecture where a *second* LLM
pass earns its keep: it re-reads the original prompt against the Planner's
structured output and flags things a single-pass extraction can miss --
e.g. the user mentioned three companies but only two tickers were extracted,
or the timeframe interpretation looks implausible for the phrasing used.

It is deliberately narrow: it can only set `needs_clarification`/append a
warning, or leave the request untouched. It never invents new tickers,
never changes dates itself, and never blocks silently -- if disabled
(CriticConfig.enabled=False) or if it fails, the pipeline proceeds with the
Planner's original output and a note that the critic step was skipped.

A cheap, fully deterministic sanity check runs unconditionally before the
optional LLM critique: comparing the Planner's chosen timeframe against
`date_utils.sniff_timeframe` run over the raw prompt, so at least one
critic-style check is always exercised even with LLM_PROVIDER=mock /
CRITIC_ENABLED=false.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.date_utils import sniff_timeframe
from backend.llm_client import LLMClient, LLMError
from backend.schemas import AnalysisRequest

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a careful reviewer checking a financial-analysis request \
extraction for mistakes before it is executed.

You will be given the user's original request and the structured interpretation \
that was extracted from it. Check for:
- Missing tickers the user clearly mentioned.
- A timeframe that doesn't match what the user asked for.
- A metric/analysis type that doesn't match the user's intent.

If everything looks correct, set approved=true with an empty issues list. \
If something looks wrong, set approved=false and list the specific issues in \
plain language. Do not rewrite the request yourself -- only flag problems.
"""


class CriticReview(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    notes: str | None = None


class CriticError(Exception):
    pass


def _deterministic_timeframe_sanity_check(request: AnalysisRequest) -> list[str]:
    """Compare the Planner's chosen timeframe pattern against a fresh,
    independent regex pass over the raw prompt. A mismatch doesn't
    necessarily mean the Planner was wrong (the regex sniffer is simpler
    than the LLM), but it's cheap and catches a real class of errors.
    """
    issues: list[str] = []
    if not request.raw_prompt or request.needs_clarification:
        return issues
    sniffed = sniff_timeframe(request.raw_prompt)
    if sniffed and request.timeframe_pattern and sniffed != request.timeframe_pattern:
        issues.append(
            f"Timeframe mismatch: planner extracted '{request.timeframe_pattern}' but "
            f"a plain-text scan of the prompt suggests '{sniffed}'."
        )
    return issues


def review(
    request: AnalysisRequest, llm_client: LLMClient, *, enabled: bool = True
) -> tuple[AnalysisRequest, list[str]]:
    """Review `request` against its raw prompt. Returns (possibly updated
    request, list of warning strings). Never raises for "review found
    issues" -- only for infrastructure failure, and even then it degrades
    gracefully rather than blocking the pipeline.
    """
    warnings: list[str] = []
    warnings.extend(_deterministic_timeframe_sanity_check(request))

    if not enabled:
        warnings.append("LLM critic step skipped (CRITIC_ENABLED=false).")
        return request, warnings

    if request.needs_clarification:
        # Nothing productive for the critic to review if the planner already
        # gave up -- skip the extra LLM call.
        return request, warnings

    try:
        schema = CriticReview.model_json_schema()
        raw = llm_client.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=(
                f"Original request: {request.raw_prompt}\n\n"
                f"Extracted interpretation: {request.model_dump_json(exclude={'raw_prompt'})}"
            ),
            json_schema=schema,
        )
        review_result = CriticReview(**raw)
    except (
        LLMError,
        Exception,
    ) as exc:
        logger.warning("Critic step failed, proceeding without it: %s", exc)
        warnings.append(f"Critic step failed and was skipped: {exc}")
        return request, warnings

    if not review_result.approved:
        warnings.extend(review_result.issues)
        if review_result.issues:
            request.needs_clarification = True
            request.clarification_message = (
                "The extracted request failed validation review: "
                + "; ".join(review_result.issues)
            )

    return request, warnings
