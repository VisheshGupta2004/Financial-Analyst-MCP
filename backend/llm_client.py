"""
LLM provider abstraction.

Every provider implements the same tiny interface: given a system prompt, a
user prompt, and a JSON schema, return a dict matching that schema. This
keeps `planner.py` and `critic.py` provider-agnostic and easy to test.

Providers:
  - MockLLMClient: a deterministic, regex/keyword-based "LLM" that requires
    no network and no API key. It exists for three legitimate reasons, not
    just as a test double:
      1. Local development and CI without burning API credits.
      2. A genuinely usable offline demo mode (LLM_PROVIDER=mock).
      3. It's what lets this repository's own test suite and eval runner
         exercise the full planner -> MCP -> tools pipeline end-to-end
         without any external dependency.
    It is intentionally limited (a fixed company-name/ticker lookup, regex
    timeframe/metric detection) and will correctly report "needs
    clarification" for anything it can't confidently parse, rather than
    guessing -- the same contract a real LLM provider must honor.
  - OpenAICompatibleClient: uses the `openai` SDK's structured-output
    (JSON schema) mode. Works with OpenAI directly and any OpenAI-compatible
    endpoint (e.g. a local vLLM/Ollama server) via LLM_BASE_URL.
  - AnthropicClient: uses the `anthropic` SDK's tool-use mode to force a
    single structured tool call matching the schema.

NOTE ON THIS DELIVERY: OpenAICompatibleClient and AnthropicClient are
implemented against the real, documented SDKs but could not be executed in
the sandbox that produced this codebase (no network access, no API key --
see README "What Was and Wasn't Executed"). MockLLMClient is fully exercised
by the automated test suite and the eval runner.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from backend.config import LLMConfig
from backend.date_utils import sniff_timeframe

# A small illustrative company-name -> ticker lookup for the mock client.
# A real LLM provider does not need this (it already "knows" tickers); this
# table exists purely so the offline mock can resolve common company names
# without a network lookup.
_COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "berkshire": "BRK-B",
    "intel": "INTC",
    "amd": "AMD",
}

_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}(?:[.\-][A-Z]{1,2})?\b")
_METRIC_KEYWORDS = {
    "normalized_returns": ["normalized", "normalize", "rebased"],
    "cumulative_returns": ["cumulative", "total return", "returns over"],
}


class LLMError(Exception):
    """Raised when an LLM call fails or returns something unparseable."""


class LLMClient(ABC):
    @abstractmethod
    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a dict that should conform to json_schema."""
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic, network-free stand-in LLM. See module docstring."""

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        # The mock only implements the AnalysisRequest extraction task; it
        # inspects the schema's declared properties to decide what shape of
        # response to produce, so it degrades gracefully (returns an empty
        # dict) for any schema it doesn't specifically recognize -- e.g. the
        # Critic's review schema, which callers should handle explicitly.
        props = set(json_schema.get("properties", {}).keys())
        if {"tickers", "timeframe_pattern"}.issubset(props):
            return self._extract_analysis_request(user_prompt)
        if {"approved"}.issubset(props):
            return {
                "approved": True,
                "issues": [],
                "notes": "mock critic: no review performed",
            }
        return {}

    def _extract_analysis_request(self, text: str) -> dict[str, Any]:
        # planner.py prefixes the user's actual request with a "Today's date
        # is YYYY-MM-DD" line so a real LLM knows how to resolve relative
        # timeframes. That's correct for a real LLM, but this mock's naive
        # regex-based year/timeframe sniffing must NOT scan that date stamp
        # itself, or it will "detect" a fake timeframe from today's date
        # rather than from anything the user actually said. Extract just the
        # user's request portion when the marker is present.
        match = re.search(r"user request:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        request_text = match.group(1).strip() if match else text
        text_lower = request_text.lower()

        tickers: list[str] = []
        for name, ticker in _COMPANY_TO_TICKER.items():
            if re.search(rf"\b{re.escape(name)}\b", text_lower):
                if ticker not in tickers:
                    tickers.append(ticker)
        # Also catch bare uppercase tokens that look like tickers already
        # (e.g. "AAPL, MSFT and NVDA"), skipping common English words.
        # Minimum length 2 avoids false positives on stray single letters
        # picked up from things like "S&P 500" (-> "S", "P").
        stopwords = {
            "I",
            "A",
            "THE",
            "OF",
            "OVER",
            "AND",
            "VS",
            "TO",
            "IN",
            "FOR",
            "SHOW",
            "ME",
            "PLOT",
            "YTD",
            "US",
            "IS",
        }
        for token in _TICKER_TOKEN_RE.findall(request_text):
            if token not in stopwords and token not in tickers and 2 <= len(token) <= 5:
                tickers.append(token)

        timeframe_pattern = sniff_timeframe(request_text)

        metric = "normalized_returns" if len(tickers) > 1 else "price"
        for candidate_metric, keywords in _METRIC_KEYWORDS.items():
            if any(k in text_lower for k in keywords):
                metric = candidate_metric
                break

        result: dict[str, Any] = {
            "tickers": tickers,
            "timeframe_pattern": timeframe_pattern,
            "metric": metric,
            "analysis_type": "comparison" if len(tickers) > 1 else "single",
            "chart_type": "line",
            "needs_clarification": False,
            "clarification_message": None,
        }
        if not tickers:
            result["needs_clarification"] = True
            result["clarification_message"] = (
                "Could not identify any stock tickers or company names in the request."
            )
        return result


def _make_json_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively enforce OpenAI/Groq strict-mode structured-output rules on
    every object schema node:
      1. additionalProperties=false
      2. every key in "properties" must also appear in "required" -- strict
         mode expresses optionality via the field's type permitting null
         (e.g. an "anyOf" with a null branch), not via omission from
         "required". Pydantic's model_json_schema() only lists fields
         without defaults as required, so this must be patched in for every
         object node (top-level and any nested under "$defs").
    Both providers reject the request with a 400 error if either rule is
    violated anywhere in the schema.
    """
    if not isinstance(schema, dict):
        return schema

    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())

    for key in ("properties", "$defs", "definitions"):
        if key in schema and isinstance(schema[key], dict):
            for value in schema[key].values():
                _make_json_schema_strict(value)

    for key in ("items", "anyOf", "allOf", "oneOf"):
        if key in schema:
            value = schema[key]
            if isinstance(value, dict):
                _make_json_schema_strict(value)
            elif isinstance(value, list):
                for item in value:
                    _make_json_schema_strict(item)

    return schema


class OpenAICompatibleClient(LLMClient):
    """Structured-output client for OpenAI and OpenAI-compatible endpoints.

    Requires the `openai` package (see requirements.txt) and network access.
    Not executed in the environment that produced this repository -- see
    README for details.
    """

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            raise LLMError(
                "LLM_API_KEY is not set. Set it in your .env, or set "
                "LLM_PROVIDER=mock to run without a real LLM."
            )
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is required for LLM_PROVIDER=openai. "
                "Install it with: pip install openai"
            ) from exc
        self._config = config

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url or None,
            timeout=self._config.timeout_seconds,
        )
        strict_schema = _make_json_schema_strict(dict(json_schema))
        response = client.chat.completions.create(
            model=self._config.model,
            temperature=self._config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": strict_schema,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"LLM returned non-JSON content: {content!r}") from exc


class AnthropicClient(LLMClient):
    """Structured-output client for Anthropic's Messages API, using a single
    forced tool call to get schema-conformant JSON back.

    Requires the `anthropic` package (see requirements.txt) and network
    access. Not executed in the environment that produced this repository.
    """

    def __init__(self, config: LLMConfig):
        if not config.api_key:
            raise LLMError(
                "LLM_API_KEY is not set. Set it in your .env, or set "
                "LLM_PROVIDER=mock to run without a real LLM."
            )
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise LLMError(
                "The 'anthropic' package is required for LLM_PROVIDER=anthropic. "
                "Install it with: pip install anthropic"
            ) from exc
        self._config = config

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic(
            api_key=self._config.api_key, timeout=self._config.timeout_seconds
        )
        tool_name = "emit_structured_response"
        response = client.messages.create(
            model=self._config.model,
            max_tokens=1024,
            temperature=self._config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Emit the structured response.",
                    "input_schema": json_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return block.input
        raise LLMError("Anthropic response did not include the expected tool call.")


def get_llm_client(config: LLMConfig) -> LLMClient:
    """Factory: build the right LLMClient implementation for config.provider."""
    provider = config.provider.lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        return OpenAICompatibleClient(config)
    if provider == "anthropic":
        return AnthropicClient(config)
    raise LLMError(
        f"Unknown LLM_PROVIDER '{config.provider}'. Supported: mock, openai, anthropic."
    )
