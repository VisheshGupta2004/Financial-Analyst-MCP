"""
Deterministic timeframe resolution.

Design principle (see ARCHITECTURE.md): the LLM is only ever allowed to
classify a user's timeframe phrase into one of a small set of *normalized*
patterns (e.g. "last_6_months", "year:2025", "since:2024-01-15"). The actual
arithmetic that turns that into concrete start/end dates is done here, in
plain deterministic Python, so date math is never left to an LLM to "figure
out" and is fully unit-testable without any LLM or network call.

This module also exposes `sniff_timeframe`, a best-effort regex-based parser
that can extract the same normalized pattern directly from raw natural
language. It is used:
  - as the timeframe extractor for the offline/mock LLM client, and
  - as a deterministic fallback/sanity-check the Critic can use to verify the
    real LLM's `timeframe_raw` output actually resolves to something sane.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta


class TimeframeError(ValueError):
    """Raised when a timeframe pattern cannot be resolved to concrete dates."""


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise TimeframeError(
                f"Resolved start date {self.start} is after end date {self.end}"
            )

    def as_iso(self) -> tuple[str, str]:
        return self.start.isoformat(), self.end.isoformat()


_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})

# Normalized pattern -> resolver. Patterns are deliberately simple strings
# (not free text) so both the LLM and the regex sniffer only ever have to
# produce one of these.
_RELATIVE_UNIT_RE = re.compile(
    r"^last_(\d+)_(day|days|week|weeks|month|months|year|years)$"
)
_YEAR_RE = re.compile(r"^year:(\d{4})$")
_SINCE_RE = re.compile(r"^since:(\d{4})-(\d{2})-(\d{2})$")
_RANGE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}):(\d{4})-(\d{2})-(\d{2})$")


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def resolve_pattern(pattern: str, reference_date: date) -> DateRange:
    """Resolve a normalized timeframe pattern into a concrete DateRange.

    Supported patterns:
      - "last_N_days" / "last_N_weeks" / "last_N_months" / "last_N_years"
      - "ytd"                          -> Jan 1 of reference year .. reference_date
      - "year:YYYY"                    -> full calendar year
      - "since:YYYY-MM-DD"             -> that date .. reference_date
      - "YYYY-MM-DD:YYYY-MM-DD"        -> explicit range, passed straight through
    """
    pattern = pattern.strip().lower()

    if pattern == "ytd":
        return DateRange(date(reference_date.year, 1, 1), reference_date)

    m = _RELATIVE_UNIT_RE.match(pattern)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if n <= 0:
            raise TimeframeError(f"Relative timeframe count must be positive, got {n}")
        if unit.startswith("day"):
            start = reference_date - timedelta(days=n)
        elif unit.startswith("week"):
            start = reference_date - timedelta(weeks=n)
        elif unit.startswith("month"):
            start = _add_months(reference_date, -n)
        else:  # year
            start = _add_months(reference_date, -12 * n)
        return DateRange(start, reference_date)

    m = _YEAR_RE.match(pattern)
    if m:
        year = int(m.group(1))
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        if end > reference_date:
            end = reference_date if start <= reference_date else end
        return DateRange(start, end)

    m = _SINCE_RE.match(pattern)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return DateRange(date(y, mo, d), reference_date)

    m = _RANGE_RE.match(pattern)
    if m:
        y1, mo1, d1, y2, mo2, d2 = (int(x) for x in m.groups())
        return DateRange(date(y1, mo1, d1), date(y2, mo2, d2))

    raise TimeframeError(f"Unrecognized timeframe pattern: {pattern!r}")


# --- Best-effort natural language -> normalized pattern -------------------

_LAST_N_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)\b"
)
_LAST_UNIT_RE = re.compile(r"\b(?:last|past|previous)\s+(day|week|month|year)\b")
_DURING_YEAR_RE = re.compile(r"\b(?:during|in|for|year)\s+(\d{4})\b")
_BARE_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
_SINCE_MONTH_YEAR_RE = re.compile(r"\bsince\s+([A-Za-z]+)\s+(\d{4})\b", re.IGNORECASE)
_SINCE_ISO_RE = re.compile(r"\bsince\s+(\d{4})-(\d{2})-(\d{2})\b", re.IGNORECASE)
_YTD_RE = re.compile(r"\b(ytd|year[\s-]?to[\s-]?date)\b", re.IGNORECASE)

_UNIT_PLURAL = {"day": "days", "week": "weeks", "month": "months", "year": "years"}


def sniff_timeframe(text: str) -> str | None:
    """Best-effort extraction of a normalized timeframe pattern from free text.

    Returns None if nothing recognizable is found (the caller should then
    treat the timeframe as ambiguous and ask for clarification rather than
    guessing).
    """
    text_l = text.lower()

    if _YTD_RE.search(text_l):
        return "ytd"

    m = _LAST_N_RE.search(text_l)
    if m:
        n, unit = m.group(1), m.group(2)
        unit_norm = _UNIT_PLURAL.get(unit.rstrip("s"), unit)
        return f"last_{n}_{unit_norm}"

    m = _LAST_UNIT_RE.search(text_l)
    if m:
        unit = m.group(1)
        return f"last_1_{_UNIT_PLURAL[unit]}"

    m = _SINCE_ISO_RE.search(text_l)
    if m:
        y, mo, d = m.groups()
        return f"since:{y}-{mo}-{d}"

    m = _SINCE_MONTH_YEAR_RE.search(text_l)
    if m:
        month_name, year = m.group(1).lower(), int(m.group(2))
        month = _MONTHS.get(month_name)
        if month:
            return f"since:{year:04d}-{month:02d}-01"

    m = _DURING_YEAR_RE.search(text_l)
    if m:
        return f"year:{m.group(1)}"

    m = _BARE_YEAR_RE.search(text_l)
    if m:
        return f"year:{m.group(1)}"

    return None
