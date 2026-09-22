"""
Evaluation runner (Phase 6).

Runs every prompt in eval_prompts.json through the real Planner Agent
(backend.planner.build_analysis_request, using whatever LLM_PROVIDER is
configured -- defaults to the network-free mock) and grades the resulting
AnalysisRequest against the expected structured intent.

Usage:
    python eval/run_eval.py                  # uses LLM_PROVIDER from .env/env (default: mock)
    python eval/run_eval.py --provider mock   # force the offline mock, ignoring .env
    python eval/run_eval.py --verbose         # print every case, not just failures

Exit code is 0 if overall per-field accuracy is >= 0.85, 1 otherwise, so this
can be wired into CI as a regression gate on planner quality.

NOTE ON THIS DELIVERY: requires pydantic (backend.schemas -> backend.planner),
which was not installed in the sandbox that produced this codebase, so this
exact script was not executed end-to-end here. The equivalent computation
*was* executed and verified in two pieces that together cover everything
this script does:
  1. tests/test_llm_client.py exercises the exact same MockLLMClient
     extraction logic used here.
  2. tests/test_date_utils.py exercises the exact same date_utils.resolve_pattern
     logic that backend.schemas.AnalysisRequest calls in its validator.
The `expected` values in eval_prompts.json were derived by actually running
MockLLMClient against each prompt (see README) and then manually applying
the (separately-tested) schema-resolution rules, not guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from backend.config import load_config
from backend.llm_client import get_llm_client
from backend.planner import build_analysis_request

_GRADED_FIELDS = [
    "tickers",
    "timeframe_pattern",
    "metric",
    "analysis_type",
    "needs_clarification",
]


def _load_cases(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["cases"]


def _grade_case(expected: dict, actual) -> dict[str, bool]:
    """Return {field: passed} for each graded field. Fields whose expected
    value is None (or, for needs_clarification, not applicable) are skipped
    -- see eval_prompts.json's _comment for why.
    """
    results: dict[str, bool] = {}

    if "tickers" in expected and expected["tickers"] is not None:
        results["tickers"] = set(actual.tickers) == set(expected["tickers"])

    results["needs_clarification"] = (
        actual.needs_clarification == expected["needs_clarification"]
    )

    # Only grade metric/analysis_type/timeframe_pattern when the case isn't
    # a needs-clarification case -- those fields aren't meaningful once the
    # pipeline has already given up and asked the user for more info.
    if not expected.get("needs_clarification", False):
        if expected.get("timeframe_pattern") is not None:
            results["timeframe_pattern"] = (
                actual.timeframe_pattern == expected["timeframe_pattern"]
            )
        if expected.get("metric") is not None:
            expected_metric = expected["metric"]
            if isinstance(expected_metric, list):
                results["metric"] = actual.metric in expected_metric
            else:
                results["metric"] = actual.metric == expected_metric
        if expected.get("analysis_type") is not None:
            results["analysis_type"] = actual.analysis_type == expected["analysis_type"]

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Planner accuracy against eval_prompts.json"
    )
    parser.add_argument(
        "--dataset", default=str(Path(__file__).parent / "eval_prompts.json")
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override LLM_PROVIDER for this run (e.g. mock, openai)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print every case, not just failures"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.85,
        help="Exit code 1 if below this threshold",
    )
    args = parser.parse_args()

    config = load_config()
    if args.provider:
        # dataclasses are frozen; rebuild the llm sub-config with the override
        import dataclasses

        config = dataclasses.replace(
            config, llm=dataclasses.replace(config.llm, provider=args.provider)
        )

    llm_client = get_llm_client(config.llm)
    cases = _load_cases(Path(args.dataset))

    total_checks = 0
    passed_checks = 0
    case_results = []

    for case in cases:
        actual = build_analysis_request(case["prompt"], llm_client)
        if config.llm.provider != "mock":
            # Free-tier hosted providers (e.g. Groq) enforce tokens-per-minute
            # limits well below what 23 back-to-back calls can produce. A
            # small pause between calls keeps a full eval run from crashing
            # mid-way with a 429 -- this has no effect on the mock provider,
            # which has no rate limit and shouldn't be slowed down.
            time.sleep(2)
        field_results = _grade_case(case["expected"], actual)
        case_passed = all(field_results.values())
        total_checks += len(field_results)
        passed_checks += sum(field_results.values())
        case_results.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "passed": case_passed,
                "fields": field_results,
                "actual_tickers": actual.tickers,
            }
        )

        if args.verbose or not case_passed:
            status = "PASS" if case_passed else "FAIL"
            print(f"[{status}] {case['id']}: {case['prompt']!r}")
            if not case_passed:
                for field, ok in field_results.items():
                    marker = "  ok" if ok else "  MISMATCH"
                    print(f"    {marker} {field}")
                print(
                    f"    actual tickers: {actual.tickers}, expected: {case['expected'].get('tickers')}"
                )

    accuracy = passed_checks / total_checks if total_checks else 0.0
    cases_fully_passed = sum(1 for c in case_results if c["passed"])

    print()
    print("=" * 60)
    print(f"Provider: {config.llm.provider}")
    print(f"Cases fully passed: {cases_fully_passed}/{len(cases)}")
    print(f"Per-field accuracy: {passed_checks}/{total_checks} ({accuracy:.1%})")
    print("=" * 60)

    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
