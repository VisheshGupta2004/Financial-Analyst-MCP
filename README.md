# financial-analyst-mcp

Natural language in, a real stock comparison chart out — via a Planner
Agent (LLM), Pydantic-validated structured output, an optional Critic
agent, and a real MCP client/server pair driving deterministic
yfinance/pandas/Matplotlib tools. **No Cursor, no arbitrary LLM-generated
code execution.**

> This project is a from-scratch redesign of a reference "Multi-Agent
> Financial Analyst (MCP)" project that used CrewAI + 3 agents +
> Cursor-IDE-as-MCP-host. This version removes the Cursor dependency
> entirely (its MCP client couldn't be driven programmatically), removes
> arbitrary code generation/execution, and reduces the agent count from
> three to two (Planner + optional Critic), with the reasoning for each
> change documented in `ARCHITECTURE.md`.

---

## What it does

```
"Compare Apple's stock performance with Microsoft's over the last 6 months
 and plot their normalized returns."
        │
        ▼
  Planner Agent (LLM) → tickers=[AAPL, MSFT], timeframe=last_6_months, metric=normalized_returns
        │
        ▼
  Pydantic validation + deterministic date resolution
        │
        ▼
  Optional Critic pass (re-checks the extraction against the raw prompt)
        │
        ▼
  MCP Client → MCP Server → fetch (yfinance) → compute (pandas) → plot (Matplotlib)
        │
        ▼
  A real PNG chart + return/volatility/drawdown metrics, back in the UI
```

## Why MCP, and why not Cursor

MCP's actual value here is **host interoperability**: `backend/mcp_server.py`
is a standalone process exposing tools over a standard protocol, callable by
this app's own client, by Claude Desktop, or by anything else that speaks
MCP — without touching the tool implementations. For a single app in
isolation, plain function calls would work identically; MCP is what makes
the tool surface reusable outside this one app.

Cursor was dropped entirely. In the reference project, Cursor served as the
MCP *host* — but Cursor's MCP client is built for a human typing into an
IDE chat panel, not for a Python agent pipeline to invoke programmatically.
There is no clean way for `orchestrator.py` to "call Cursor." This project
ships its own ~150-line MCP client (`backend/orchestrator.py`) using the
official `mcp` SDK instead, so the whole system runs standalone, with zero
IDE dependency, for anyone who clones it. Full reasoning in
`ARCHITECTURE.md`.

## How the Planner works

The Planner (`backend/planner.py`) sends the user's prompt plus today's date
to an LLM and asks it to emit a JSON object matching the `AnalysisRequest`
Pydantic schema (`backend/schemas.py`): tickers, a *normalized* timeframe
pattern (e.g. `"last_6_months"`, `"year:2025"`, `"since:2024-01-01"` — never
raw date arithmetic), a metric, and an analysis type. Pydantic then
deterministically resolves that pattern into concrete `start_date`/
`end_date` via `backend/date_utils.py` — **date math is never left to the
LLM**. If the LLM can't confidently identify a ticker or timeframe, the
request comes back with `needs_clarification=true` instead of a guess.

## How MCP client/server communication works

`backend/mcp_server.py` registers four tools (`fetch_stock_data`,
`calculate_returns`, `compute_comparison`, `plot_comparison`) using the
official `mcp` SDK's `FastMCP` API, each wrapping a deterministic function
in `backend/tools/`. `backend/orchestrator.py`'s `MCPClient` launches that
server as a subprocess over stdio, calls `list_tools()` to discover what's
available, and `call_tool(name, arguments)` to invoke one, with a timeout
and structured error translation (`OrchestratorError`) at every step.

## Available tools

| Tool | Purpose |
|---|---|
| `fetch_stock_data(ticker, start_date, end_date)` | Raw OHLCV + headline metrics for one ticker |
| `calculate_returns(ticker, start_date, end_date, kind)` | Daily or cumulative return series for one ticker |
| `compute_comparison(tickers, start_date, end_date, metric)` | Multi-ticker aligned comparison data, no chart |
| `plot_comparison(tickers, start_date, end_date, metric, chart_title)` | The primary end-to-end tool: fetch + compute + render a PNG in one call |

---

## Installation

```bash
git clone <this repo>
cd financial-analyst-mcp
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment setup

```bash
cp .env.example .env
```

By default `LLM_PROVIDER=mock`, which needs **no API key and no network** —
see "Offline mock mode" below. To use a real LLM, set `LLM_PROVIDER=openai`
or `LLM_PROVIDER=anthropic` and fill in `LLM_API_KEY` in `.env`.

## How to start the MCP server

You normally don't need to start this yourself — `backend/orchestrator.py`
launches it automatically as a subprocess per request. To run/inspect it
standalone:

**macOS/Linux/WSL/Git Bash:**
```bash
./scripts/run_mcp_server.sh
# or, with the official inspector UI:
mcp dev backend/mcp_server.py
```

**Windows PowerShell** (`.sh` scripts won't run natively in PowerShell —
use the `.ps1` equivalents, or run the commands directly):
```powershell
.\scripts\run_mcp_server.ps1
# or directly:
python -m backend.mcp_server
```
If PowerShell blocks script execution, either run once per session:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, or just
run the underlying command shown above directly instead of the `.ps1` file.

## How to start the FastAPI backend

**macOS/Linux/WSL/Git Bash:**
```bash
./scripts/run_backend.sh
```

**Windows PowerShell:**
```powershell
.\scripts\run_backend.ps1
# or directly:
if (!(Test-Path ".env")) { Copy-Item ".env.example" ".env" }
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Note: there is no route at `/` — visiting `http://127.0.0.1:8000/` will
correctly show `{"detail":"Not Found"}`. Use these instead:
- `http://127.0.0.1:8000/health` — status check
- `http://127.0.0.1:8000/docs` — interactive API docs, lets you try `/analyze` directly

## How to start Streamlit

In a second terminal, with the backend already running:

**macOS/Linux/WSL/Git Bash:**
```bash
./scripts/run_frontend.sh
```

**Windows PowerShell:**
```powershell
.\scripts\run_frontend.ps1
# or directly:
streamlit run frontend/app.py
```

Visit the URL Streamlit prints (typically `http://localhost:8501`).

## How to run tests

**macOS/Linux/WSL/Git Bash:**
```bash
pip install -r requirements.txt   # needed for the full suite, see below
./scripts/run_tests.sh
```

**Windows PowerShell:**
```powershell
pip install -r requirements.txt
.\scripts\run_tests.ps1
# or directly:
python -m pytest tests\ -v
```

Every test file also runs with plain stdlib unittest, no pytest required:
```bash
python -m unittest discover -s tests -v
```

## How to run evaluations

```bash
python eval/run_eval.py                    # offline, uses LLM_PROVIDER=mock by default
python eval/run_eval.py --provider openai   # grade a real LLM provider instead
python eval/run_eval.py --verbose
```

---

## Offline mock mode

`LLM_PROVIDER=mock` (the default) uses a deterministic, regex/keyword-based
stand-in LLM (`backend.llm_client.MockLLMClient`) that requires no API key
and no network access. This exists for three real reasons, not just as a
test double:

1. **Zero-friction local development and CI** — no API credits burned, no
   network flakiness.
2. **A genuinely usable offline demo** — clone, `pip install`, run, and the
   whole pipeline (Planner → Critic → MCP → tools → chart) works end to end
   with zero external accounts, aside from yfinance itself needing network
   access to fetch real market data.
3. It's what let this repository's own automated tests and evaluation
   dataset (`eval/eval_prompts.json`) be exercised without any external
   dependency.

It is intentionally limited: a fixed dictionary of ~13 well-known company
names, and regex-based timeframe/metric detection. See "Known Limitations."

---

## Example prompts

```
Compare Apple's stock performance with Microsoft's over the last 6 months and plot their normalized returns.
Show me Tesla's performance during 2025.
Compare AAPL, MSFT and NVDA over the past 6 months.
Plot normalized returns for Tesla and Apple since January 2024.
Compare Google and Amazon cumulative returns since 2023-01-01.
```

## Expected output

A JSON response from `POST /analyze` shaped like:

```json
{
  "success": true,
  "summary": "Analyzed AAPL, MSFT from 2026-02-21 to 2026-08-21 (normalized_returns). AAPL: total return +18.4%, ...",
  "chart_path": "output/comparison_ab12cd34ef.png",
  "request_id": "a1b2c3d4e5f6",
  "extracted_request": { "tickers": ["AAPL", "MSFT"], "start_date": "2026-02-21", "...": "..." },
  "metrics": { "AAPL": { "total_return": 0.184, "cagr": 0.41, "annualized_volatility": 0.28, "max_drawdown": -0.11 }, "...": "..." },
  "errors": [],
  "warnings": []
}
```

...and, in Streamlit, that same information rendered as a summary, a line
chart, and a metrics table.

---

## What Was and Wasn't Executed (read this before judging "does it work")

This codebase was written and tested in a sandboxed environment **with no
network access whatsoever** (verified: `pip install` and raw HTTP requests
both fail there) and only `pandas`/`numpy`/`matplotlib` pre-installed.
Being upfront about exactly what that means:

**Fully executed and verified in that sandbox (82 automated tests, all
passing, plus real generated PNG files inspected):**
- `backend/date_utils.py` — 22 tests, including month-end edge cases
- `backend/tools/data.py` — 18 tests using a fake yfinance module
  (dependency-injected), covering retries, empty results, malformed
  ticker rejection, and a real bug this caught and fixed (`fetch_multiple`
  was keying results by unnormalized ticker strings)
- `backend/tools/analysis.py` — 19 tests of the financial math against
  hand-computed expected values
- `backend/tools/plotting.py` — 6 tests that generate real PNG files and
  verify them
- `backend/llm_client.py`'s `MockLLMClient` — 9 tests, including two real
  bugs this caught and fixed (the bare-ticker regex was picking up "YTD"
  and "S"/"P" from "S&P" as fake stock tickers)
- `backend/config.py` and `backend/logging_config.py` — 8 tests

**Written completely (no TODOs, no stubs, no placeholders) against the
documented, real APIs, but not executable in that no-network sandbox**
because they require `pydantic`, `fastapi`, `mcp`, or `streamlit`, none of
which could be installed:
- `backend/schemas.py`, `backend/planner.py`, `backend/critic.py`
- `backend/mcp_server.py`, `backend/orchestrator.py`
- `backend/main.py` (FastAPI), `frontend/app.py` (Streamlit)
- `tests/test_planner.py`, `tests/test_mcp.py`, `tests/test_api.py` — these
  are written and ready to run; they were not executed here, but the logic
  they exercise (mock extraction, date resolution) *was* independently
  verified via the modules above
- `eval/run_eval.py` — written and ready to run; the ground-truth `expected`
  values in `eval_prompts.json` were derived by actually running
  `MockLLMClient` against each prompt in the sandbox and then applying the
  separately-tested schema-resolution rules by hand, not guessed

**To close this gap yourself:** `pip install -r requirements.txt` (needs
network) and run `./scripts/run_tests.sh`. Everything above should pass; if
anything doesn't, it's a real bug for you to report/fix, not an
intentional gap — the design has been reasoned through carefully, but I
have not had the chance to execute the pydantic/FastAPI/MCP layers myself.

---

## Known limitations

- **`MockLLMClient`'s company dictionary is small** (~13 names). Uncommon
  company names will fail to resolve and correctly trigger
  `needs_clarification` rather than silently guessing — but this means the
  offline mode has real, documented gaps (see `eval/eval_prompts.json`
  case `p23`, included deliberately to demonstrate this rather than hide
  it). A real LLM provider does not have this limitation.
- **No support for index tickers** (S&P 500, Dow, etc.) — yfinance can
  fetch these via tickers like `^GSPC`, but nothing in this system maps
  natural-language index names to those symbols yet.
- **No quarter-based timeframes** ("last quarter", "Q3 2024") are in the
  supported timeframe vocabulary yet — these correctly trigger
  clarification rather than a wrong guess, but adding quarter support to
  `date_utils.py` would be a small, contained change.
- **`compute_comparison`'s inner-join alignment** trims to the shortest
  overlapping trading history across tickers — a very recently-IPO'd
  company compared against a decades-old one will only compare over the
  newer company's whole lifetime, which is correct but not necessarily
  obvious from the output alone.
- **The MCP server and client run as a fresh subprocess per `/analyze`
  request** (see `orchestrator.py`'s `async with MCPClient(...)`) rather
  than a persistent long-lived connection, which is simpler and more
  robust for a single-instance demo but adds subprocess-startup latency to
  every request. A production version would likely pool a persistent MCP
  session instead.

## Future improvements

- Persistent MCP session pooling instead of per-request subprocess launch.
- Quarter/half-year timeframe support in `date_utils.py`.
- A larger, LLM-maintained company→ticker lookup (or a real symbol-search
  API) instead of the mock's fixed dictionary.
- Multi-step analysis chains (e.g. "compare X and Y, then tell me which had
  lower volatility") requiring the Planner to sequence multiple tool calls
  rather than always calling `plot_comparison` once.
- A measured comparison of Planner accuracy *with* vs. *without* the Critic
  step on a shared eval set, to quantify whether the Critic is actually
  earning its cost.
- Response caching (`CacheConfig` is already wired into `config.py` but not
  yet consumed by `tools/data.py` — a natural next step, keyed on
  ticker+date range).
- Observability/tracing across the full pipeline (currently: structured
  logs with request-ID correlation, but no span-level tracing).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not reach the backend` in Streamlit | FastAPI isn't running, or `BACKEND_URL` mismatch | Start it with `./scripts/run_backend.sh`; check `BACKEND_URL` in `.env` matches |
| `/health` returns `"status": "degraded"` | The MCP server subprocess failed to start | Run `./scripts/run_mcp_server.sh` directly to see the real error; check `MCP_SERVER_COMMAND`/`MCP_SERVER_ARGS` in `.env` |
| `LLM_API_KEY is not set` error | `LLM_PROVIDER` is `openai`/`anthropic` but no key in `.env` | Add a real key, or switch to `LLM_PROVIDER=mock` |
| `No trading data found for ticker` | Ticker doesn't exist, is delisted, or has no data in the requested range | Check spelling; try a shorter/more recent range |
| Tests in `test_planner.py`/`test_mcp.py`/`test_api.py` are skipped | Missing `pydantic`/`mcp`/`fastapi` | `pip install -r requirements.txt` |
| Chart doesn't render in Streamlit | Backend's `OUTPUT_DIR` isn't reachable at `/output/...` | Confirm `output/` exists and the backend mounted it (it does, in `main.py`, at startup) |
| `mcp` still shows as skipped after `pip install -r requirements.txt` | Check `pip show mcp` — if not found, likely a version-compatibility gap between `mcp` and a very new Python version (e.g. 3.14) | `pip install mcp` directly and read the error; if it's a compatibility issue, create the venv with Python 3.12 or 3.13 instead: `py -3.12 -m venv .venv` |
| `.sh` scripts don't run in Windows PowerShell | PowerShell doesn't execute bash scripts natively | Use the `.ps1` equivalents in `scripts/` (e.g. `.\scripts\run_backend.ps1`), or run the underlying command shown inside each script directly |
| PowerShell refuses to run `.ps1` scripts ("running scripts is disabled") | Default PowerShell execution policy | Run once per session: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then retry |
| `http://127.0.0.1:8000/` shows `{"detail":"Not Found"}` | Expected — there is no route at `/` | Use `/health` or `/docs` instead |

---

## License

Provided as-is for portfolio/educational use.
