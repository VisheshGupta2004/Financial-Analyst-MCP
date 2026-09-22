# Architecture

## Diagram

```
┌─────────────────────────┐
│      Streamlit UI         │  frontend/app.py
│  (prompt in, chart out)   │  -- talks HTTP only, no business logic
└────────────┬───────────────┘
             │ POST /analyze  {prompt}
             ▼
┌─────────────────────────────┐
│     FastAPI Backend           │  backend/main.py
│  POST /analyze, GET /health   │  -- request-ID logging, CORS, error mapping
└────────────┬───────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Planner Agent (LLM)          │  backend/planner.py
│  NL prompt → AnalysisRequest   │  -- the ONLY agent that reads free text
│  (Pydantic-validated)          │  backend/schemas.py
└────────────┬───────────────────┘
             │ structured request
             ▼
┌─────────────────────────────┐
│ Critic / Validator (optional)  │  backend/critic.py
│ - deterministic timeframe       │  -- always runs a cheap regex sanity
│   sanity check (always runs)    │     check; LLM review is optional/
│ - LLM re-check (CRITIC_ENABLED) │     configurable (CRITIC_ENABLED)
└────────────┬───────────────────┘
             │ validated plan
             ▼
┌─────────────────────────────┐
│   MCP Client (this project's   │  backend/orchestrator.py
│   own, NOT Cursor)              │  -- stdio subprocess, list_tools/call_tool
└────────────┬───────────────────┘
             │ MCP protocol (stdio, JSON-RPC)
             ▼
┌─────────────────────────────┐
│     MCP Server                  │  backend/mcp_server.py
│  tools: fetch_stock_data,       │  -- pure tool-execution surface, callable
│  calculate_returns,             │     by ANY MCP host, not just this app
│  compute_comparison,            │
│  plot_comparison                │
└────────────┬───────────────────┘
             ▼
┌─────────────────────────────────┐
│ Deterministic Python tools         │  backend/tools/{data,analysis,plotting}.py
│ yfinance → pandas → Matplotlib     │  -- NO LLM involvement, NO arbitrary
│                                     │     code generation/execution
└────────────┬─────────────────────┘
             ▼
    chart PNG + metrics dict
             │
             ▼
    back through FastAPI → Streamlit UI
```

## Why this shape

**Two agents, not three.** The original reference project split "interpret
request," "select tool," and "generate/execute code" into three agents. In
this redesign:
- Interpretation and tool-argument selection are merged into one Planner
  call, because deciding *which* tool to call given a structured intent is
  a near-deterministic mapping (single ticker → single-series tools,
  multiple tickers → comparison tools), not something that benefits from a
  second independent LLM pass.
- "Generate and execute code" is removed entirely. There is no code
  generation step because there is no arbitrary code execution anywhere in
  this system -- the LLM only ever selects and parameterizes pre-written,
  tested Python functions exposed as MCP tools.
- The optional third agent (Critic) exists for a genuinely different
  reason than the original's third agent: it re-checks the Planner's output
  against the original prompt to catch missed tickers or mismatched
  timeframes, which is a real second opinion, not a redundant restatement
  of step one.

**No Cursor.** The reference project's Cursor-IDE-as-MCP-host approach
requires a human-driven IDE session and cannot be invoked programmatically
by an agent pipeline in any standard way. This app's own `orchestrator.py`
is a ~150-line MCP client using the official SDK -- there's no reason to
depend on an IDE for something that small.

**No arbitrary code execution.** The reference project's "text-to-code"
framing (LLM writes Python, something executes it) is a security and
reliability liability for very little benefit. Every operation this system
can perform -- fetch data, compute returns, plot a comparison -- is a
fixed, tested function. The LLM's job is entirely classification/extraction:
turn language into arguments for functions that already exist.

**Why MCP at all, then, if a single app could just call these functions
directly?** Because MCP's real value is host-interoperability, not
in-process convenience. The `mcp_server.py` in this repo is a standalone
process that speaks a standard protocol -- it can be pointed at by this
app's own client, by Claude Desktop, or by any other MCP host, without
changing a line of the tool implementations. For a single-app demo, plain
function calls would work identically; the MCP layer is what makes the tool
surface reusable outside this specific app.

## Request/response flow

1. User submits a prompt in Streamlit.
2. Streamlit POSTs `{"prompt": "..."}` to `FastAPI /analyze`.
3. FastAPI assigns a request ID, calls `orchestrator.run_pipeline`.
4. `run_pipeline` calls the Planner (LLM), which returns a Pydantic
   `AnalysisRequest` -- tickers, resolved date range, metric, chart type, or
   `needs_clarification=True` with a human-readable reason.
5. A deterministic timeframe sanity check always runs (comparing the
   Planner's chosen pattern against an independent regex scan of the raw
   prompt); the optional LLM Critic pass runs next if `CRITIC_ENABLED=true`.
6. If the (possibly Critic-updated) request still needs clarification, the
   pipeline returns immediately with `success=false` and the reason -- no
   MCP call is made.
7. Otherwise, the orchestrator's MCP client calls the `plot_comparison` MCP
   tool with the validated arguments.
8. The MCP server fetches data via yfinance, computes metrics via pandas,
   renders a chart via Matplotlib, and returns `{success, chart_path,
   metrics}` (or a structured error) as JSON over the MCP transport.
9. FastAPI builds a plain-English summary from the metrics and returns
   `AnalyzeApiResponse` to Streamlit.
10. Streamlit renders the summary, the chart (served from `/output/...`),
    and a metrics table.

## Agent responsibilities

| Component | Responsibility | NOT responsible for |
|---|---|---|
| Planner (LLM) | NL → structured intent | Date math, tool selection logic, execution |
| Critic (optional, LLM + deterministic) | Flag likely extraction errors | Rewriting the request, execution |
| Orchestrator (deterministic Python) | MCP connection lifecycle, tool invocation, retries/timeouts | Interpreting language, financial math |
| MCP Server (deterministic Python) | Expose tools with schemas; run fetch/compute/plot | Interpreting language |
| `backend/tools/*` (deterministic Python) | Data fetching, math, charting | Anything LLM-related |

## Backend responsibilities
FastAPI owns HTTP concerns only: request/response schemas, CORS, request-ID
correlation, structured logging, and mapping internal errors
(`OrchestratorError`, clarification responses) to clean HTTP responses. It
does not itself decide what to fetch or how to interpret the prompt.

## Frontend responsibilities
Streamlit owns presentation only: collecting the prompt, calling
`/analyze`, and rendering the response (summary text, chart image, metrics
table, errors/warnings). It has zero knowledge of the Planner, MCP, or the
tools layer.

## Data flow
`str (prompt)` → `dict` (raw LLM JSON) → `AnalysisRequest` (validated
Pydantic model) → `dict` (MCP tool arguments, JSON-serializable) → MCP
protocol → `pandas.DataFrame` (inside the tool) → `dict` (JSON tool result:
records + metrics + chart path) → `AnalyzeApiResponse` (Pydantic) → JSON
over HTTP → Streamlit rendering.

## Failure-handling flow

| Failure | Where caught | User-facing result |
|---|---|---|
| Empty/ambiguous prompt | Planner | `needs_clarification=true`, explanation, no MCP call made |
| Missing/unsupported timeframe phrase | Pydantic validator in `AnalysisRequest` | Same as above |
| Invalid ticker syntax | `tools.data.validate_ticker`, surfaced via `InvalidTickerError` | Structured error via MCP tool result → HTTP 200 with `errors: [...]` |
| Ticker doesn't exist / no data in range | `tools.data.fetch_stock_data` raises `NoDataError` | Same |
| Transient yfinance/network failure | Retried with backoff (`DATA_FETCH_MAX_RETRIES`); raises `DataFetchError` after exhausting retries | Same |
| No overlapping trading dates across tickers | `analysis.compute_comparison` raises `ToolError` | Same |
| MCP server subprocess fails to start / crashes | `orchestrator.MCPClient.__aenter__` raises `OrchestratorError` | HTTP 503 |
| MCP tool call times out | `asyncio.wait_for` in `MCPClient.call_tool` | HTTP 503 (surfaced as `mcp_tool_timeout`) |
| Malformed LLM JSON output | `planner.build_analysis_request` catches `pydantic.ValidationError` | Degrades to `needs_clarification=true` rather than crashing |
| Critic LLM call fails | `critic.review` catches broadly, logs a warning | Pipeline proceeds using the Planner's original output plus a warning |

No failure path returns a raw traceback or a generic "something went
wrong" -- every branch above produces either a specific `error_code` +
message or a `clarification_message` a user can act on.

## Project folder structure

```
financial-analyst-mcp/
├── backend/
│   ├── main.py            # FastAPI app (Phase 4)
│   ├── planner.py          # Planner Agent (Phase 2)
│   ├── critic.py            # Optional Critic/Validator Agent (Phase 2)
│   ├── orchestrator.py      # MCP client + pipeline (Phase 1 client, Phase 3)
│   ├── mcp_server.py        # MCP server (Phase 1)
│   ├── llm_client.py         # LLM provider abstraction incl. offline mock
│   ├── schemas.py             # Pydantic models
│   ├── date_utils.py          # Deterministic timeframe resolution
│   ├── config.py               # Env-based configuration (stdlib only)
│   ├── logging_config.py        # Structured logging + request IDs
│   └── tools/
│       ├── errors.py             # Typed tool errors
│       ├── data.py                # yfinance wrappers (Phase 0)
│       ├── analysis.py             # Returns/volatility/comparison (Phase 0)
│       └── plotting.py              # Matplotlib chart generation (Phase 0)
├── frontend/
│   └── app.py                        # Streamlit UI (Phase 5)
├── eval/
│   ├── eval_prompts.json               # 23 graded prompts (Phase 6)
│   └── run_eval.py                      # Planner accuracy evaluator
├── tests/                                 # unittest-style, pytest-compatible
├── scripts/                                 # run_backend.sh / run_mcp_server.sh / run_frontend.sh / run_tests.sh
├── output/                                    # generated chart PNGs (git-ignored)
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── .env.example
├── .gitignore
└── pytest.ini
```
