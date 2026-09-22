"""
Streamlit frontend (Phase 5).

Talks to the FastAPI backend over plain HTTP (`requests`) -- it has no
knowledge of the Planner, MCP, or the deterministic tools; it is purely a
thin UI over the /analyze endpoint. This keeps the boundary honest: swapping
this for a different frontend (a React app, a CLI, a Slack bot) would not
require touching any backend code.

NOTE ON THIS DELIVERY: requires `streamlit` and `requests` (not installed in
the sandbox that produced this codebase -- see README).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

EXAMPLE_PROMPTS = [
    "Compare Apple and Microsoft over the last 6 months.",
    "Show me Tesla's performance during 2025.",
    "Compare AAPL, MSFT and NVDA over the past 6 months.",
    "Plot normalized returns for Tesla and Apple since January 2024.",
]

st.set_page_config(
    page_title="Financial Analyst (MCP)", page_icon="📈", layout="centered"
)

st.title("📈 Multi-Agent Financial Analyst")
st.caption(
    "Natural language in, a real stock chart out. Planner Agent (LLM) → Pydantic "
    "validation → optional Critic → MCP client/server → deterministic yfinance/"
    "Matplotlib tools. No Cursor, no arbitrary code execution."
)

with st.expander("Example prompts"):
    for ex in EXAMPLE_PROMPTS:
        st.code(ex, language=None)

prompt = st.text_area(
    "Ask a financial question",
    placeholder="Compare Apple's stock performance with Microsoft's over the last 6 months "
    "and plot their normalized returns.",
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    submitted = st.button("Analyze", type="primary")

if submitted:
    if not prompt.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Interpreting your request and fetching data..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/analyze",
                    json={"prompt": prompt.strip()},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the backend at {BACKEND_URL}. "
                    "Is it running? (see README: 'How to start the FastAPI backend')"
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out. The data source or LLM provider may be slow or unreachable."
                )
                st.stop()
            except requests.exceptions.HTTPError as exc:
                st.error(f"Backend returned an error: {exc}")
                st.stop()

        if data.get("extracted_request"):
            with st.expander("Interpreted request", expanded=not data["success"]):
                st.json(data["extracted_request"])

        for warning in data.get("warnings", []):
            st.info(f"ℹ️ {warning}")

        if not data["success"]:
            for err in data.get("errors", []):
                st.error(err)
            st.stop()

        st.success(data.get("summary") or "Analysis complete.")

        chart_path = data.get("chart_path")
        if chart_path:
            filename = os.path.basename(chart_path)
            chart_url = f"{BACKEND_URL}/output/{filename}"
            try:
                st.image(chart_url, use_container_width=True)
            except Exception:
                st.warning(
                    f"Chart was generated at {chart_path} but could not be displayed "
                    f"inline. Try opening {chart_url} directly."
                )

        metrics = data.get("metrics")
        if metrics:
            st.subheader("Metrics")
            st.dataframe(
                {
                    ticker: {
                        "Total Return": f"{m['total_return']:+.1%}",
                        "CAGR": f"{m['cagr']:+.1%}",
                        "Volatility (ann.)": f"{m['annualized_volatility']:.1%}",
                        "Max Drawdown": f"{m['max_drawdown']:.1%}",
                    }
                    for ticker, m in metrics.items()
                },
                use_container_width=True,
            )

st.divider()
st.caption(f"Request ID of last call: `{st.session_state.get('last_request_id', '—')}`")
