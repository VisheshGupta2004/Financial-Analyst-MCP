#!/usr/bin/env bash
# Starts the Streamlit frontend. Requires the backend to already be running
# (see run_backend.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
streamlit run frontend/app.py
