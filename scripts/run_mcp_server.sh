#!/usr/bin/env bash
# Starts the MCP server standalone (mainly for manual testing / `mcp dev`).
# In normal operation, backend/orchestrator.py launches this as a subprocess
# automatically -- you do not need to run this yourself to use the app.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m backend.mcp_server
