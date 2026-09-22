#!/usr/bin/env bash
# Runs the full test suite. Modules that need fastapi/pydantic/mcp will
# self-skip with a clear message if those packages aren't installed yet --
# run `pip install -r requirements.txt` first for full coverage.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/ -v
