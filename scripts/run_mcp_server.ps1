# Starts the MCP server standalone (mainly for manual testing / `mcp dev`).
# In normal operation, backend/orchestrator.py launches this as a subprocess
# automatically -- you do not need to run this yourself to use the app.
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m backend.mcp_server
