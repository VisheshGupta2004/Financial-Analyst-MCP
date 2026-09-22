# Runs the full test suite.
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m pytest tests/ -v
