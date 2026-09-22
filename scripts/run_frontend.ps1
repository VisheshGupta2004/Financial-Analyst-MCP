# Starts the Streamlit frontend. Requires the backend to already be running
# (see run_backend.ps1) in a separate terminal.
Set-Location (Split-Path $PSScriptRoot -Parent)
streamlit run frontend/app.py
