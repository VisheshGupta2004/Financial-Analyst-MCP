# Starts the FastAPI backend. Reads configuration from .env (see .env.example).
if (!(Test-Path ".env")) {
    Write-Host "No .env found -- copying .env.example. Edit .env to set a real LLM_API_KEY if you want LLM_PROVIDER=openai/anthropic."
    Copy-Item ".env.example" ".env"
}
Set-Location (Split-Path $PSScriptRoot -Parent)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
