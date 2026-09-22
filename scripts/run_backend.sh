#!/usr/bin/env bash
# Starts the FastAPI backend. Reads configuration from .env (see .env.example).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  echo "No .env found -- copying .env.example. Edit .env to set a real LLM_API_KEY if you want LLM_PROVIDER=openai/anthropic." 
  cp .env.example .env
fi
uvicorn backend.main:app --host "${BACKEND_HOST:-127.0.0.1}" --port "${BACKEND_PORT:-8000}" --reload
