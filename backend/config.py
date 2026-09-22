"""
Centralized configuration, read entirely from environment variables.

Deliberately implemented with the standard library only (dataclasses + os.environ)
rather than pydantic-settings, so configuration loading has zero third-party
dependencies and can be imported/tested even in environments where the rest of
the stack (FastAPI, pydantic, mcp) is not installed.

Never hard-code secrets here. Real values come from a local `.env` file
(loaded via python-dotenv in main.py / mcp_server.py entry points) or from the
process environment. See .env.example for the full list of variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    # "mock" requires no API key or network and is used for local development,
    # offline demos, and CI. "openai" and "anthropic" are real providers.
    provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "mock")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini")
    )
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", ""))
    temperature: float = field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0)
    )
    timeout_seconds: int = field(
        default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 30)
    )


@dataclass(frozen=True)
class CriticConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("CRITIC_ENABLED", True))


@dataclass(frozen=True)
class MCPConfig:
    server_command: str = field(
        default_factory=lambda: os.environ.get("MCP_SERVER_COMMAND", "python")
    )
    server_args: str = field(
        default_factory=lambda: os.environ.get(
            "MCP_SERVER_ARGS", "-m backend.mcp_server"
        )
    )
    tool_call_timeout_seconds: int = field(
        default_factory=lambda: _env_int("MCP_TOOL_TIMEOUT_SECONDS", 30)
    )

    def server_args_list(self) -> list[str]:
        return self.server_args.split() if self.server_args else []


@dataclass(frozen=True)
class BackendConfig:
    host: str = field(
        default_factory=lambda: os.environ.get("BACKEND_HOST", "127.0.0.1")
    )
    port: int = field(default_factory=lambda: _env_int("BACKEND_PORT", 8000))
    cors_origins: str = field(
        default_factory=lambda: os.environ.get("CORS_ORIGINS", "http://localhost:8501")
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: _env_int("REQUEST_TIMEOUT_SECONDS", 45)
    )

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))
    dir: str = field(default_factory=lambda: os.environ.get("CACHE_DIR", ".cache"))
    ttl_seconds: int = field(
        default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 3600)
    )


@dataclass(frozen=True)
class DataConfig:
    max_retries: int = field(
        default_factory=lambda: _env_int("DATA_FETCH_MAX_RETRIES", 3)
    )
    retry_backoff_seconds: float = field(
        default_factory=lambda: _env_float("DATA_FETCH_RETRY_BACKOFF_SECONDS", 1.0)
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: _env_int("DATA_FETCH_TIMEOUT_SECONDS", 20)
    )


@dataclass(frozen=True)
class LoggingConfig:
    level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    json_format: bool = field(default_factory=lambda: _env_bool("LOG_JSON", False))


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = field(
        default_factory=lambda: os.environ.get("OUTPUT_DIR", "output")
    )


def load_config() -> AppConfig:
    """Build an AppConfig snapshot from the current environment.

    Called fresh (not cached at import time) so tests can monkeypatch
    os.environ and get a consistent config back.
    """
    return AppConfig()
