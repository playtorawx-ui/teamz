"""Environment-driven configuration.

Nothing here is ever hard-coded to a secret: every key is read from the
environment, and a keyless provider is a first-class citizen with an empty key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    version: str
    cors_origins: tuple[str, ...]
    request_timeout_s: float
    max_attempts: int
    backoff_base_s: float
    circuit_failure_threshold: int
    circuit_cooldown_s: float
    health_probe_interval_s: float
    default_token_budget: int
    require_api_key: bool
    rate_limit_rpm: int
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = _env("TEAMZ_CORS_ORIGINS", "*")
    return Settings(
        app_name=_env("TEAMZ_APP_NAME", "teamz-orchestrator"),
        version=_env("TEAMZ_VERSION", "0.2.0"),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()) or ("*",),
        request_timeout_s=_env_float("TEAMZ_REQUEST_TIMEOUT_S", 30.0),
        max_attempts=_env_int("TEAMZ_MAX_ATTEMPTS", 4),
        backoff_base_s=_env_float("TEAMZ_BACKOFF_BASE_S", 0.25),
        circuit_failure_threshold=_env_int("TEAMZ_CIRCUIT_FAILURE_THRESHOLD", 3),
        circuit_cooldown_s=_env_float("TEAMZ_CIRCUIT_COOLDOWN_S", 30.0),
        health_probe_interval_s=_env_float("TEAMZ_HEALTH_PROBE_INTERVAL_S", 60.0),
        default_token_budget=_env_int("TEAMZ_TOKEN_BUDGET", 12000),
        require_api_key=_env_bool("TEAMZ_REQUIRE_API_KEY", False),
        rate_limit_rpm=_env_int("TEAMZ_RATE_LIMIT_RPM", 120),
        log_level=_env("TEAMZ_LOG_LEVEL", "INFO"),
    )
