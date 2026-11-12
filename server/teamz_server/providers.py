"""Provider registry — keyless providers are first-class citizens.

Every provider is modelled as a record (name, base_url, api_key, models,
rpm_limit, is_local, priority, health) and is probed with
``GET {base_url}/models``. A provider that fails repeatedly trips a circuit
breaker and drops out of rotation until its cooldown expires.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import httpx

from .config import get_settings


@dataclass
class Health:
    status: str = "unknown"  # unknown | healthy | degraded | dead
    last_checked: float | None = None
    latency_ms: int | None = None
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    last_error: str | None = None

    @property
    def circuit_open(self) -> bool:
        return time.monotonic() < self.circuit_open_until

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "lastChecked": self.last_checked,
            "latencyMs": self.latency_ms,
            "consecutiveFailures": self.consecutive_failures,
            "circuitOpen": self.circuit_open,
            "circuitOpenForMs": max(0, int((self.circuit_open_until - time.monotonic()) * 1000)),
            "lastError": self.last_error,
        }


@dataclass
class Provider:
    name: str
    base_url: str
    models: list[str] = field(default_factory=list)
    api_key: str | None = None
    rpm_limit: int = 0
    is_local: bool = False
    priority: int = 50
    kind: str = "keyless"  # keyless | keyed | local
    health: Health = field(default_factory=Health)
    _hits: list[float] = field(default_factory=list, repr=False)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"

    @property
    def cost_tier(self) -> int:
        """0 = free (local or keyless), 1 = billed."""
        return 0 if (self.is_local or self.kind == "keyless") else 1

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def note_hit(self) -> None:
        now = time.monotonic()
        self._hits = [t for t in self._hits if now - t < 60.0]
        self._hits.append(now)

    def remaining_budget(self) -> int | None:
        """Requests left in the current minute, or None when unlimited."""
        if self.rpm_limit <= 0:
            return None
        now = time.monotonic()
        used = len([t for t in self._hits if now - t < 60.0])
        return max(0, self.rpm_limit - used)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseUrl": self.base_url,
            "models": list(self.models),
            "rpmLimit": self.rpm_limit,
            "isLocal": self.is_local,
            "priority": self.priority,
            "kind": self.kind,
            "costTier": self.cost_tier,
            "hasKey": bool(self.api_key),
            "remainingBudget": self.remaining_budget(),
            "health": self.health.as_dict(),
        }


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


# Model ids that are not chat models. A `/models` probe returns every model a
# provider serves, including audio/embedding/rerank ones; routing a chat
# completion at `whisper-large-v3` is a guaranteed 404.
NON_CHAT_MARKERS = (
    "whisper",
    "embed",
    "embedding",
    "tts",
    "text-to-speech",
    "rerank",
    "moderation",
    "dall-e",
    "stable-diffusion",
    "flux",
    "sdxl",
    "image",
    "vision-ocr",
    "bge-",
    "clip",
)


def is_chat_model(model_id: str) -> bool:
    """True when a model id looks like a text chat model."""
    lowered = (model_id or "").lower()
    if not lowered:
        return False
    return not any(marker in lowered for marker in NON_CHAT_MARKERS)


def filter_chat_models(ids: list[str]) -> list[str]:
    return [i for i in ids if is_chat_model(i)]


def default_providers() -> list[Provider]:
    """The zero-config registry: local runtimes, then keyless remote, then keyed.

    Keyless providers are the zero-cost tier and are used first for cheap work;
    keyed providers are only registered when a key is actually present in the
    environment, so nothing is ever hard-coded.
    """
    providers: list[Provider] = [
        # --- local runtimes: keyless, no network, highest priority ----------
        Provider("ollama", "http://localhost:11434/v1", api_key="ollama", is_local=True, priority=1, kind="local"),
        Provider("lmstudio", "http://localhost:1234/v1", api_key="lm-studio", is_local=True, priority=2, kind="local"),
        Provider("llamacpp", "http://localhost:8080/v1", api_key="none", is_local=True, priority=3, kind="local"),
        Provider("vllm", "http://localhost:8000/v1", api_key="none", is_local=True, priority=4, kind="local"),
        # llama.cpp and LocalAI both default to :8080 upstream; LocalAI is
        # registered on :8081 so the two never shadow each other.
        Provider("localai", "http://localhost:8081/v1", api_key="none", is_local=True, priority=5, kind="local"),
        # --- remote, no key at all -----------------------------------------
        Provider(
            "ovhcloud",
            "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
            models=[
                "Meta-Llama-3_3-70B-Instruct",
                "Qwen2.5-VL-72B-Instruct",
                "gpt-oss-120b",
                "gpt-oss-20b",
            ],
            api_key=_env("OVH_AI_ENDPOINTS_ACCESS_TOKEN") or None,
            rpm_limit=2,
            priority=10,
        ),
        Provider(
            "pollinations",
            "https://text.pollinations.ai/openai",
            models=["openai", "mistral", "llama"],
            api_key=None,
            rpm_limit=20,
            priority=20,
        ),
        Provider(
            "llm7",
            "https://api.llm7.io/v1",
            models=["codestral-latest", "mistral-Nemo-Instruct-2407"],
            api_key=_env("LLM7_API_KEY") or "unused",
            rpm_limit=10,
            priority=30,
        ),
    ]

    # --- keyed providers: registered only when a key exists ---------------
    if _env("OPENAI_API_KEY"):
        providers.append(
            Provider(
                "openai",
                "https://api.openai.com/v1",
                api_key=_env("OPENAI_API_KEY"),
                models=["gpt-4o-mini", "gpt-4o"],
                rpm_limit=500,
                priority=60,
                kind="keyed",
            )
        )
    if _env("ANTHROPIC_API_KEY"):
        providers.append(
            Provider(
                "anthropic",
                "https://api.anthropic.com/v1",
                api_key=_env("ANTHROPIC_API_KEY"),
                models=["claude-3-5-sonnet-latest"],
                rpm_limit=300,
                priority=61,
                kind="keyed",
            )
        )
    if _env("GEMINI_API_KEY"):
        providers.append(
            Provider(
                "gemini",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                api_key=_env("GEMINI_API_KEY"),
                models=["gemini-2.0-flash"],
                rpm_limit=300,
                priority=62,
                kind="keyed",
            )
        )
    return providers


class ProviderRegistry:
    """Holds the live provider set, their health and their rate budget."""

    def __init__(
        self,
        providers: Iterable[Provider] | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._providers: dict[str, Provider] = {
            p.name: p for p in (providers if providers is not None else default_providers())
        }
        self._client_factory = client_factory or self._default_client

    @staticmethod
    def _default_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=get_settings().request_timeout_s)

    # --- access -----------------------------------------------------------
    def all(self) -> list[Provider]:
        return list(self._providers.values())

    def get(self, name: str) -> Provider | None:
        return self._providers.get(name)

    def add(self, provider: Provider) -> Provider:
        self._providers[provider.name] = provider
        return provider

    def remove(self, name: str) -> bool:
        return self._providers.pop(name, None) is not None

    def names(self) -> list[str]:
        return list(self._providers)

    def candidates(
        self,
        *,
        model: str | None = None,
        local_only: bool = False,
        keyless_only: bool = False,
        include_dead: bool = False,
    ) -> list[Provider]:
        """Providers eligible for a request.

        ``model`` is deliberately NOT a filter: an agent's model string is a
        *preference* (e.g. `qwen2.5-coder:32b` from a local template), and
        hard-filtering on it would exclude every provider that does not happen
        to list that exact id — turning a preference into a total outage.
        Scoring rewards a provider that does serve it.
        """
        out: list[Provider] = []
        for p in self._providers.values():
            if local_only and not p.is_local:
                continue
            if keyless_only and p.cost_tier != 0:
                continue
            if not include_dead and p.health.status == "dead":
                continue
            out.append(p)
        return out

    # --- health -----------------------------------------------------------
    async def probe(self, provider: Provider, client: httpx.AsyncClient | None = None) -> Health:
        started = time.monotonic()
        own = client is None
        client = client or self._client_factory()
        try:
            res = await client.get(provider.models_url, headers=provider.headers())
            latency = int((time.monotonic() - started) * 1000)
            if res.status_code >= 500:
                self.record_failure(provider.name, f"probe {res.status_code}")
            else:
                provider.health.status = "healthy" if res.status_code < 400 else "degraded"
                provider.health.last_checked = time.time()
                provider.health.latency_ms = latency
                provider.health.consecutive_failures = 0
                provider.health.circuit_open_until = 0.0
                provider.health.last_error = None
                if res.status_code < 400:
                    try:
                        data = res.json()
                        entries = [m for m in (data.get("data") or []) if isinstance(m, dict)]
                        # LLM7 marks paid models `tier: pro`; they return
                        # `model_unavailable` for an anonymous key, so they must
                        # never enter the rotation.
                        entries = [
                            m
                            for m in entries
                            if str(m.get("tier", "")).lower() != "pro"
                        ]
                        ids = [m.get("id") for m in entries if m.get("id")]
                        ids = filter_chat_models([i for i in ids if isinstance(i, str)])
                        if ids:
                            # Keep the curated order first: a provider's own
                            # listing is alphabetical-ish and often leads with a
                            # reasoning model that returns empty content. The
                            # curated ids are the ones known to answer chat.
                            curated = [m for m in provider.models if m in ids]
                            provider.models = curated + [i for i in ids if i not in curated]
                    except Exception:  # noqa: BLE001 - a bad body must not kill the probe
                        pass
        except Exception as exc:  # noqa: BLE001 - any transport error is a failure
            self.record_failure(provider.name, str(exc))
        finally:
            if own:
                await client.aclose()
        return provider.health

    async def probe_all(self, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
        own = client is None
        client = client or self._client_factory()
        try:
            await asyncio.gather(*(self.probe(p, client) for p in self._providers.values()))
        finally:
            if own:
                await client.aclose()
        return {p.name: p.health.as_dict() for p in self._providers.values()}

    def record_success(self, name: str, latency_ms: int | None = None) -> None:
        p = self._providers.get(name)
        if not p:
            return
        p.health.status = "healthy"
        p.health.last_checked = time.time()
        if latency_ms is not None:
            p.health.latency_ms = latency_ms
        p.health.consecutive_failures = 0
        p.health.circuit_open_until = 0.0
        p.health.last_error = None

    def record_failure(self, name: str, error: str) -> None:
        p = self._providers.get(name)
        if not p:
            return
        settings = get_settings()
        p.health.consecutive_failures += 1
        p.health.last_checked = time.time()
        p.health.last_error = error
        if p.health.consecutive_failures >= settings.circuit_failure_threshold:
            p.health.circuit_open_until = time.monotonic() + settings.circuit_cooldown_s
            p.health.status = "dead"
        elif p.health.status == "unknown":
            p.health.status = "degraded"

    def as_dict(self) -> list[dict[str, Any]]:
        return [p.as_dict() for p in self._providers.values()]
