"""Capability + cost + health aware routing with an observable fallback chain.

On 429 / 5xx / timeout / ``model_unavailable`` the router advances to the next
provider, logs the hop, and never surfaces a raw provider error to the caller.
Backoff is exponential with jitter and each provider has a circuit breaker.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import httpx

from .config import get_settings
from .providers import Provider, ProviderRegistry

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
MODEL_UNAVAILABLE_MARKERS = (
    "model_unavailable",
    "model not found",
    "does not exist",
    "no such model",
    "unknown model",
)


class AllProvidersFailed(RuntimeError):
    """Raised when every provider in the fallback chain failed."""

    def __init__(self, hops: list["Hop"]) -> None:
        self.hops = hops
        detail = "; ".join(f"{h.provider}: {h.error}" for h in hops) or "no candidates"
        super().__init__(f"every provider in the fallback chain failed ({detail})")


@dataclass
class Hop:
    provider: str
    model: str
    ok: bool
    status: int | None = None
    error: str | None = None
    latency_ms: int = 0
    retryable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "latencyMs": self.latency_ms,
            "retryable": self.retryable,
        }


@dataclass
class RouteDecision:
    provider: str
    model: str
    reason: str
    score: float
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reason": self.reason,
            "score": self.score,
            "candidates": self.candidates,
        }


@dataclass
class CompletionResult:
    provider: str
    model: str
    text: str
    usage: dict[str, int]
    hops: list[Hop]
    decision: RouteDecision
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars per token)."""
    return max(1, len(text or "") // 4)


def extract_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if choices:
        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)
            )
        if isinstance(first.get("text"), str):
            return first["text"]
    for key in ("reply", "text", "message", "content", "output"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        # Anthropic-compatible: { content: [{ type: "text", text: "..." }] }
        if isinstance(value, list):
            joined = "".join(
                b.get("text", "")
                for b in value
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            )
            if joined:
                return joined
    return ""


def extract_usage(data: Any, messages: list[dict[str, Any]] | None = None) -> dict[str, int]:
    usage = (data or {}).get("usage") or {} if isinstance(data, dict) else {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    if not prompt:
        prompt = estimate_tokens(" ".join(str(m.get("content") or "") for m in (messages or [])))
    if not completion:
        completion = estimate_tokens(extract_text(data))
    return {"promptTokens": prompt, "completionTokens": completion, "totalTokens": prompt + completion}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def resolve_model_for(provider: Provider, requested: str | None) -> str:
    """The model id to actually send to ``provider``.

    A requested id is only used when that provider serves it; otherwise the
    provider's own first model is used. Sending an unknown id is a guaranteed
    404 (`model_not_found`) and would burn a fallback hop for nothing.
    """
    if requested and (not provider.models or requested in provider.models):
        return requested
    if provider.models:
        return provider.models[0]
    return requested or "default"


def score_provider(provider: Provider, *, model: str | None, hard_task: bool) -> tuple[float, list[str]]:
    """Lower is better. Returns the score and a human-readable reason list."""
    score = float(provider.priority)
    why: list[str] = [f"priority {provider.priority}"]

    if provider.health.circuit_open:
        score += 5000
        why.append("circuit open")
    elif provider.health.status == "dead":
        score += 2000
        why.append("marked dead")
    elif provider.health.status == "degraded":
        # A provider that just failed a probe must not outrank a healthy one:
        # otherwise a dead localhost runtime eats the whole attempt budget
        # before the chain ever reaches a working remote provider.
        score += 150
        why.append("degraded (last probe failed)")
    elif provider.health.status == "unknown":
        score += 50
        why.append("unprobed")

    if provider.health.latency_ms is not None:
        score += min(30.0, provider.health.latency_ms / 100.0)
        why.append(f"latency {provider.health.latency_ms}ms")

    if model:
        if model in provider.models:
            score -= 40
            why.append("serves requested model")
        elif provider.models:
            score += 60
            why.append("model not listed")

    if provider.is_local:
        score -= 5
        why.append("local (no network)")

    if hard_task and provider.cost_tier == 0 and not provider.is_local:
        # A hard task (tools / structured output / long context) escalates past
        # the free remote tier: the penalty must outweigh the priority gap.
        score += 60
        why.append("hard task prefers a keyed tier")

    if provider.remaining_budget() == 0:
        score += 400
        why.append("rate budget exhausted")

    return score, why


class Router:
    """Selects a provider and runs the fallback chain."""

    def __init__(
        self,
        registry: ProviderRegistry,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.registry = registry
        # Only close a client we created ourselves: a caller-supplied factory
        # (or an injected client) may hand back a shared, long-lived client.
        self._owns_clients = client_factory is None
        self._client_factory = client_factory or ProviderRegistry._default_client

    # --- selection --------------------------------------------------------
    def rank(
        self,
        *,
        model: str | None = None,
        local_only: bool = False,
        keyless_only: bool = False,
        hard_task: bool = False,
        only_provider: str | None = None,
        include_dead: bool = True,
    ) -> list[tuple[float, Provider, list[str]]]:
        if only_provider:
            pinned = self.registry.get(only_provider)
            pool = [pinned] if pinned else []
        else:
            pool = self.registry.candidates(
                model=model,
                local_only=local_only,
                keyless_only=keyless_only,
                include_dead=include_dead,
            )
        ranked = [(*score_provider(p, model=model, hard_task=hard_task), p) for p in pool]
        ranked.sort(key=lambda t: (t[0], t[2].name))
        return [(score, provider, why) for score, why, provider in ranked]

    def select(
        self,
        *,
        model: str | None = None,
        local_only: bool = False,
        keyless_only: bool = False,
        hard_task: bool = False,
        only_provider: str | None = None,
    ) -> RouteDecision:
        ranked = self.rank(
            model=model,
            local_only=local_only,
            keyless_only=keyless_only,
            hard_task=hard_task,
            only_provider=only_provider,
        )
        if not ranked:
            raise AllProvidersFailed([])
        score, provider, why = ranked[0]
        chosen_model = resolve_model_for(provider, model)
        return RouteDecision(
            provider=provider.name,
            model=chosen_model,
            reason="; ".join(why),
            score=round(score, 2),
            candidates=[
                {"provider": p.name, "score": round(s, 2), "reason": "; ".join(w)} for s, p, w in ranked
            ],
        )

    def preview(self, **kwargs: Any) -> dict[str, Any]:
        return self.select(**kwargs).as_dict()

    # --- execution --------------------------------------------------------
    async def complete(
        self,
        body: dict[str, Any],
        *,
        model: str | None = None,
        local_only: bool = False,
        keyless_only: bool = False,
        hard_task: bool = False,
        only_provider: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> CompletionResult:
        settings = get_settings()
        ranked = self.rank(
            model=model,
            local_only=local_only,
            keyless_only=keyless_only,
            hard_task=hard_task,
            only_provider=only_provider,
        )
        if not ranked:
            raise AllProvidersFailed([])
        decision = self.select(
            model=model,
            local_only=local_only,
            keyless_only=keyless_only,
            hard_task=hard_task,
            only_provider=only_provider,
        )

        hops: list[Hop] = []
        own = client is None and self._owns_clients
        client = client or self._client_factory()
        # Never spend an attempt on a provider whose circuit is open: it is
        # known-bad until its cooldown expires. Fall back to the full list only
        # when every candidate is tripped.
        usable = [t for t in ranked if not t[1].health.circuit_open] or ranked
        try:
            for attempt, (_score, provider, _why) in enumerate(usable[: settings.max_attempts]):
                if attempt:
                    await asyncio.sleep(
                        settings.backoff_base_s * (2 ** (attempt - 1)) + random.random() * 0.1
                    )
                chosen_model = resolve_model_for(provider, model)
                payload = {**body, "model": chosen_model, "stream": False}
                started = time.monotonic()
                try:
                    provider.note_hit()
                    res = await client.post(provider.chat_url, headers=provider.headers(), json=payload)
                    latency = int((time.monotonic() - started) * 1000)
                    if res.status_code >= 400:
                        text = res.text[:400]
                        retryable = res.status_code in RETRYABLE_STATUS or any(
                            m in text.lower() for m in MODEL_UNAVAILABLE_MARKERS
                        )
                        hops.append(
                            Hop(provider.name, chosen_model, False, res.status_code, text, latency, retryable)
                        )
                        self.registry.record_failure(provider.name, f"{res.status_code} {text[:120]}")
                        continue
                    data = res.json()
                    text = extract_text(data)
                    usage = extract_usage(data, payload.get("messages"))
                    self.registry.record_success(provider.name, latency)
                    hops.append(Hop(provider.name, chosen_model, True, res.status_code, None, latency))
                    return CompletionResult(provider.name, chosen_model, text, usage, hops, decision, data)
                except Exception as exc:  # noqa: BLE001 - any transport error advances the chain
                    latency = int((time.monotonic() - started) * 1000)
                    hops.append(Hop(provider.name, chosen_model, False, None, str(exc), latency, True))
                    self.registry.record_failure(provider.name, str(exc))
                    continue
        finally:
            if own:
                await client.aclose()
        raise AllProvidersFailed(hops)

    async def stream(
        self,
        body: dict[str, Any],
        *,
        model: str | None = None,
        local_only: bool = False,
        keyless_only: bool = False,
        hard_task: bool = False,
        only_provider: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield ``start`` / ``delta`` / ``done`` events, falling back on failure."""
        settings = get_settings()
        ranked = self.rank(
            model=model,
            local_only=local_only,
            keyless_only=keyless_only,
            hard_task=hard_task,
            only_provider=only_provider,
        )
        if not ranked:
            raise AllProvidersFailed([])

        hops: list[Hop] = []
        own = client is None and self._owns_clients
        client = client or self._client_factory()
        usable = [t for t in ranked if not t[1].health.circuit_open] or ranked
        try:
            for attempt, (_score, provider, _why) in enumerate(usable[: settings.max_attempts]):
                if attempt:
                    await asyncio.sleep(
                        settings.backoff_base_s * (2 ** (attempt - 1)) + random.random() * 0.1
                    )
                chosen_model = resolve_model_for(provider, model)
                payload = {**body, "model": chosen_model, "stream": True}
                started = time.monotonic()
                try:
                    provider.note_hit()
                    async with client.stream(
                        "POST", provider.chat_url, headers=provider.headers(), json=payload
                    ) as res:
                        if res.status_code >= 400:
                            text = (await res.aread()).decode()[:400]
                            latency = int((time.monotonic() - started) * 1000)
                            hops.append(
                                Hop(provider.name, chosen_model, False, res.status_code, text, latency, True)
                            )
                            self.registry.record_failure(provider.name, f"{res.status_code} {text[:120]}")
                            continue
                        content_type = res.headers.get("content-type", "")
                        emitted = 0
                        started_event = False

                        def start_event() -> dict[str, Any]:
                            return {"type": "start", "provider": provider.name, "model": chosen_model}

                        if "text/event-stream" not in content_type:
                            # Some providers ignore `stream: true` and answer
                            # with one JSON body. Emit it as a single delta
                            # instead of silently producing an empty stream.
                            raw = (await res.aread()).decode()
                            try:
                                whole = extract_text(json.loads(raw))
                            except json.JSONDecodeError:
                                whole = raw.strip()
                            if whole:
                                emitted += 1
                                if not started_event:
                                    started_event = True
                                    yield start_event()
                                yield {"type": "delta", "text": whole}
                        else:
                            async for line in res.aiter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                chunk = line[5:].strip()
                                if chunk == "[DONE]":
                                    break
                                try:
                                    data = json.loads(chunk)
                                except json.JSONDecodeError:
                                    continue
                                delta = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content")
                                if delta:
                                    emitted += 1
                                    if not started_event:
                                        started_event = True
                                        yield start_event()
                                    yield {"type": "delta", "text": delta}
                        if emitted == 0:
                            # A 200 with no content is a failed provider, not a
                            # successful empty answer: advance the chain.
                            latency = int((time.monotonic() - started) * 1000)
                            hops.append(
                                Hop(provider.name, chosen_model, False, res.status_code, "empty stream", latency, True)
                            )
                            self.registry.record_failure(provider.name, "empty stream")
                            continue
                        self.registry.record_success(provider.name)
                        yield {"type": "done", "provider": provider.name, "model": chosen_model}
                        return
                except Exception as exc:  # noqa: BLE001
                    latency = int((time.monotonic() - started) * 1000)
                    hops.append(Hop(provider.name, chosen_model, False, None, str(exc), latency, True))
                    self.registry.record_failure(provider.name, str(exc))
                    continue
        finally:
            if own:
                await client.aclose()
        raise AllProvidersFailed(hops)
