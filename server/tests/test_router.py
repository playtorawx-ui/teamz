"""Routing: scoring, fallback chain, backoff, circuit breaker, streaming."""

from __future__ import annotations

import httpx
import pytest

from teamz_server.providers import Provider, ProviderRegistry
from teamz_server.router import (
    AllProvidersFailed,
    Router,
    estimate_tokens,
    extract_text,
    extract_usage,
    resolve_model_for,
    score_provider,
)
from tests.conftest import (
    FakeTransport,
    json_response,
    make_client,
    ok_body,
    sse_response,
    text_response,
)


def test_estimate_tokens_is_positive() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_extract_text_openai_shape() -> None:
    assert extract_text(ok_body("hi")) == "hi"


def test_extract_text_anthropic_shape() -> None:
    payload = {"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}
    assert extract_text(payload) == "hello world"


def test_extract_text_plain_keys() -> None:
    assert extract_text({"reply": "r"}) == "r"
    assert extract_text({"text": "t"}) == "t"
    assert extract_text({}) == ""
    assert extract_text(None) == ""


def test_extract_usage_prefers_reported_numbers() -> None:
    usage = extract_usage(ok_body("hi"), [{"content": "x"}])
    assert usage == {"promptTokens": 11, "completionTokens": 7, "totalTokens": 18}


def test_extract_usage_estimates_when_absent() -> None:
    usage = extract_usage({"choices": [{"message": {"content": "abcd"}}]}, [{"content": "abcdefgh"}])
    assert usage["promptTokens"] >= 1
    assert usage["completionTokens"] >= 1


def test_resolve_model_falls_back_to_provider_model() -> None:
    p = Provider("a", "http://a.test/v1", models=["gpt-oss-20b"])
    # A model the provider does not serve must not be sent to it.
    assert resolve_model_for(p, "qwen2.5-coder:32b") == "gpt-oss-20b"
    assert resolve_model_for(p, "gpt-oss-20b") == "gpt-oss-20b"
    assert resolve_model_for(p, None) == "gpt-oss-20b"
    empty = Provider("b", "http://b.test/v1")
    assert resolve_model_for(empty, "anything") == "anything"
    assert resolve_model_for(empty, None) == "default"


def test_score_prefers_local_then_keyless() -> None:
    local = Provider("l", "http://l.test/v1", is_local=True, priority=1, kind="local")
    free = Provider("f", "http://f.test/v1", priority=10)
    paid = Provider("p", "http://p.test/v1", api_key="k", priority=60, kind="keyed")
    scores = [score_provider(p, model=None, hard_task=False)[0] for p in (local, free, paid)]
    assert scores[0] < scores[1] < scores[2]


def test_score_penalises_dead_and_open_circuit() -> None:
    p = Provider("a", "http://a.test/v1", priority=1)
    p.health.status = "dead"
    dead_score, why = score_provider(p, model=None, hard_task=False)
    assert dead_score > 1000
    assert "marked dead" in why


def test_degraded_provider_ranks_below_healthy_remote() -> None:
    """A dead localhost runtime must not outrank a healthy remote provider."""
    local = Provider("l", "http://l.test/v1", is_local=True, priority=1, kind="local")
    local.health.status = "degraded"
    remote = Provider("r", "http://r.test/v1", priority=10)
    remote.health.status = "healthy"
    assert score_provider(remote, model=None, hard_task=False)[0] < score_provider(
        local, model=None, hard_task=False
    )[0]


def test_score_rewards_serving_requested_model() -> None:
    serves = Provider("a", "http://a.test/v1", models=["m1"], priority=10)
    other = Provider("b", "http://b.test/v1", models=["m2"], priority=10)
    assert score_provider(serves, model="m1", hard_task=False)[0] < score_provider(
        other, model="m1", hard_task=False
    )[0]


def test_score_penalises_exhausted_budget() -> None:
    p = Provider("a", "http://a.test/v1", rpm_limit=1, priority=10)
    p.note_hit()
    score, why = score_provider(p, model=None, hard_task=False)
    assert score > 300
    assert "rate budget exhausted" in why


def test_hard_task_escalates_past_free_remote() -> None:
    free = Provider("f", "http://f.test/v1", priority=10)
    paid = Provider("p", "http://p.test/v1", api_key="k", priority=60, kind="keyed")
    assert score_provider(paid, model=None, hard_task=True)[0] < score_provider(
        free, model=None, hard_task=True
    )[0]


def test_select_returns_decision_with_candidates(registry: ProviderRegistry) -> None:
    decision = Router(registry).select()
    assert decision.provider == "local-a"
    assert decision.model == "m1"
    assert len(decision.candidates) == 3
    assert decision.reason


def test_select_honours_only_provider(registry: ProviderRegistry) -> None:
    decision = Router(registry).select(only_provider="keyed-c")
    assert decision.provider == "keyed-c"


def test_select_raises_when_nothing_available() -> None:
    with pytest.raises(AllProvidersFailed):
        Router(ProviderRegistry([])).select()


def test_preview_shape(registry: ProviderRegistry) -> None:
    payload = Router(registry).preview()
    assert payload["provider"] == "local-a"
    assert "score" in payload and "candidates" in payload


@pytest.mark.asyncio
async def test_complete_uses_first_healthy_provider(registry: ProviderRegistry) -> None:
    transport = FakeTransport({"http://local-a.test/v1": lambda r: json_response(ok_body("hi"))})
    async with make_client(transport) as client:
        result = await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert result.provider == "local-a"
    assert result.text == "hi"
    assert result.usage["totalTokens"] == 18
    assert len(result.hops) == 1


@pytest.mark.asyncio
async def test_complete_falls_back_on_500(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("boom", 500),
            "http://keyless-b.test/v1": lambda r: json_response(ok_body("fallback")),
        }
    )
    async with make_client(transport) as client:
        result = await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert result.provider == "keyless-b"
    assert result.text == "fallback"
    assert [h.provider for h in result.hops] == ["local-a", "keyless-b"]
    assert result.hops[0].ok is False


@pytest.mark.asyncio
async def test_complete_falls_back_on_429(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("rate limited", 429),
            "http://keyless-b.test/v1": lambda r: json_response(ok_body("ok")),
        }
    )
    async with make_client(transport) as client:
        result = await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert result.provider == "keyless-b"
    assert result.hops[0].status == 429


@pytest.mark.asyncio
async def test_complete_falls_back_on_model_unavailable(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("model_unavailable", 400),
            "http://keyless-b.test/v1": lambda r: json_response(ok_body("ok")),
        }
    )
    async with make_client(transport) as client:
        result = await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert result.provider == "keyless-b"


@pytest.mark.asyncio
async def test_complete_falls_back_on_connect_error(registry: ProviderRegistry) -> None:
    transport = FakeTransport({"http://keyless-b.test/v1": lambda r: json_response(ok_body("ok"))})
    async with make_client(transport) as client:
        result = await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert result.provider == "keyless-b"
    assert result.hops[0].error


@pytest.mark.asyncio
async def test_complete_raises_when_all_fail(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("x", 500),
            "http://keyless-b.test/v1": lambda r: text_response("x", 500),
            "http://keyed-c.test/v1": lambda r: text_response("x", 500),
        }
    )
    async with make_client(transport) as client:
        with pytest.raises(AllProvidersFailed) as exc:
            await Router(registry, client_factory=lambda: client).complete(
                {"messages": [{"role": "user", "content": "hi"}]}
            )
    assert len(exc.value.hops) == 3


@pytest.mark.asyncio
async def test_failure_deprioritises_and_circuit_opens_at_threshold(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("x", 500),
            "http://keyless-b.test/v1": lambda r: json_response(ok_body("ok")),
        }
    )
    router = Router(registry, client_factory=lambda: client)
    async with make_client(transport) as client:
        for _ in range(3):
            await router.complete({"messages": [{"role": "user", "content": "hi"}]})
    local = registry.get("local-a")
    assert local.health.consecutive_failures >= 3
    assert local.health.circuit_open is True
    # A tripped provider is skipped entirely, not retried.
    assert router.select().provider == "keyless-b"


@pytest.mark.asyncio
async def test_open_circuit_provider_is_skipped(registry: ProviderRegistry) -> None:
    registry.get("local-a").health.circuit_open_until = 10**9
    transport = FakeTransport({"http://keyless-b.test/v1": lambda r: json_response(ok_body("ok"))})
    async with make_client(transport) as client:
        result = await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
    assert result.provider == "keyless-b"
    assert [h.provider for h in result.hops] == ["keyless-b"]


@pytest.mark.asyncio
async def test_stream_yields_start_delta_done(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {"http://local-a.test/v1": lambda r: sse_response("Hel", "lo")}
    )
    async with make_client(transport) as client:
        events = [
            e
            async for e in Router(registry, client_factory=lambda: client).stream(
                {"messages": [{"role": "user", "content": "hi"}]}
            )
        ]
    assert events[0]["type"] == "start"
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Hello"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_falls_back_when_first_provider_fails(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("nope", 500),
            "http://keyless-b.test/v1": lambda r: sse_response("fallback"),
        }
    )
    async with make_client(transport) as client:
        events = [
            e
            async for e in Router(registry, client_factory=lambda: client).stream(
                {"messages": [{"role": "user", "content": "hi"}]}
            )
        ]
    assert events[0]["provider"] == "keyless-b"
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "fallback"


@pytest.mark.asyncio
async def test_stream_handles_provider_that_ignores_stream_flag(registry: ProviderRegistry) -> None:
    """A provider answering with one JSON body must still produce a delta."""
    transport = FakeTransport(
        {"http://local-a.test/v1": lambda r: json_response(ok_body("whole answer"))}
    )
    async with make_client(transport) as client:
        events = [
            e
            async for e in Router(registry, client_factory=lambda: client).stream(
                {"messages": [{"role": "user", "content": "hi"}]}
            )
        ]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "whole answer"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_empty_response_advances_chain(registry: ProviderRegistry) -> None:
    """A 200 with no content is a failed provider, not an empty answer."""
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: sse_response(),
            "http://keyless-b.test/v1": lambda r: sse_response("real"),
        }
    )
    async with make_client(transport) as client:
        events = [
            e
            async for e in Router(registry, client_factory=lambda: client).stream(
                {"messages": [{"role": "user", "content": "hi"}]}
            )
        ]
    assert events[0]["provider"] == "keyless-b"
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "real"


@pytest.mark.asyncio
async def test_stream_raises_when_all_fail(registry: ProviderRegistry) -> None:
    transport = FakeTransport(
        {
            "http://local-a.test/v1": lambda r: text_response("x", 500),
            "http://keyless-b.test/v1": lambda r: text_response("x", 500),
            "http://keyed-c.test/v1": lambda r: text_response("x", 500),
        }
    )
    async with make_client(transport) as client:
        with pytest.raises(AllProvidersFailed):
            async for _ in Router(registry, client_factory=lambda: client).stream(
                {"messages": [{"role": "user", "content": "hi"}]}
            ):
                pass


@pytest.mark.asyncio
async def test_complete_sends_resolved_model_not_unknown_id(registry: ProviderRegistry) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return json_response(ok_body("ok"))

    transport = FakeTransport({"http://local-a.test/v1": handler})
    async with make_client(transport) as client:
        await Router(registry, client_factory=lambda: client).complete(
            {"messages": [{"role": "user", "content": "hi"}]}, model="qwen2.5-coder:32b"
        )
    assert seen[0]["model"] == "m1"
