"""End-to-end API tests against the real FastAPI app with a fake transport."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from teamz_server.app import create_app
from teamz_server.keys import ApiKeyStore
from teamz_server.memory import ConversationStore
from teamz_server.providers import Provider, ProviderRegistry
from teamz_server.router import Router
from teamz_server.usage import UsageLedger
from tests.conftest import (
    FakeTransport,
    json_response,
    make_client,
    ok_body,
    sse_response,
    text_response,
)


def build_app(handlers: dict | None = None, *, require_key: bool = False):
    providers = [
        Provider("local-a", "http://local-a.test/v1", models=["m1"], is_local=True, priority=1, kind="local"),
        Provider("keyless-b", "http://keyless-b.test/v1", models=["m1"], priority=10),
    ]
    registry = ProviderRegistry(providers)
    transport = FakeTransport(handlers or {})
    client = make_client(transport)
    router = Router(registry, client_factory=lambda: client)
    ledger = UsageLedger()
    keys = ApiKeyStore(default_rpm=120)
    memory = ConversationStore(token_budget=200)
    app = create_app(
        registry=registry,
        router=router,
        ledger=ledger,
        keys=keys,
        memory=memory,
        probe_on_startup=False,
    )
    return app, {"registry": registry, "ledger": ledger, "keys": keys, "memory": memory}


@pytest.fixture
def app_and_state():
    app, state = build_app({"http://local-a.test/v1": lambda r: json_response(ok_body("hi there"))})
    with TestClient(app) as client:
        yield client, state


def test_healthz(app_and_state) -> None:
    client, _ = app_and_state
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_readyz_reports_healthy_providers(app_and_state) -> None:
    client, state = app_and_state
    state["registry"].get("local-a").health.status = "healthy"
    payload = client.get("/readyz").json()
    assert payload["ready"] is True
    assert "local-a" in payload["healthyProviders"]


def test_providers_endpoint_exposes_health_and_budget(app_and_state) -> None:
    client, _ = app_and_state
    payload = client.get("/providers").json()
    assert payload["count"] == 2
    assert payload["keylessCount"] == 2
    first = payload["providers"][0]
    assert "health" in first
    assert "remainingBudget" in first
    assert "hasKey" in first


def test_route_preview_explains_the_choice(app_and_state) -> None:
    client, _ = app_and_state
    payload = client.post("/route/preview", json={"estimated_tokens": 1000}).json()
    assert payload["provider"] == "local-a"
    assert payload["reason"]
    assert payload["candidates"]
    assert payload["estimatedCostUsd"] == 0.0


def test_route_preview_keyless_only(app_and_state) -> None:
    client, _ = app_and_state
    payload = client.post("/route/preview", json={"keyless_only": True}).json()
    assert payload["provider"] in {"local-a", "keyless-b"}


def test_chat_completion_returns_openai_shape(app_and_state) -> None:
    client, _ = app_and_state
    res = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi there"
    assert body["usage"]["total_tokens"] == 18
    assert body["teamz"]["provider"] == "local-a"
    assert body["teamz"]["hops"][0]["ok"] is True


def test_chat_completion_records_usage(app_and_state) -> None:
    client, state = app_and_state
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    summary = client.get("/usage").json()
    assert summary["requests"] == 1
    assert summary["totalTokens"] == 18
    assert summary["successRate"] == 1.0
    assert "local-a" in summary["byProvider"]


def test_chat_completion_falls_back_and_reports_hops() -> None:
    app, _ = build_app(
        {
            "http://local-a.test/v1": lambda r: text_response("down", 503),
            "http://keyless-b.test/v1": lambda r: json_response(ok_body("rescued")),
        }
    )
    with TestClient(app) as client:
        body = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        ).json()
    assert body["teamz"]["provider"] == "keyless-b"
    assert [h["provider"] for h in body["teamz"]["hops"]] == ["local-a", "keyless-b"]


def test_chat_completion_503_when_all_fail() -> None:
    app, _ = build_app(
        {
            "http://local-a.test/v1": lambda r: text_response("x", 500),
            "http://keyless-b.test/v1": lambda r: text_response("x", 500),
        }
    )
    with TestClient(app) as client:
        res = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert res.status_code == 503
    assert res.json()["error"]["type"] == "all_providers_failed"
    assert len(res.json()["error"]["hops"]) == 2


def test_chat_completion_rejects_bad_tools(app_and_state) -> None:
    client, _ = app_and_state
    res = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": ""}}],
        },
    )
    assert res.status_code == 422


def test_chat_completion_stores_conversation(app_and_state) -> None:
    client, _ = app_and_state
    client.post(
        "/v1/chat/completions",
        json={
            "conversation_id": "conv-1",
            "messages": [{"role": "user", "content": "remember me"}],
        },
    )
    payload = client.get("/conversations/conv-1").json()
    assert payload["messages"][0]["content"] == "remember me"
    assert payload["messages"][-1]["role"] == "assistant"


def test_conversation_404(app_and_state) -> None:
    client, _ = app_and_state
    assert client.get("/conversations/nope").status_code == 404


def test_conversation_delete(app_and_state) -> None:
    client, _ = app_and_state
    client.post(
        "/v1/chat/completions",
        json={"conversation_id": "c2", "messages": [{"role": "user", "content": "x"}]},
    )
    assert client.delete("/conversations/c2").json()["ok"] is True
    assert client.get("/conversations/c2").status_code == 404


def test_conversations_index(app_and_state) -> None:
    client, _ = app_and_state
    client.post(
        "/v1/chat/completions",
        json={"conversation_id": "c3", "messages": [{"role": "user", "content": "x"}]},
    )
    payload = client.get("/conversations").json()
    assert payload["conversations"][0]["id"] == "c3"


def test_stream_endpoint_emits_sse() -> None:
    app, _ = build_app(
        {"http://local-a.test/v1": lambda r: sse_response("str", "eam")}
    )
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as res:
            assert res.status_code == 200
            text = "".join(res.iter_text())
    assert '"type": "start"' in text
    assert '"type": "delta"' in text
    assert "[DONE]" in text


def test_usage_records_and_reset(app_and_state) -> None:
    client, _ = app_and_state
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    records = client.get("/usage/records").json()
    assert records["count"] == 1
    assert records["records"][0]["provider"] == "local-a"
    assert client.post("/usage/reset").json()["ok"] is True
    assert client.get("/usage").json()["requests"] == 0


def test_keys_lifecycle(app_and_state) -> None:
    client, _ = app_and_state
    created = client.post("/keys", json={"name": "ci", "rpm_limit": 5}).json()
    token = created["token"]
    assert token.startswith("tz_")
    assert created["key"]["prefix"] == token[:10]
    listed = client.get("/keys").json()["keys"]
    assert listed[0]["name"] == "ci"
    assert "token" not in listed[0]
    assert client.delete(f"/keys/{created['key']['id']}").json()["ok"] is True
    assert client.delete("/keys/nope").status_code == 404


def test_tools_capability_and_validation(app_and_state) -> None:
    client, _ = app_and_state
    assert client.get("/tools").json()["toolCalling"] is True
    ok = client.post(
        "/tools/validate",
        json={
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
            "reply": '{"answer": 42}',
            "schema": {"type": "object", "required": ["answer"]},
        },
    ).json()
    assert ok["ok"] is True
    assert ok["tools"] == ["search"]
    assert ok["parsed"] == {"answer": 42}
    bad = client.post("/tools/validate", json={"reply": "not json"})
    assert bad.status_code == 422


def test_api_key_required_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEAMZ_REQUIRE_API_KEY", "true")
    from teamz_server.config import get_settings

    get_settings.cache_clear()
    try:
        app, state = build_app({"http://local-a.test/v1": lambda r: json_response(ok_body("ok"))})
        with TestClient(app) as client:
            assert client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
            ).status_code == 401
            token = client.post("/keys", json={"name": "t"}).json()["token"]
            res = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert res.status_code == 200
    finally:
        monkeypatch.delenv("TEAMZ_REQUIRE_API_KEY", raising=False)
        get_settings.cache_clear()
