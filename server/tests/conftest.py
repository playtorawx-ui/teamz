"""Shared fixtures: a fake httpx transport so no test ever touches the network."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from teamz_server.keys import ApiKeyStore
from teamz_server.memory import ConversationStore
from teamz_server.providers import Provider, ProviderRegistry
from teamz_server.router import Router
from teamz_server.usage import UsageLedger


class FakeTransport(httpx.AsyncBaseTransport):
    """Routes requests to a per-provider handler.

    ``handlers`` maps a provider base URL to a callable that receives the
    request and returns an ``httpx.Response``. A provider with no handler
    behaves like a dead host (connection error).
    """

    def __init__(self, handlers: dict[str, Callable[[httpx.Request], httpx.Response]] | None = None):
        self.handlers = handlers or {}
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        for base, handler in self.handlers.items():
            if str(request.url).startswith(base.rstrip("/")):
                return handler(request)
        raise httpx.ConnectError("no handler for this host", request=request)


def make_client(transport: FakeTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, timeout=5.0)


def ok_body(text: str = "hello", model: str = "test-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def json_response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def text_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=text)


@pytest.fixture
def providers() -> list[Provider]:
    return [
        Provider("local-a", "http://local-a.test/v1", models=["m1"], is_local=True, priority=1, kind="local"),
        Provider("keyless-b", "http://keyless-b.test/v1", models=["m1"], priority=10),
        Provider("keyed-c", "http://keyed-c.test/v1", models=["m1"], api_key="sk-test", priority=60, kind="keyed"),
    ]


@pytest.fixture
def registry(providers: list[Provider]) -> ProviderRegistry:
    return ProviderRegistry(providers)


@pytest.fixture
def ledger() -> UsageLedger:
    return UsageLedger()


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(default_rpm=120)


@pytest.fixture
def memory() -> ConversationStore:
    return ConversationStore(token_budget=200)


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def client(transport: FakeTransport) -> httpx.AsyncClient:
    return make_client(transport)


@pytest.fixture
def router(registry: ProviderRegistry, client: httpx.AsyncClient) -> Router:
    return Router(registry, client_factory=lambda: client)


def sse(*chunks: str) -> str:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}" for c in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def sse_response(*chunks: str) -> httpx.Response:
    """A real SSE response — the content-type is what selects the parser."""
    return httpx.Response(
        200, text=sse(*chunks), headers={"content-type": "text/event-stream"}
    )
