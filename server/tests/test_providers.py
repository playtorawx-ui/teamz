"""Provider registry: keyless-first defaults, health probing, circuit breaker."""

from __future__ import annotations

import httpx
import pytest

from teamz_server.providers import Health, Provider, ProviderRegistry, default_providers
from tests.conftest import FakeTransport, json_response, make_client, text_response


def test_default_registry_has_keyless_and_local_first() -> None:
    providers = default_providers()
    names = [p.name for p in providers]
    # Every documented keyless provider is present.
    for expected in ("ollama", "lmstudio", "llamacpp", "vllm", "localai", "ovhcloud", "pollinations", "llm7"):
        assert expected in names, expected
    # Local runtimes outrank remote keyless, which outrank keyed.
    by_name = {p.name: p for p in providers}
    assert by_name["ollama"].priority < by_name["ovhcloud"].priority < by_name["llm7"].priority
    assert by_name["ollama"].is_local is True
    assert by_name["ovhcloud"].cost_tier == 0
    assert by_name["pollinations"].cost_tier == 0


def test_keyless_providers_need_no_key() -> None:
    by_name = {p.name: p for p in default_providers()}
    assert by_name["pollinations"].api_key is None
    assert by_name["ovhcloud"].headers().get("Authorization") is None
    # A placeholder key is still a valid, non-secret configuration.
    assert by_name["llm7"].headers()["Authorization"] == "Bearer unused"


def test_keyed_providers_only_registered_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "openai" not in [p.name for p in default_providers()]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    openai = {p.name: p for p in default_providers()}["openai"]
    assert openai.kind == "keyed"
    assert openai.cost_tier == 1
    assert openai.headers()["Authorization"] == "Bearer sk-live"


def test_no_hardcoded_secret_in_source() -> None:
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "teamz_server" / "providers.py"
    text = source.read_text()
    assert "sk-" not in text.replace("sk-test", "")


def test_non_chat_models_are_filtered_out() -> None:
    """A `/models` probe returns audio/embedding models too — never route chat at them."""
    from teamz_server.providers import filter_chat_models, is_chat_model

    assert is_chat_model("Meta-Llama-3_3-70B-Instruct") is True
    assert is_chat_model("gpt-oss-120b") is True
    assert is_chat_model("whisper-large-v3") is False
    assert is_chat_model("text-embedding-3-small") is False
    assert is_chat_model("bge-m3") is False
    assert is_chat_model("") is False
    assert filter_chat_models(["whisper-large-v3", "gpt-oss-20b"]) == ["gpt-oss-20b"]


@pytest.mark.asyncio
async def test_probe_drops_pro_tier_and_non_chat_models() -> None:
    reg = ProviderRegistry([Provider("llm7", "http://llm7.test/v1")])
    transport = FakeTransport(
        {
            "http://llm7.test/v1": lambda r: json_response(
                {
                    "data": [
                        {"id": "codestral-latest", "tier": "free"},
                        {"id": "mistral-Nemo-Instruct-2407", "tier": "free"},
                        {"id": "gpt-5-pro", "tier": "pro"},
                        {"id": "whisper-large-v3", "tier": "free"},
                    ]
                }
            )
        }
    )
    async with make_client(transport) as client:
        await reg.probe(reg.get("llm7"), client)
    assert reg.get("llm7").models == ["codestral-latest", "mistral-Nemo-Instruct-2407"]


def test_remaining_budget_counts_hits() -> None:
    p = Provider("x", "http://x.test/v1", rpm_limit=2)
    assert p.remaining_budget() == 2
    p.note_hit()
    assert p.remaining_budget() == 1
    p.note_hit()
    assert p.remaining_budget() == 0
    unlimited = Provider("y", "http://y.test/v1", rpm_limit=0)
    assert unlimited.remaining_budget() is None


def test_candidates_filters() -> None:
    reg = ProviderRegistry(
        [
            Provider("local", "http://l.test/v1", is_local=True, priority=1, kind="local"),
            Provider("free", "http://f.test/v1", priority=10),
            Provider("paid", "http://p.test/v1", api_key="k", priority=60, kind="keyed"),
        ]
    )
    assert [p.name for p in reg.candidates(local_only=True)] == ["local"]
    assert sorted(p.name for p in reg.candidates(keyless_only=True)) == ["free", "local"]
    assert len(reg.candidates()) == 3


def test_model_is_a_preference_not_a_filter() -> None:
    """An agent's model string must never exclude every provider."""
    reg = ProviderRegistry(
        [
            Provider("a", "http://a.test/v1", models=["gpt-oss-20b"]),
            Provider("b", "http://b.test/v1", models=["llama-3"]),
        ]
    )
    # `qwen2.5-coder:32b` is served by nobody, yet both stay eligible.
    assert len(reg.candidates(model="qwen2.5-coder:32b")) == 2


def test_candidates_skip_dead_unless_asked() -> None:
    reg = ProviderRegistry([Provider("a", "http://a.test/v1"), Provider("b", "http://b.test/v1")])
    reg.get("a").health.status = "dead"
    assert [p.name for p in reg.candidates()] == ["b"]
    assert sorted(p.name for p in reg.candidates(include_dead=True)) == ["a", "b"]


@pytest.mark.asyncio
async def test_probe_marks_healthy_and_learns_models() -> None:
    reg = ProviderRegistry([Provider("a", "http://a.test/v1")])
    transport = FakeTransport(
        {"http://a.test/v1": lambda r: json_response({"data": [{"id": "m-new"}, {"id": "m-two"}]})}
    )
    async with make_client(transport) as client:
        health = await reg.probe(reg.get("a"), client)
    assert health.status == "healthy"
    assert reg.get("a").models == ["m-new", "m-two"]
    assert health.latency_ms is not None


@pytest.mark.asyncio
async def test_probe_failure_opens_circuit_after_threshold() -> None:
    reg = ProviderRegistry([Provider("a", "http://a.test/v1")])
    transport = FakeTransport({"http://a.test/v1": lambda r: text_response("boom", 503)})
    async with make_client(transport) as client:
        for _ in range(3):
            await reg.probe(reg.get("a"), client)
    health = reg.get("a").health
    assert health.consecutive_failures >= 3
    assert health.circuit_open is True
    assert health.status == "dead"


@pytest.mark.asyncio
async def test_probe_transport_error_is_recorded() -> None:
    reg = ProviderRegistry([Provider("a", "http://a.test/v1")])
    async with make_client(FakeTransport()) as client:
        health = await reg.probe(reg.get("a"), client)
    assert health.status == "degraded"
    assert health.last_error


@pytest.mark.asyncio
async def test_probe_all_returns_every_provider() -> None:
    reg = ProviderRegistry([Provider("a", "http://a.test/v1"), Provider("b", "http://b.test/v1")])
    transport = FakeTransport({"http://a.test/v1": lambda r: json_response({"data": []})})
    async with make_client(transport) as client:
        result = await reg.probe_all(client)
    assert set(result) == {"a", "b"}


def test_record_success_clears_circuit() -> None:
    reg = ProviderRegistry([Provider("a", "http://a.test/v1")])
    for _ in range(3):
        reg.record_failure("a", "nope")
    assert reg.get("a").health.circuit_open
    reg.record_success("a", 12)
    assert reg.get("a").health.circuit_open is False
    assert reg.get("a").health.status == "healthy"


def test_health_as_dict_shape() -> None:
    payload = Health(status="healthy", latency_ms=5).as_dict()
    assert payload["status"] == "healthy"
    assert payload["circuitOpen"] is False
    assert "circuitOpenForMs" in payload


def test_registry_add_remove_and_dict() -> None:
    reg = ProviderRegistry([])
    reg.add(Provider("a", "http://a.test/v1"))
    assert reg.names() == ["a"]
    assert reg.as_dict()[0]["name"] == "a"
    assert reg.remove("a") is True
    assert reg.remove("a") is False
