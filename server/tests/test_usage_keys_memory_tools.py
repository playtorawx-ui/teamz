"""Unit tests for the ledger, key store, memory window and tool helpers."""

from __future__ import annotations

import pytest

from teamz_server.keys import ApiKeyStore
from teamz_server.memory import ConversationStore, estimate_tokens
from teamz_server.tools import (
    ToolError,
    extract_tool_calls,
    parse_structured,
    tool_names,
    validate_against_schema,
    validate_tools,
)
from teamz_server.usage import UsageLedger, cost_usd, price_for


# ------------------------------------------------------------------- ledger
def test_cost_is_zero_for_keyless_providers() -> None:
    assert cost_usd("pollinations", "openai", 1_000_000, 1_000_000) == 0.0
    assert cost_usd("ollama", "llama3", 5000, 5000) == 0.0


def test_cost_uses_price_table_for_keyed() -> None:
    assert price_for("openai", "gpt-4o-mini") == (0.15, 0.60)
    assert cost_usd("openai", "gpt-4o-mini", 1_000_000, 0) == 0.15
    assert cost_usd("openai", "gpt-4o-mini", 0, 1_000_000) == 0.60


def test_ledger_summary_aggregates() -> None:
    ledger = UsageLedger()
    ledger.record(provider="a", model="m", prompt_tokens=10, completion_tokens=5, cost_usd=0.0, latency_ms=100, ok=True)
    ledger.record(provider="a", model="m", prompt_tokens=20, completion_tokens=5, cost_usd=0.0, latency_ms=200, ok=False)
    ledger.record(provider="b", model="n", prompt_tokens=1, completion_tokens=1, cost_usd=0.5, latency_ms=300, ok=True)
    summary = ledger.summary()
    assert summary["requests"] == 3
    assert summary["okRequests"] == 2
    assert summary["errorRequests"] == 1
    assert summary["totalTokens"] == 42
    assert summary["costUsd"] == 0.5
    assert summary["avgLatencyMs"] == 200
    assert summary["byProvider"]["a"]["requests"] == 2
    assert summary["byProvider"]["a"]["errors"] == 1
    assert summary["byModel"]["n"]["tokens"] == 2
    assert len(summary["byDay"]) == 1


def test_ledger_is_bounded() -> None:
    ledger = UsageLedger(max_records=3)
    for i in range(6):
        ledger.record(provider="a", model="m", prompt_tokens=i, completion_tokens=0, cost_usd=0.0, latency_ms=1, ok=True)
    assert len(ledger.records()) == 3
    assert ledger.records()[-1].prompt_tokens == 5


def test_ledger_since_filter() -> None:
    ledger = UsageLedger()
    ledger.record(provider="a", model="m", prompt_tokens=1, completion_tokens=0, cost_usd=0.0, latency_ms=1, ok=True, ts=100.0)
    ledger.record(provider="a", model="m", prompt_tokens=1, completion_tokens=0, cost_usd=0.0, latency_ms=1, ok=True, ts=200.0)
    assert ledger.summary(since=150.0)["requests"] == 1


def test_ledger_reset() -> None:
    ledger = UsageLedger()
    ledger.record(provider="a", model="m", prompt_tokens=1, completion_tokens=0, cost_usd=0.0, latency_ms=1, ok=True)
    ledger.reset()
    assert ledger.records() == []


def test_empty_summary_is_successful() -> None:
    summary = UsageLedger().summary()
    assert summary["requests"] == 0
    assert summary["successRate"] == 1.0


# --------------------------------------------------------------------- keys
def test_key_is_hashed_and_verifiable() -> None:
    store = ApiKeyStore()
    key, token = store.create("ci")
    assert key.hashed != token
    assert store.verify(token) is key
    assert store.verify("wrong") is None
    assert store.verify(None) is None


def test_revoked_key_no_longer_verifies() -> None:
    store = ApiKeyStore()
    key, token = store.create("ci")
    assert store.revoke(key.id) is True
    assert store.verify(token) is None
    assert store.revoke("nope") is False


def test_rate_limit_blocks_after_rpm() -> None:
    store = ApiKeyStore()
    key, _ = store.create("ci", rpm_limit=2)
    assert store.check_rate(key) == (True, 0)
    assert store.check_rate(key) == (True, 0)
    allowed, retry_after = store.check_rate(key)
    assert allowed is False
    assert retry_after >= 1


def test_zero_rpm_means_unlimited() -> None:
    store = ApiKeyStore()
    key, _ = store.create("ci", rpm_limit=0)
    for _ in range(50):
        assert store.check_rate(key)[0] is True


def test_key_list_never_exposes_secret() -> None:
    store = ApiKeyStore()
    store.create("ci")
    listed = store.list()[0]
    assert "hashed" not in listed
    assert "token" not in listed


# ------------------------------------------------------------------- memory
def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1


def test_memory_window_respects_budget() -> None:
    store = ConversationStore(token_budget=40)
    for i in range(20):
        store.append("c", "user", "x" * 100)
    window = store.window("c")
    assert 0 < len(window) < 20
    # The most recent message is always kept.
    assert window[-1] is store.get("c")[-1]


def test_memory_keeps_last_turn_even_over_budget() -> None:
    store = ConversationStore(token_budget=1)
    store.append("c", "user", "x" * 1000)
    assert len(store.window("c")) == 1


def test_memory_is_bounded_per_conversation() -> None:
    store = ConversationStore(max_messages=5)
    for i in range(10):
        store.append("c", "user", str(i))
    assert len(store.get("c")) == 5
    assert store.get("c")[-1].content == "9"


def test_memory_clear_and_index() -> None:
    store = ConversationStore()
    store.append("c", "user", "hi")
    assert store.conversations()[0]["id"] == "c"
    assert store.clear("c") is True
    assert store.clear("c") is False


# -------------------------------------------------------------------- tools
def test_validate_tools_accepts_openai_shape() -> None:
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    assert validate_tools(tools) == tools
    assert tool_names(tools) == ["search"]


def test_validate_tools_rejects_duplicates_and_missing_names() -> None:
    with pytest.raises(ToolError):
        validate_tools([{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "a"}}])
    with pytest.raises(ToolError):
        validate_tools([{"type": "function", "function": {}}])
    with pytest.raises(ToolError):
        validate_tools([{"type": "function", "function": {"name": "a", "parameters": "nope"}}])


def test_validate_tools_empty_is_fine() -> None:
    assert validate_tools(None) == []
    assert validate_tools([]) == []


def test_extract_tool_calls() -> None:
    payload = {"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]}
    assert extract_tool_calls(payload) == [{"id": "1"}]
    assert extract_tool_calls({}) == []
    assert extract_tool_calls(None) == []


def test_parse_structured_validates_schema() -> None:
    schema = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "integer"}}}
    assert parse_structured('{"answer": 42}', schema) == {"answer": 42}
    with pytest.raises(ToolError):
        parse_structured('{"answer": "no"}', schema)
    with pytest.raises(ToolError):
        parse_structured("not json")


def test_validate_against_schema_nested_and_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
    }
    assert validate_against_schema({"items": ["x"], "mode": "a"}, schema) == []
    errors = validate_against_schema({"items": [1], "mode": "z"}, schema)
    assert len(errors) == 2


def test_validate_against_schema_type_mismatch() -> None:
    errors = validate_against_schema("str", {"type": "object"})
    assert errors and "expected object" in errors[0]
