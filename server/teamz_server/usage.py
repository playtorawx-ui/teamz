"""Usage + cost accounting — the data behind the usage dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# USD per 1M tokens (input, output). Keyless/local providers are free by
# definition, so anything absent from this table costs 0.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "openai": (0.15, 0.60),
    "anthropic": (3.00, 15.00),
    "gemini": (0.10, 0.40),
}


@dataclass
class UsageRecord:
    ts: float
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    ok: bool
    conversation_id: str | None = None
    api_key_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "provider": self.provider,
            "model": self.model,
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "totalTokens": self.prompt_tokens + self.completion_tokens,
            "costUsd": self.cost_usd,
            "latencyMs": self.latency_ms,
            "ok": self.ok,
            "conversationId": self.conversation_id,
            "apiKeyId": self.api_key_id,
        }


def price_for(provider: str, model: str) -> tuple[float, float]:
    return PRICE_TABLE.get(provider, (0.0, 0.0))


def cost_usd(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = price_for(provider, model)
    return round(
        (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out, 8
    )


class UsageLedger:
    """In-memory, bounded ledger of every routed request."""

    def __init__(self, max_records: int = 5000) -> None:
        self._records: list[UsageRecord] = []
        self.max_records = max_records

    def record(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: int,
        ok: bool,
        conversation_id: str | None = None,
        api_key_id: str | None = None,
        ts: float | None = None,
    ) -> UsageRecord:
        rec = UsageRecord(
            ts=ts if ts is not None else time.time(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            ok=ok,
            conversation_id=conversation_id,
            api_key_id=api_key_id,
        )
        self._records.append(rec)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
        return rec

    def records(self) -> list[UsageRecord]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()

    def summary(self, since: float | None = None) -> dict[str, Any]:
        rows = [r for r in self._records if since is None or r.ts >= since]
        by_provider: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        by_day: dict[str, dict[str, Any]] = {}
        total_prompt = total_completion = 0
        total_cost = 0.0
        total_latency = 0
        ok_count = 0

        for r in rows:
            total_prompt += r.prompt_tokens
            total_completion += r.completion_tokens
            total_cost += r.cost_usd
            total_latency += r.latency_ms
            ok_count += 1 if r.ok else 0

            for bucket, key in ((by_provider, r.provider), (by_model, r.model)):
                entry = bucket.setdefault(
                    key, {"requests": 0, "tokens": 0, "costUsd": 0.0, "errors": 0}
                )
                entry["requests"] += 1
                entry["tokens"] += r.prompt_tokens + r.completion_tokens
                entry["costUsd"] = round(entry["costUsd"] + r.cost_usd, 8)
                entry["errors"] += 0 if r.ok else 1

            day = time.strftime("%Y-%m-%d", time.gmtime(r.ts))
            day_entry = by_day.setdefault(day, {"requests": 0, "tokens": 0, "costUsd": 0.0})
            day_entry["requests"] += 1
            day_entry["tokens"] += r.prompt_tokens + r.completion_tokens
            day_entry["costUsd"] = round(day_entry["costUsd"] + r.cost_usd, 8)

        count = len(rows)
        return {
            "requests": count,
            "okRequests": ok_count,
            "errorRequests": count - ok_count,
            "successRate": round(ok_count / count, 4) if count else 1.0,
            "promptTokens": total_prompt,
            "completionTokens": total_completion,
            "totalTokens": total_prompt + total_completion,
            "costUsd": round(total_cost, 8),
            "avgLatencyMs": round(total_latency / count) if count else 0,
            "byProvider": by_provider,
            "byModel": by_model,
            "byDay": dict(sorted(by_day.items())),
        }
