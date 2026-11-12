"""Conversation memory with a token budget.

The store keeps the full transcript per conversation and hands the router a
window that fits the budget, always retaining at least the last turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


@dataclass
class StoredMessage:
    role: str
    content: str
    ts: float
    name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "ts": self.ts, "name": self.name}


class ConversationStore:
    def __init__(self, token_budget: int = 12000, max_messages: int = 500) -> None:
        self.token_budget = token_budget
        self.max_messages = max_messages
        self._conversations: dict[str, list[StoredMessage]] = {}

    def append(
        self, conversation_id: str, role: str, content: str, name: str | None = None
    ) -> StoredMessage:
        msg = StoredMessage(role=role, content=content or "", ts=time.time(), name=name)
        bucket = self._conversations.setdefault(conversation_id, [])
        bucket.append(msg)
        if len(bucket) > self.max_messages:
            del bucket[: len(bucket) - self.max_messages]
        return msg

    def get(self, conversation_id: str) -> list[StoredMessage]:
        return list(self._conversations.get(conversation_id, []))

    def window(self, conversation_id: str, budget: int | None = None) -> list[StoredMessage]:
        """Most recent messages whose cumulative token cost fits the budget."""
        budget = budget if budget is not None else self.token_budget
        history = self._conversations.get(conversation_id, [])
        if not history:
            return []
        kept: list[StoredMessage] = []
        used = 0
        for msg in reversed(history):
            cost = estimate_tokens(msg.content) + 8
            if kept and used + cost > budget:
                break
            kept.append(msg)
            used += cost
        kept.reverse()
        return kept

    def clear(self, conversation_id: str) -> bool:
        return self._conversations.pop(conversation_id, None) is not None

    def conversations(self) -> list[dict[str, Any]]:
        return [
            {
                "id": cid,
                "messages": len(msgs),
                "tokens": sum(estimate_tokens(m.content) for m in msgs),
                "lastTs": msgs[-1].ts if msgs else None,
            }
            for cid, msgs in self._conversations.items()
        ]
