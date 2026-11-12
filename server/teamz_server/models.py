"""Pydantic v2 request/response models for the orchestrator API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion, plus teamz routing hints."""

    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    # --- teamz extensions -------------------------------------------------
    conversation_id: str | None = None
    provider: str | None = None
    local_only: bool = False
    keyless_only: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutePreviewRequest(BaseModel):
    model: str | None = None
    local_only: bool = False
    keyless_only: bool = False
    estimated_tokens: int = 0
    hard_task: bool = False


class ApiKeyCreate(BaseModel):
    name: str = "default"
    rpm_limit: int | None = None
