"""FastAPI application factory — the teamz orchestrator.

Endpoints
---------
GET  /healthz                     liveness
GET  /readyz                      readiness (registry probed)
GET  /providers                   live registry: health + remaining rate budget
POST /route/preview               which provider a request would pick, and why
POST /v1/chat/completions         OpenAI-compatible, routed + fallback + ledger
POST /v1/chat/completions/stream  SSE streaming variant
GET  /usage                       usage + cost summary (dashboard data)
GET  /usage/records               raw ledger rows
POST /usage/reset                 clear the ledger
GET  /conversations               conversation memory index
GET  /conversations/{id}          stored transcript
DELETE /conversations/{id}        forget a conversation
GET  /keys                        list API keys (never the secret)
POST /keys                        mint a key (secret returned once)
DELETE /keys/{id}                 revoke a key
GET  /tools                       tool-calling capability probe
POST /tools/validate              validate tool definitions / structured reply
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings
from .keys import ApiKey, ApiKeyStore
from .memory import ConversationStore
from .models import ApiKeyCreate, ChatCompletionRequest, RoutePreviewRequest
from .providers import ProviderRegistry
from .router import AllProvidersFailed, Router, estimate_tokens
from .tools import ToolError, extract_tool_calls, parse_structured, tool_names, validate_tools
from .usage import UsageLedger, cost_usd


def create_app(
    *,
    registry: ProviderRegistry | None = None,
    router: Router | None = None,
    ledger: UsageLedger | None = None,
    keys: ApiKeyStore | None = None,
    memory: ConversationStore | None = None,
    probe_on_startup: bool = True,
) -> FastAPI:
    settings = get_settings()
    registry = registry or ProviderRegistry()
    router = router or Router(registry)
    ledger = ledger or UsageLedger()
    keys = keys or ApiKeyStore(default_rpm=settings.rate_limit_rpm)
    memory = memory or ConversationStore(token_budget=settings.default_token_budget)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if probe_on_startup:
            try:
                await registry.probe_all()
            except Exception:  # noqa: BLE001 - a failed probe must not stop boot
                pass
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Keyless-first LLM routing and agent orchestration for teamz.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.registry = registry
    app.state.router = router
    app.state.ledger = ledger
    app.state.keys = keys
    app.state.memory = memory

    # ------------------------------------------------------------------ auth
    async def require_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> ApiKey | None:
        if not settings.require_api_key:
            return None
        token = x_api_key
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        key = keys.verify(token)
        if key is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        allowed, retry_after = keys.check_rate(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for this API key.",
                headers={"Retry-After": str(retry_after)},
            )
        return key

    # -------------------------------------------------------------- health
    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "service": settings.app_name, "version": settings.version}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> dict[str, Any]:
        healthy = [p.name for p in registry.all() if p.health.status == "healthy"]
        return {
            "ready": bool(healthy),
            "healthyProviders": healthy,
            "providerCount": len(registry.all()),
        }

    # ----------------------------------------------------------- providers
    @app.get("/providers", tags=["routing"])
    async def list_providers() -> dict[str, Any]:
        return {
            "providers": registry.as_dict(),
            "count": len(registry.all()),
            "keylessCount": len([p for p in registry.all() if p.cost_tier == 0]),
        }

    @app.post("/providers/probe", tags=["routing"])
    async def probe_providers() -> dict[str, Any]:
        return {"health": await registry.probe_all()}

    @app.post("/route/preview", tags=["routing"])
    async def route_preview(req: RoutePreviewRequest) -> dict[str, Any]:
        try:
            decision = router.select(
                model=req.model,
                local_only=req.local_only,
                keyless_only=req.keyless_only,
                hard_task=req.hard_task,
            )
        except AllProvidersFailed:
            raise HTTPException(status_code=503, detail="No provider is available for that request.")
        return {
            **decision.as_dict(),
            "estimatedTokens": req.estimated_tokens,
            "estimatedCostUsd": cost_usd(
                decision.provider, decision.model, req.estimated_tokens, 0
            ),
        }

    # ---------------------------------------------------------- completions
    def _body(req: ChatCompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        }
        for field in ("temperature", "max_tokens", "tools", "tool_choice", "response_format"):
            value = getattr(req, field)
            if value is not None:
                body[field] = value
        return body

    def _record(result: Any, req: ChatCompletionRequest, key: ApiKey | None, latency_ms: int) -> None:
        ledger.record(
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.usage["promptTokens"],
            completion_tokens=result.usage["completionTokens"],
            cost_usd=cost_usd(
                result.provider,
                result.model,
                result.usage["promptTokens"],
                result.usage["completionTokens"],
            ),
            latency_ms=latency_ms,
            ok=True,
            conversation_id=req.conversation_id,
            api_key_id=key.id if key else None,
        )

    @app.post("/v1/chat/completions", tags=["routing"])
    async def chat_completions(
        req: ChatCompletionRequest, key: ApiKey | None = Depends(require_key)
    ) -> JSONResponse:
        try:
            validate_tools(req.tools)
        except ToolError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        hard_task = bool(req.tools) or bool(req.response_format) or len(req.messages) > 12
        started = time.monotonic()
        try:
            result = await router.complete(
                _body(req),
                model=req.model,
                local_only=req.local_only,
                keyless_only=req.keyless_only,
                hard_task=hard_task,
                only_provider=req.provider,
            )
        except AllProvidersFailed as exc:
            ledger.record(
                provider="none",
                model=req.model or "default",
                prompt_tokens=estimate_tokens(
                    " ".join(m.content or "" for m in req.messages)
                ),
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=int((time.monotonic() - started) * 1000),
                ok=False,
                conversation_id=req.conversation_id,
                api_key_id=key.id if key else None,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "all_providers_failed",
                        "message": "Every provider in the fallback chain failed.",
                        "hops": [h.as_dict() for h in exc.hops],
                    }
                },
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        _record(result, req, key, latency_ms)

        if req.conversation_id:
            for msg in req.messages:
                if msg.role in {"user", "assistant"}:
                    memory.append(req.conversation_id, msg.role, msg.content or "", msg.name)
            memory.append(req.conversation_id, "assistant", result.text)

        return JSONResponse(
            content={
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result.usage["promptTokens"],
                    "completion_tokens": result.usage["completionTokens"],
                    "total_tokens": result.usage["totalTokens"],
                },
                "teamz": {
                    "provider": result.provider,
                    "routing": result.decision.as_dict(),
                    "hops": [h.as_dict() for h in result.hops],
                    "latencyMs": latency_ms,
                    "toolCalls": extract_tool_calls(result.raw),
                },
            }
        )

    @app.post("/v1/chat/completions/stream", tags=["routing"])
    async def chat_completions_stream(
        req: ChatCompletionRequest, key: ApiKey | None = Depends(require_key)
    ) -> StreamingResponse:
        hard_task = bool(req.tools) or bool(req.response_format) or len(req.messages) > 12

        async def events() -> AsyncIterator[str]:
            started = time.monotonic()
            try:
                async for event in router.stream(
                    _body(req),
                    model=req.model,
                    local_only=req.local_only,
                    keyless_only=req.keyless_only,
                    hard_task=hard_task,
                    only_provider=req.provider,
                ):
                    if event["type"] == "done":
                        ledger.record(
                            provider=event["provider"],
                            model=event["model"],
                            prompt_tokens=estimate_tokens(
                                " ".join(m.content or "" for m in req.messages)
                            ),
                            completion_tokens=0,
                            cost_usd=0.0,
                            latency_ms=int((time.monotonic() - started) * 1000),
                            ok=True,
                            conversation_id=req.conversation_id,
                            api_key_id=key.id if key else None,
                        )
                    yield f"data: {json.dumps(event)}\n\n"
            except AllProvidersFailed as exc:
                yield f"data: {json.dumps({'type': 'error', 'hops': [h.as_dict() for h in exc.hops]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---------------------------------------------------------------- usage
    @app.get("/usage", tags=["usage"])
    async def usage(since: float | None = Query(default=None)) -> dict[str, Any]:
        return ledger.summary(since=since)

    @app.get("/usage/records", tags=["usage"])
    async def usage_records(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        rows = ledger.records()[-limit:]
        return {"records": [r.as_dict() for r in rows], "count": len(rows)}

    @app.post("/usage/reset", tags=["usage"])
    async def usage_reset() -> dict[str, Any]:
        ledger.reset()
        return {"ok": True}

    # --------------------------------------------------------- conversations
    @app.get("/conversations", tags=["memory"])
    async def conversations() -> dict[str, Any]:
        return {"conversations": memory.conversations()}

    @app.get("/conversations/{conversation_id}", tags=["memory"])
    async def conversation(conversation_id: str) -> dict[str, Any]:
        msgs = memory.get(conversation_id)
        if not msgs:
            raise HTTPException(status_code=404, detail="No such conversation.")
        return {
            "id": conversation_id,
            "messages": [m.as_dict() for m in msgs],
            "window": [m.as_dict() for m in memory.window(conversation_id)],
        }

    @app.delete("/conversations/{conversation_id}", tags=["memory"])
    async def forget_conversation(conversation_id: str) -> dict[str, Any]:
        return {"ok": memory.clear(conversation_id)}

    # ----------------------------------------------------------------- keys
    @app.get("/keys", tags=["keys"])
    async def list_keys() -> dict[str, Any]:
        return {"keys": keys.list()}

    @app.post("/keys", tags=["keys"])
    async def create_key(req: ApiKeyCreate) -> dict[str, Any]:
        key, token = keys.create(req.name, req.rpm_limit)
        return {
            "key": key.as_dict(),
            "token": token,
            "note": "Store this token now — it is shown once and never persisted in plaintext.",
        }

    @app.delete("/keys/{key_id}", tags=["keys"])
    async def revoke_key(key_id: str) -> dict[str, Any]:
        if not keys.revoke(key_id):
            raise HTTPException(status_code=404, detail="No such key.")
        return {"ok": True}

    # ---------------------------------------------------------------- tools
    @app.get("/tools", tags=["tools"])
    async def tools_capability() -> dict[str, Any]:
        return {
            "toolCalling": True,
            "structuredOutput": True,
            "supportedSchemaKeywords": ["type", "required", "properties", "items", "enum"],
        }

    @app.post("/tools/validate", tags=["tools"])
    async def tools_validate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            tools = validate_tools(payload.get("tools"))
            result: dict[str, Any] = {"ok": True, "tools": tool_names(tools)}
            if "reply" in payload:
                result["parsed"] = parse_structured(payload["reply"], payload.get("schema"))
            return result
        except ToolError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # ------------------------------------------------------------- errors
    @app.exception_handler(AllProvidersFailed)
    async def _all_failed(_request: Request, exc: AllProvidersFailed) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "all_providers_failed",
                    "message": str(exc),
                    "hops": [h.as_dict() for h in exc.hops],
                }
            },
        )

    return app


app = create_app()
