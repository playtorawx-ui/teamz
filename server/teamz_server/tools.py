"""Tool/function calling + structured output helpers.

Deliberately dependency-free: a small JSON-Schema subset is validated by hand
so the orchestrator can reject a malformed tool definition before it reaches a
provider, and can validate a structured reply before returning it.
"""

from __future__ import annotations

import json
from typing import Any


class ToolError(ValueError):
    """Raised when a tool definition or a structured reply is malformed."""


def validate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    if not isinstance(tools, list):
        raise ToolError("tools must be a list")
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise ToolError("each tool must be an object")
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(fn, dict):
            raise ToolError("tool.function must be an object")
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolError("every tool needs a non-empty function.name")
        if name in seen:
            raise ToolError(f"duplicate tool name: {name}")
        seen.add(name)
        params = fn.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ToolError(f"tool {name}: parameters must be a JSON schema object")
    return tools


def tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        fn = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
    return names


def extract_tool_calls(data: Any) -> list[dict[str, Any]]:
    """Pull OpenAI-style tool calls out of a completion payload."""
    if not isinstance(data, dict):
        return []
    choices = data.get("choices") or []
    if not choices:
        return []
    message = (choices[0] or {}).get("message") or {}
    calls = message.get("tool_calls")
    return calls if isinstance(calls, list) else []


def parse_structured(text: str, schema: dict[str, Any] | None = None) -> Any:
    """Parse a JSON reply, optionally validating it against a schema subset."""
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ToolError(f"reply is not valid JSON: {exc}") from exc
    if schema:
        errors = validate_against_schema(value, schema)
        if errors:
            raise ToolError("reply does not match schema: " + "; ".join(errors))
    return value


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Minimal JSON-Schema validation: type, required, properties, items, enum."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, t) for t in types):
            errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    if isinstance(value, dict):
        for key in schema.get("required", []) or []:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                errors.extend(validate_against_schema(value[key], sub, f"{path}.{key}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate_against_schema(item, schema["items"], f"{path}[{i}]"))

    return errors


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True
