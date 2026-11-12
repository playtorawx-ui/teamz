"""API keys + a sliding-window rate limiter.

Keys are stored hashed; the plaintext token is returned exactly once, at
creation time, and never persisted.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class ApiKey:
    id: str
    name: str
    prefix: str
    hashed: str
    rpm_limit: int
    created_at: float
    last_used_at: float | None = None
    revoked: bool = False
    _hits: list[float] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "rpmLimit": self.rpm_limit,
            "createdAt": self.created_at,
            "lastUsedAt": self.last_used_at,
            "revoked": self.revoked,
        }


class ApiKeyStore:
    def __init__(self, default_rpm: int = 120) -> None:
        self._keys: dict[str, ApiKey] = {}
        self.default_rpm = default_rpm

    def create(self, name: str = "default", rpm_limit: int | None = None) -> tuple[ApiKey, str]:
        token = f"tz_{secrets.token_urlsafe(24)}"
        key = ApiKey(
            id=f"key_{secrets.token_hex(6)}",
            name=name or "default",
            prefix=token[:10],
            hashed=_hash(token),
            rpm_limit=rpm_limit if rpm_limit is not None else self.default_rpm,
            created_at=time.time(),
        )
        self._keys[key.id] = key
        return key, token

    def verify(self, token: str | None) -> ApiKey | None:
        if not token:
            return None
        digest = _hash(token)
        for key in self._keys.values():
            if not key.revoked and secrets.compare_digest(key.hashed, digest):
                return key
        return None

    def revoke(self, key_id: str) -> bool:
        key = self._keys.get(key_id)
        if not key:
            return False
        key.revoked = True
        return True

    def list(self) -> list[dict[str, Any]]:
        return [k.as_dict() for k in self._keys.values()]

    def check_rate(self, key: ApiKey) -> tuple[bool, int]:
        """Returns ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        key._hits = [t for t in key._hits if now - t < 60.0]
        limit = key.rpm_limit or self.default_rpm
        if limit <= 0:
            return True, 0
        if len(key._hits) >= limit:
            retry_after = max(1, int(60 - (now - key._hits[0])))
            return False, retry_after
        key._hits.append(now)
        key.last_used_at = time.time()
        return True, 0
