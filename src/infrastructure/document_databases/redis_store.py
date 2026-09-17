"""Redis persistence for render-ready document payloads."""

from __future__ import annotations

import json
import os
from typing import Any


def _redis_client():
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "redis client is not installed. Add 'redis>=5.0.0' to processor-service dependencies."
        ) from exc

    host = (os.getenv("REDIS_HOST") or "localhost").strip()
    port = int(os.getenv("REDIS_PORT") or "6379")
    password = os.getenv("REDIS_PASSWORD")
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)


def render_cache_key(document_id: str) -> str:
    return f"render:{document_id}"


def get_render_payload(*, document_id: str) -> dict[str, Any] | None:
    """Load a cached render payload written by conversion_for_rendering."""
    client = _redis_client()
    raw = client.get(render_cache_key(document_id))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def store_render_payload(*, document_id: str, payload: dict[str, Any], ttl_seconds: int | None = None) -> str:
    key = render_cache_key(document_id)
    client = _redis_client()
    client.set(key, json.dumps(payload, default=str))
    if ttl_seconds and ttl_seconds > 0:
        client.expire(key, ttl_seconds)
    return f"redis://{key}"
