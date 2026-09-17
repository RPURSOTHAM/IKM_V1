from __future__ import annotations

import json
import os
from typing import Any


def _redis_client():
    import redis

    host = (os.getenv("REDIS_HOST") or "localhost").strip()
    port = int(os.getenv("REDIS_PORT") or "6379")
    password = os.getenv("REDIS_PASSWORD")
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)


def get_render_payload(*, document_id: str) -> dict[str, Any] | None:
    try:
        raw = _redis_client().get(f"render:{document_id}")
    except Exception:
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def delete_render_payload(*, document_id: str) -> dict[str, Any]:
    """Delete all render-cache keys for a document."""
    client = _redis_client()
    key = f"render:{document_id}"
    deleted = 0
    try:
        if client.delete(key):
            deleted += 1
        for match_key in client.scan_iter(match=f"render:{document_id}:*"):
            if client.delete(match_key):
                deleted += 1
    except Exception as exc:
        return {"attempted": True, "deleted": deleted, "error": str(exc)}
    return {"attempted": True, "deleted": deleted, "keys_removed": deleted}


def store_render_payload(*, document_id: str, payload: dict[str, Any], ttl_seconds: int | None = None) -> None:
    client = _redis_client()
    key = f"render:{document_id}"
    client.set(key, json.dumps(payload, default=str))
    if ttl_seconds and ttl_seconds > 0:
        client.expire(key, ttl_seconds)
