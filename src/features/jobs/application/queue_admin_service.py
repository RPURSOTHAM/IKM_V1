from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests
from fastapi import HTTPException
from pydantic import BaseModel, Field
from requests.auth import HTTPBasicAuth

from src.features.documents.infrastructure.message_publisher import QueuePublisher
from src.features.configuration.platform_settings import settings


class QueueTestDocumentMessage(BaseModel):
    document_id: str = Field(..., min_length=1)
    document_name: str = Field(..., min_length=1)
    collection_name: str = Field(default="DocumentChunk", min_length=1)
    tenant_id: str | None = None


def _is_production() -> bool:
    return settings.app_env in ("production", "prod")


def _management_auth() -> HTTPBasicAuth:
    return HTTPBasicAuth(settings.rabbitmq_management_user, settings.rabbitmq_management_pass)


def _vhost_encoded(vhost: str) -> str:
    if vhost in ("", "default", "/"):
        return quote("/", safe="")
    return quote(vhost, safe="")


def require_queue_admin_enabled() -> None:
    if not settings.enable_queue_admin:
        raise HTTPException(
            status_code=503,
            detail="Queue admin is disabled. Set RAG_API_ENABLE_QUEUE_ADMIN=true.",
        )


def _management_get(path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{settings.rabbitmq_management_url.rstrip('/')}/api{path}"
    r = requests.get(url, auth=_management_auth(), params=params or {}, timeout=settings.rabbitmq_management_timeout)
    if r.status_code == 401:
        raise HTTPException(status_code=502, detail="RabbitMQ management API authentication failed.")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"RabbitMQ management API error: {r.status_code} {r.text[:500]}")
    return r.json() if r.text else None


def _management_post(path: str, body: dict[str, Any]) -> Any:
    url = f"{settings.rabbitmq_management_url.rstrip('/')}/api{path}"
    r = requests.post(
        url,
        auth=_management_auth(),
        json=body,
        timeout=settings.rabbitmq_management_timeout,
    )
    if r.status_code == 401:
        raise HTTPException(status_code=502, detail="RabbitMQ management API authentication failed.")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"RabbitMQ management API error: {r.status_code} {r.text[:500]}")
    return r.json() if r.text else None


def list_queues(
    *,
    vhost: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    require_queue_admin_enabled()
    try:
        params: dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if page_size is not None:
            params["page_size"] = page_size
        if vhost is not None:
            ve = _vhost_encoded(vhost)
            data = _management_get(f"/queues/{ve}", params=params or None)
        else:
            data = _management_get("/queues", params=params or None)
        if data is None:
            data = []
        if not isinstance(data, list):
            raise HTTPException(status_code=502, detail="Unexpected RabbitMQ management response.")
        slim = [
            {
                "name": q.get("name"),
                "vhost": q.get("vhost"),
                "messages": q.get("messages"),
                "messages_ready": q.get("messages_ready"),
                "messages_unacknowledged": q.get("messages_unacknowledged"),
                "state": q.get("state"),
                "durable": q.get("durable"),
                "auto_delete": q.get("auto_delete"),
            }
            for q in data
        ]
        return {"queues": slim, "count": len(slim)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list queues: {exc}") from exc


def peek_queue_messages(
    vhost: str,
    queue_name: str,
    *,
    count: int = 50,
    encoding: str = "auto",
) -> dict[str, Any]:
    require_queue_admin_enabled()
    ve = _vhost_encoded(vhost)
    qe = quote(queue_name, safe="")
    try:
        body = {"count": count, "ackmode": "ack_requeue_true", "encoding": encoding}
        data = _management_post(f"/queues/{ve}/{qe}/get", body)
        messages = data if isinstance(data, list) else []
        return {
            "vhost": "/" if vhost in ("default", "/") else vhost,
            "queue": queue_name,
            "count_returned": len(messages),
            "messages": messages,
            "note": "Messages were requeued (ack_requeue_true). Use DELETE .../messages?count=N to remove from head.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to read queue messages: {exc}") from exc


def delete_queue_messages(vhost: str, queue_name: str, *, count: int = 1) -> dict[str, Any]:
    require_queue_admin_enabled()
    ve = _vhost_encoded(vhost)
    qe = quote(queue_name, safe="")
    removed = 0
    last_batch: list[Any] = []
    try:
        for _ in range(count):
            batch = _management_post(
                f"/queues/{ve}/{qe}/get",
                {"count": 1, "ackmode": "ack_requeue_false", "encoding": "auto"},
            )
            if not batch:
                break
            if isinstance(batch, list) and len(batch) == 0:
                break
            last_batch = batch if isinstance(batch, list) else [batch]
            removed += 1
        return {
            "vhost": "/" if vhost in ("default", "/") else vhost,
            "queue": queue_name,
            "removed": removed,
            "last_removed": last_batch[-1] if last_batch else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to delete queue messages: {exc}") from exc


def publish_test_document_message(
    vhost: str,
    queue_name: str,
    body: QueueTestDocumentMessage,
) -> dict[str, Any]:
    require_queue_admin_enabled()
    if _is_production():
        raise HTTPException(
            status_code=403,
            detail="Test document messages are disabled in production (RAG_API_ENV / APP_ENV).",
        )
    try:
        payload = {
            "document_id": body.document_id,
            "document_name": body.document_name,
            "collection_name": body.collection_name,
            "tenant_id": body.tenant_id,
        }
        vh = "/" if vhost in ("default", "/", "") else vhost
        publisher = QueuePublisher(
            settings.rabbitmq_host,
            settings.rabbitmq_port,
            settings.rabbitmq_user,
            settings.rabbitmq_pass,
            queue_name,
        )
        publisher.publish_json_to_queue(queue_name, payload, virtual_host=vh)
        return {
            "published": True,
            "queue": queue_name,
            "vhost": vh,
            "payload": payload,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to publish test message: {exc}") from exc
