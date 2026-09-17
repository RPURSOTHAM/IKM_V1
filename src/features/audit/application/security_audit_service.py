from __future__ import annotations

import logging
from typing import Any

from src.features.authorization.application.authorization_service import AuthenticatedUser, user_to_audit_actor
from src.features.users.infrastructure.user_repository import PlatformSecurityStore

logger = logging.getLogger(__name__)


class AuditService:
    def __init__(self, store: PlatformSecurityStore) -> None:
        self._store = store

    def record(
        self,
        *,
        event_category: str,
        event_type: str,
        action: str,
        outcome: str,
        actor: AuthenticatedUser | None = None,
        repository_role: str | None = None,
        repository_id: str | None = None,
        document_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        old_value: Any = None,
        new_value: Any = None,
        change_reason: str | None = None,
        request_id: str | None = None,
        client_ip: str | None = None,
        service_component: str = "dms_api",
    ) -> str | None:
        event: dict[str, Any] = {
            "event_category": event_category,
            "event_type": event_type,
            "action": action,
            "outcome": outcome,
            "repository_id": repository_id,
            "document_id": document_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "old_value_json": old_value,
            "new_value_json": new_value,
            "change_reason": change_reason,
            "request_id": request_id,
            "client_ip": client_ip,
            "service_component": service_component,
        }
        if actor:
            event.update(user_to_audit_actor(actor, repository_role))
        try:
            return self._store.insert_audit_event(event)
        except Exception:
            logger.exception("Failed to write audit event %s", event_type)
            return None
