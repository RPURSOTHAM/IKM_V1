from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_raw_user_identifier(value: str, owner_user_id: str = "") -> bool:
    """True when a value is a user id token, not a human-readable display name."""
    token = str(value or "").strip()
    if not token:
        return True
    owner = str(owner_user_id or "").strip()
    if owner and token.lower() == owner.lower():
        return True
    return bool(_UUID_RE.match(token))


def resolve_owner_user_name(
    owner_user_id: str,
    *,
    stored_name: str | None = None,
    platform_names: dict[str, str] | None = None,
) -> str | None:
    stored = str(stored_name or "").strip()
    if stored and not is_raw_user_identifier(stored, owner_user_id):
        return stored
    owner_id = str(owner_user_id or "").strip()
    if not owner_id:
        return None
    if platform_names:
        platform_name = str(platform_names.get(owner_id) or "").strip()
        if platform_name:
            return platform_name
    return None


def load_platform_display_names(user_ids: Iterable[str]) -> dict[str, str]:
    ids = {str(user_id).strip() for user_id in user_ids if str(user_id or "").strip()}
    if not ids:
        return {}

    try:
        from src.features.users.infrastructure.user_repository import get_platform_security_store

        store = get_platform_security_store()
        if store is None:
            return {}
        return store.get_user_display_names(ids)
    except Exception:
        logger.debug("Platform user display-name lookup failed", exc_info=True)
        return {}
