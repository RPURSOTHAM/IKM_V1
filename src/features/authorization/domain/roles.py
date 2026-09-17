"""Catalog of assignable platform roles for user administration UIs and API clients."""

from __future__ import annotations

from typing import Any

from src.features.users.infrastructure.user_repository import PLATFORM_ROLES

# role_id, label, description, sequence, assignable_on_create
_PLATFORM_ROLE_DEFINITIONS: tuple[tuple[str, str, str, int, bool], ...] = (
    (
        "administrator",
        "Administrator",
        "Full platform access: user administration, configuration, repository registration approval, and global audit.",
        1,
        False,
    ),
    (
        "platform_owner",
        "Platform Owner",
        "Platform management delegate; cannot create other Platform Owners or additional Administrators.",
        2,
        False,
    ),
)


def build_platform_role_catalog(*, api_prefix: str = "/api/v1") -> dict[str, Any]:
    prefix = api_prefix.rstrip("/")
    roles: list[dict[str, Any]] = []
    assignable_on_create: list[str] = []

    for role_id, label, description, sequence, assignable_on_create_flag in _PLATFORM_ROLE_DEFINITIONS:
        if role_id not in PLATFORM_ROLES:
            continue
        entry: dict[str, Any] = {
            "role_id": role_id,
            "label": label,
            "description": description,
            "sequence": sequence,
            "assignable_on_create": assignable_on_create_flag,
            "grant_endpoint": None,
            "revoke_endpoint": None,
            "grant_requires_role": None,
        }
        if role_id == "platform_owner":
            owner_path = f"{prefix}/platform/users/{{user_id}}/roles/platform-owner"
            entry["grant_endpoint"] = owner_path
            entry["revoke_endpoint"] = owner_path
            entry["grant_requires_role"] = "administrator"
        roles.append(entry)
        if assignable_on_create_flag:
            assignable_on_create.append(role_id)

    roles.sort(key=lambda item: int(item["sequence"]))

    return {
        "roles": roles,
        "count": len(roles),
        "assignable_on_create": assignable_on_create,
        "role_ids": [item["role_id"] for item in roles],
    }
