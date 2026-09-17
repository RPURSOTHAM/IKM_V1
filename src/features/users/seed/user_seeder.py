from __future__ import annotations

import logging

from src.features.authentication.configuration.authentication_config import bootstrap_admin_password, bootstrap_admin_user
from src.features.authentication.application.password_service import hash_password
from src.features.users.infrastructure.user_repository import PlatformSecurityStore

logger = logging.getLogger(__name__)


def seed_bootstrap_admin(store: PlatformSecurityStore) -> bool:
    if store.has_administrator():
        return False

    password = bootstrap_admin_password()
    if not password:
        logger.warning(
            "No PLATFORM_BOOTSTRAP_ADMIN_PASSWORD set; skipping bootstrap administrator seed"
        )
        return False

    user_id = bootstrap_admin_user()
    store.insert_user(
        user_id=user_id,
        password_hash=hash_password(password),
        platform_role="administrator",
        display_name="Platform Administrator",
        must_change_password=True,
        created_by=None,
    )
    logger.info("Seeded bootstrap platform administrator user_id=%s", user_id)
    return True
