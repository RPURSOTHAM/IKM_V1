from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def collection_multi_tenancy_enabled(collection: Any) -> bool:
    """Return True when the Weaviate collection has multi-tenancy enabled."""
    try:
        config = collection.config.get()
        mt = getattr(config, "multi_tenancy_config", None)
        if mt is None:
            return False
        return bool(getattr(mt, "enabled", False))
    except Exception as exc:
        logger.debug("Could not read multi-tenancy config: %s", exc)
        return False


def resolve_collection(collection: Any, tenant: str | None) -> Any:
    """Apply tenant scoping only when the collection supports multi-tenancy."""
    tenant_value = str(tenant).strip() if tenant else ""
    if not tenant_value:
        return collection
    if collection_multi_tenancy_enabled(collection):
        return collection.with_tenant(tenant_value)
    logger.info(
        "Collection '%s' has multi-tenancy disabled; ignoring tenant %r.",
        getattr(collection, "name", collection),
        tenant_value,
    )
    return collection
