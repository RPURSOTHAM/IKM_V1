from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.infrastructure.vector_store.shared_weaviate.tenancy import collection_multi_tenancy_enabled, resolve_collection


def test_collection_multi_tenancy_enabled_reads_config_flag():
    collection = MagicMock()
    collection.config.get.return_value = SimpleNamespace(
        multi_tenancy_config=SimpleNamespace(enabled=True),
    )
    assert collection_multi_tenancy_enabled(collection) is True


def test_collection_multi_tenancy_disabled_when_config_missing():
    collection = MagicMock()
    collection.config.get.return_value = SimpleNamespace(multi_tenancy_config=None)
    assert collection_multi_tenancy_enabled(collection) is False


def test_resolve_collection_applies_tenant_when_enabled():
    collection = MagicMock(name="SOP")
    tenant_collection = MagicMock()
    collection.with_tenant.return_value = tenant_collection
    collection.config.get.return_value = SimpleNamespace(
        multi_tenancy_config=SimpleNamespace(enabled=True),
    )

    resolved = resolve_collection(collection, "tenant-a")

    collection.with_tenant.assert_called_once_with("tenant-a")
    assert resolved is tenant_collection


def test_resolve_collection_ignores_tenant_when_disabled():
    collection = MagicMock(name="SOP")
    collection.config.get.return_value = SimpleNamespace(
        multi_tenancy_config=SimpleNamespace(enabled=False),
    )

    resolved = resolve_collection(collection, "tenant-a")

    collection.with_tenant.assert_not_called()
    assert resolved is collection
