from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.features.repositories.application import temporary_repository_service as temporary
from src.features.repositories.domain.repository_policy import normalize_repository_name


def test_missing_repository_is_the_only_temporary_upload_trigger() -> None:
    assert temporary.should_route_to_temporary_repository(None)
    assert temporary.should_route_to_temporary_repository("")
    assert temporary.should_route_to_temporary_repository(" ")
    assert not temporary.should_route_to_temporary_repository("repository-id")


def test_system_temporary_repository_batch_name_is_allowed() -> None:
    assert normalize_repository_name("_temp") == "_temp"
    assert normalize_repository_name("_temp_550e8400-e29b-41d4-a716-446655440000") == "_temp_550e8400-e29b-41d4-a716-446655440000"


def test_temporary_configuration_is_environment_controlled(monkeypatch) -> None:
    monkeypatch.setenv("TEMP_REPOSITORY_ENABLED", "true")
    monkeypatch.setenv("TEMP_REPOSITORY_NAME", "_temp")
    monkeypatch.setenv("TEMP_REPOSITORY_RETENTION_HOURS", "48")
    monkeypatch.setenv("TEMP_REPOSITORY_TEST_MODE", "true")
    monkeypatch.setenv("TEMP_REPOSITORY_TEST_RETENTION_SECONDS", "60")
    monkeypatch.setenv("TEMP_REPOSITORY_CLEANUP_INTERVAL_SECONDS", "60")

    cfg = temporary.get_temporary_repository_config()

    assert cfg.enabled is True
    assert cfg.name == "_temp"
    assert cfg.retention_hours == 48
    assert cfg.retention_seconds == 60
    assert cfg.test_mode is True
    assert cfg.cleanup_interval_seconds == 60


def test_production_retention_ignores_test_seconds_when_test_mode_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TEMP_REPOSITORY_RETENTION_HOURS", "24")
    monkeypatch.setenv("TEMP_REPOSITORY_TEST_MODE", "false")
    monkeypatch.setenv("TEMP_REPOSITORY_TEST_RETENTION_SECONDS", "60")

    cfg = temporary.get_temporary_repository_config()

    assert cfg.test_mode is False
    assert cfg.retention_seconds == 24 * 3600


def test_expired_temporary_repository_is_purged_before_deletion(monkeypatch) -> None:
    class FakeRepositoryService:
        def list_repositories(self) -> dict:
            return {
                "repositories": [
                    {"repository_id": "temp-id", "name": "_temp_550e8400-e29b-41d4-a716-446655440000"}
                ]
            }

        def list_documents(self, repository_id: str, *, limit: int) -> dict:
            return {
                "documents": [
                    {
                        "document_id": "doc-id",
                        "status": "completed",
                        "processing_completion_timestamp": (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp(),
                    }
                ]
            }

        def purge_all_documents(self, repository_id: str, *, delete_file: bool) -> dict:
            assert repository_id == "temp-id"
            assert delete_file is True
            return {"remaining": 0, "failed": 0, "deleted": 1}

        def delete_repository(self, repository_id: str) -> dict:
            assert repository_id == "temp-id"
            return {"repository_id": repository_id, "deleted": True}

    monkeypatch.setenv("TEMP_REPOSITORY_ENABLED", "true")
    monkeypatch.setenv("TEMP_REPOSITORY_RETENTION_HOURS", "24")
    monkeypatch.setattr(
        "src.features.repositories.application.repository_service.get_repository_service",
        lambda: FakeRepositoryService(),
    )

    result = temporary.cleanup_expired_temporary_repository()

    assert result["deleted"] is True
    assert result["deleted_count"] == 1
    assert result["results"][0]["repository"]["repository_id"] == "temp-id"
