"""Observability framework unit tests."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from src.features.observability.audit.domain.audit_models import AuditEventRecord, AuditQuery
from src.features.observability.audit.application.audit_service import AuditService
from src.features.observability.audit.application.change_tracker import build_change_metadata, get_changes, get_deep_changes
from src.features.observability.metrics.domain.metrics_models import MetricEventRecord, MetricsQuery
from src.features.observability.metrics.application.metrics_service import MetricsService
from src.features.observability.middleware.request_context import (
    apply_synthetic_context,
    extract_call_context,
    reset_context,
    timing_metadata,
)


class TestChangeTracker:
    def test_get_changes_modified_only(self):
        old = {"chunk_size": 512, "model": "bge-base"}
        new = {"chunk_size": 1024, "model": "bge-base"}
        changes = get_changes(old, new)
        assert len(changes) == 1
        assert changes[0]["field"] == "chunk_size"
        assert changes[0]["old"] == 512
        assert changes[0]["new"] == 1024

    def test_get_deep_changes_nested_settings(self):
        old = {
            "embedding_model": {"model_id": "bge-small", "provider": "local"},
            "chunk_size": 512,
            "retrieval_search_mode": "vector",
        }
        new = {
            "embedding_model": {"model_id": "bge-large", "provider": "local"},
            "chunk_size": 1024,
            "retrieval_search_mode": "hybrid",
        }
        changes = get_deep_changes(old, new)
        fields = {c["field"] for c in changes}
        assert "embedding_model.model_id" in fields
        assert "chunk_size" in fields
        assert "retrieval_search_mode" in fields
        summary = build_change_metadata(changes)
        assert summary["number_of_fields_changed"] == 3
        assert "Embedding Model" in summary["change_summary"]
        assert "Search Mode" in summary["change_summary"]

    def test_get_changes_nested_metadata(self):
        old = {"metadata": {"title": "A"}, "embedding_model": "m1"}
        new = {"metadata": {"title": "B"}, "embedding_model": "m1"}
        changes = get_changes(old, new)
        assert len(changes) == 1
        assert changes[0]["field"] == "metadata"

    def test_get_changes_no_diff(self):
        assert get_changes({"a": 1}, {"a": 1}) == []


class TestContextPropagation:
    def test_synthetic_context(self):
        tokens = apply_synthetic_context(
            request_id="req-1",
            correlation_id="corr-1",
            application_name="claims-portal",
            document_id="doc-1",
        )
        from src.features.observability.middleware.request_context import (
            get_application_name,
            get_correlation_id,
            get_request_id,
        )

        assert get_request_id() == "req-1"
        assert get_correlation_id() == "corr-1"
        assert get_application_name() == "claims-portal"
        reset_context(tokens)

    def test_extract_call_context_from_request_object(self):
        class Req:
            document_id = "d-99"
            repository_id = "r-1"

        ids = extract_call_context((Req(),), {})
        assert ids["document_id"] == "d-99"
        assert ids["repository_id"] == "r-1"

    def test_timing_metadata(self):
        meta = timing_metadata(1.0, 2.0, func_name="fn")
        assert meta["duration_ms"] == 1000.0
        assert meta["function"] == "fn"


class TestUserAuditContext:
    def test_enrich_audit_metadata_from_context(self):
        from src.features.observability.audit.application.user_context import enrich_audit_metadata
        from src.features.observability.middleware.request_context import apply_synthetic_context, reset_context

        tokens = apply_synthetic_context(
            request_id="req-42",
            correlation_id="corr-42",
            application_name="claims-portal",
            user_id="alice",
            metadata={
                "username": "alice",
                "display_name": "Alice Admin",
                "role": "Admin",
                "authentication_provider": "Local",
                "client_ip": "10.0.0.5",
            },
        )
        try:
            meta = enrich_audit_metadata({"repository_id": "repo-1"})
            assert meta["user_id"] == "alice"
            assert meta["application_name"] == "claims-portal"
            assert meta["display_name"] == "Alice Admin"
            assert meta["role"] == "Admin"
            assert meta["request_id"] == "req-42"
            assert meta["correlation_id"] == "corr-42"
            assert meta["client_ip"] == "10.0.0.5"
            assert meta["user"]["username"] == "alice"
        finally:
            reset_context(tokens)


class TestAuditService:
    def test_record_does_not_raise_when_db_unavailable(self):
        with patch("src.features.observability.audit.application.audit_service.get_audit_repository", return_value=None):
            AuditService().record("TEST_EVENT", entity_type="test", entity_id="1")

    def test_record_change_skips_empty(self):
        svc = AuditService()
        with patch.object(svc, "_async_persist") as mock_persist:
            svc.record_change("X", entity_type="r", entity_id="1", old={"a": 1}, new={"a": 1})
            mock_persist.assert_not_called()

    def test_record_failure_swallows_db_errors(self):
        svc = AuditService()
        with patch.object(svc, "_async_persist") as mock_async:
            svc.record_failure("FAIL", error="oops")
            mock_async.assert_called_once()

    def test_record_captures_application_and_user_context(self):
        svc = AuditService()
        tokens = apply_synthetic_context(
            application_name="claims-portal",
            user_id="alice",
        )
        try:
            with patch.object(svc, "_async_persist") as mock_persist:
                svc.record("SEARCH_COMPLETED", action="search")
                event = mock_persist.call_args.args[0]
                assert event.application_name == "claims-portal"
                assert event.user_id == "alice"
        finally:
            reset_context(tokens)


class TestMetricsService:
    def test_record_does_not_raise_when_db_unavailable(self):
        with patch("src.features.observability.metrics.application.metrics_service.get_metrics_repository", return_value=None):
            MetricsService().record("embedding", duration_ms=10.0)

    @patch("src.features.observability.metrics.application.metrics_service.threading.Thread")
    def test_async_persist_uses_thread(self, mock_thread):
        mock_thread.return_value = MagicMock()
        with patch("src.features.observability.metrics.application.metrics_service.get_metrics_repository") as mock_repo:
            mock_repo.return_value.insert_event = MagicMock(return_value=1)
            svc = MetricsService()
            svc.record("chunking", duration_ms=5.0)
            mock_thread.assert_called()

    @patch("src.features.observability.metrics.application.metrics_service.threading.Thread")
    def test_non_benchmark_metric_is_skipped(self, mock_thread):
        mock_thread.return_value = MagicMock()
        with patch("src.features.observability.metrics.application.metrics_service.get_metrics_repository") as mock_repo:
            mock_repo.return_value.insert_event = MagicMock(return_value=1)
            MetricsService().record("http_request", duration_ms=1.0)
            mock_thread.assert_not_called()


class TestBenchmarkMetrics:
    def test_catalog_matches_excel_row_count(self):
        from src.features.observability.metrics.application.benchmark_metrics import BENCHMARK_METRICS, benchmark_summary

        assert len(BENCHMARK_METRICS) == 60
        summary = benchmark_summary()
        assert summary["total"] == 60


class TestDecorators:
    def test_capture_metric_success_metadata(self):
        from src.features.observability.metrics.application.metrics_decorator import capture_metric

        class Req:
            document_id = "doc-1"

        @capture_metric("test_metric")
        def work(req):
            return 42

        tokens = apply_synthetic_context(request_id="r1", correlation_id="c1")
        try:
            with patch("src.features.observability.metrics.application.metrics_decorator.get_metrics_service") as mock_svc:
                mock_svc.return_value.record = MagicMock()
                assert work(Req()) == 42
                kwargs = mock_svc.return_value.record.call_args.kwargs
                assert kwargs["status"] == "success"
                assert kwargs["metadata"]["request_id"] == "r1"
                assert kwargs["document_id"] == "doc-1"
        finally:
            reset_context(tokens)

    def test_capture_metric_failure(self):
        from src.features.observability.metrics.application.metrics_decorator import capture_metric

        @capture_metric("test_metric")
        def fail():
            raise ValueError("boom")

        with patch("src.features.observability.metrics.application.metrics_decorator.get_metrics_service") as mock_svc:
            mock_svc.return_value.record = MagicMock()
            with pytest.raises(ValueError):
                fail()
            assert mock_svc.return_value.record.call_args.kwargs["status"] == "failure"

    def test_audit_event_decorator(self):
        from src.features.observability.audit.application.audit_decorator import audit_event

        @audit_event("EMBEDDING", entity_type="document")
        def embed():
            return "ok"

        with patch("src.features.observability.audit.application.audit_decorator.get_audit_service") as mock_audit:
            mock_audit.return_value.record = MagicMock()
            assert embed() == "ok"
            mock_audit.return_value.record.assert_called_once()
            assert "duration_ms" in mock_audit.return_value.record.call_args.kwargs["metadata"]


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_observability_middleware_sets_headers(self):
        from src.features.observability.middleware.observability_middleware import ObservabilityMiddleware

        async def app(scope, receive, send):
            response = Response("ok", status_code=200)
            await response(scope, receive, send)

        middleware = ObservabilityMiddleware(app)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "headers": [],
            "query_string": b"",
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "scheme": "http",
            "root_path": "",
        }
        with (
            patch("src.features.observability.middleware.observability_middleware.get_metrics_service") as mock_metrics,
            patch("src.features.observability.middleware.observability_middleware.get_audit_service") as mock_audit,
        ):
            mock_metrics.return_value.record_http_request = MagicMock()
            request = Request(scope)

            async def call_next(req):
                return Response("ok", status_code=200)

            response = await middleware.dispatch(request, call_next)
            assert response.status_code == 200
            assert response.headers.get("X-Request-ID")
            assert response.headers.get("X-Correlation-ID")
            mock_metrics.return_value.record_http_request.assert_called_once()
            mock_audit.return_value.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_observability_middleware_binds_application_and_user_headers(self):
        from src.features.observability.middleware.observability_middleware import ObservabilityMiddleware
        from src.features.observability.middleware.request_context import current_context

        async def app(scope, receive, send):
            response = Response("ok", status_code=200)
            await response(scope, receive, send)

        middleware = ObservabilityMiddleware(app)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/retrieve",
            "headers": [
                (b"x-application-name", b"claims-portal"),
                (b"x-user-id", b"alice"),
            ],
            "query_string": b"",
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "scheme": "http",
            "root_path": "",
        }
        observed = {}

        async def call_next(req):
            ctx = current_context()
            observed.update(application_name=ctx.application_name, user_id=ctx.user_id)
            return Response("ok", status_code=200)

        with (
            patch("src.features.observability.middleware.observability_middleware.get_metrics_service"),
            patch("src.features.observability.middleware.observability_middleware.get_audit_service") as mock_audit,
        ):
            await middleware.dispatch(Request(scope), call_next)

        assert observed == {"application_name": "claims-portal", "user_id": "alice"}
        assert mock_audit.return_value.record.call_args.kwargs["action"] == "search"


class TestApis:
    def test_audit_events_route_empty_when_no_db(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.features.observability.audit.audit_api import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with patch("src.features.observability.audit.audit_api.get_audit_repository", return_value=None):
            client = TestClient(app)
            resp = client.get("/api/v1/audit/events")
            assert resp.status_code == 200
            assert resp.json()["items"] == []

    def test_metrics_summary_route_empty_when_no_db(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.features.observability.metrics.metrics_api import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        empty_summary = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
        }
        mock_agg = MagicMock()
        mock_agg.summary.return_value = empty_summary
        with (
            patch("src.features.observability.metrics.metrics_api.get_metrics_repository", return_value=None),
            patch("src.features.observability.metrics.metrics_api.get_aggregation_service", return_value=mock_agg),
        ):
            client = TestClient(app)
            resp = client.get("/api/v1/metrics/summary")
            assert resp.status_code == 200
            body = resp.json()
            assert body["summary"]["total"] == 0

    def test_application_usage_metrics(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.features.observability.metrics.api.metrics_routes import router

        repo = MagicMock()
        repo.application_usage.return_value = [
            {
                "application_name": "claims-portal",
                "total_requests": 8,
                "documents_uploaded": 2,
                "search_requests": 4,
                "ai_queries": 1,
                "documents_deleted": 1,
                "failed_requests": 1,
                "active_users": 3,
            }
        ]
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with patch(
            "src.features.observability.metrics.api.metrics_routes.get_audit_repository",
            return_value=repo,
        ):
            response = TestClient(app).get("/api/v1/metrics/applications")
        assert response.status_code == 200
        assert response.json()["totals"]["documents_uploaded"] == 2
        assert response.json()["applications"][0]["application_name"] == "claims-portal"

    def test_audit_get_by_id_503_when_unavailable(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.features.observability.audit.audit_api import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        with patch("src.features.observability.audit.audit_api.get_audit_repository", return_value=None):
            client = TestClient(app)
            resp = client.get("/api/v1/audit/1")
            assert resp.status_code == 503


@pytest.mark.skipif(
    os.getenv("OBSERVABILITY_TEST_POSTGRES", "").lower() not in {"1", "true", "yes"},
    reason="Set OBSERVABILITY_TEST_POSTGRES=1 to run PostgreSQL integration tests",
)
class TestPostgresIntegration:
    @pytest.fixture(autouse=True)
    def setup_db(self):
        from src.features.observability.infrastructure.db import get_observability_db, reset_observability_db_for_tests
        from src.features.observability.audit.infrastructure.audit_repository import reset_audit_repository_for_tests
        from src.features.observability.metrics.infrastructure.metrics_repository import reset_metrics_repository_for_tests

        reset_observability_db_for_tests()
        reset_audit_repository_for_tests()
        reset_metrics_repository_for_tests()
        db = get_observability_db()
        assert db is not None and db.ping()
        yield
        reset_observability_db_for_tests()
        reset_audit_repository_for_tests()
        reset_metrics_repository_for_tests()

    def test_audit_persistence_and_query(self):
        from src.features.observability.audit.infrastructure.audit_repository import get_audit_repository

        repo = get_audit_repository()
        assert repo is not None
        rid = str(uuid.uuid4())
        event_id = repo.insert(
            AuditEventRecord(
                timestamp=datetime.now(timezone.utc),
                request_id=rid,
                correlation_id=rid,
                event_type="DOCUMENT_UPLOADED",
                entity_type="document",
                entity_id="doc-test",
                status="success",
                changes=[{"field": "status", "old": "new", "new": "uploaded"}],
            )
        )
        assert event_id is not None
        time.sleep(0.2)
        items, total = repo.list_events(AuditQuery(request_id=rid))
        assert total >= 1
        fetched = repo.get_by_id(event_id)
        assert fetched is not None
        assert fetched["event_type"] == "DOCUMENT_UPLOADED"

    def test_metrics_persistence_and_api_filter(self):
        from src.features.observability.metrics.infrastructure.metrics_repository import get_metrics_repository

        repo = get_metrics_repository()
        assert repo is not None
        rid = str(uuid.uuid4())
        repo.insert_event(
            MetricEventRecord(
                timestamp=datetime.now(timezone.utc),
                request_id=rid,
                metric_name="embedding",
                duration_ms=42.0,
                status="success",
                document_id="doc-1",
            )
        )
        time.sleep(0.2)
        items, total = repo.list_events(MetricsQuery(request_id=rid, metric_name="embedding"))
        assert total >= 1
        assert items[0]["metric_name"] == "embedding"

    def test_correlation_id_in_audit(self):
        from src.features.observability.audit.infrastructure.audit_repository import get_audit_repository

        corr = f"corr-{uuid.uuid4()}"
        tokens = apply_synthetic_context(request_id="r1", correlation_id=corr)
        try:
            from src.features.observability.audit.application.audit_service import get_audit_service

            get_audit_service().record("TEST_CORR", entity_type="test", entity_id="1")
            time.sleep(0.3)
            repo = get_audit_repository()
            items, _ = repo.list_events(AuditQuery(correlation_id=corr))
            assert any(i.get("correlation_id") == corr for i in items)
        finally:
            reset_context(tokens)
