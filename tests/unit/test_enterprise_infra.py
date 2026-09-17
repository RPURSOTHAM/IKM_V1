"""Enterprise infrastructure unit tests."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from src.infrastructure.application_support.async_exec import gather_bounded, map_sync, run_async, run_sync
from src.infrastructure.application_support.health import HealthRegistry
from src.infrastructure.application_support.logging_json import JsonFormatter, configure_json_logging
from src.infrastructure.application_support.metrics import get_metrics, reset_metrics
from src.infrastructure.application_support.registry import (
    ConfigDrivenModelLoader,
    PluginRegistry,
    ServiceContainer,
    reset_infra_globals,
)
from src.infrastructure.application_support.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryPolicy,
    reset_circuit_breakers,
)
from src.infrastructure.application_support.tracing import start_span


@pytest.fixture(autouse=True)
def _reset_infra_state(monkeypatch):
    reset_infra_globals()
    reset_metrics()
    reset_circuit_breakers()
    monkeypatch.setenv("INFRA_JSON_LOGGING", "true")
    monkeypatch.setenv("INFRA_PROMETHEUS_ENABLED", "true")
    monkeypatch.delenv("INFRA_OTEL_ENABLED", raising=False)
    yield
    reset_infra_globals()
    reset_metrics()
    reset_circuit_breakers()


def test_json_formatter_emits_structured_fields() -> None:
    formatter = JsonFormatter(service="test-service")
    record = logging.LogRecord(
        name="unit",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.request_id = "req-1"
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hello world"
    assert payload["service"] == "test-service"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "req-1"
    assert "ts" in payload


def test_configure_json_logging_sets_json_handler() -> None:
    logger = configure_json_logging(service="cfg-test", force=True)
    assert logger is not None
    root = logging.getLogger()
    assert root.handlers
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_plugin_registry_and_config_driven_model_loading(monkeypatch) -> None:
    registry = PluginRegistry()
    created: list[str] = []

    class FakeEmbedder:
        def __init__(self, name: str) -> None:
            self.name = name

    registry.register("embedder", "minilm", lambda: FakeEmbedder("minilm"), default=True)
    registry.register("embedder", "bge-m3", lambda: FakeEmbedder("bge-m3"))

    monkeypatch.setenv("INFRA_EMBEDDER_MODEL", "bge-m3")
    loader = ConfigDrivenModelLoader(registry)
    assert loader.resolve("embedder") == "bge-m3"
    instance = loader.get("embedder")
    assert instance.name == "bge-m3"

    # Preferred override wins over env
    assert loader.resolve("embedder", preferred="minilm") == "minilm"
    catalog = loader.catalog()
    assert "embedder" in catalog["roles"]
    assert any(p["name"] == "bge-m3" for p in catalog["plugins"])


def test_service_container_singleton_di() -> None:
    container = ServiceContainer()
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return {"id": counter["n"]}

    container.register_factory("svc", factory, singleton=True)
    a = container.get("svc")
    b = container.get("svc")
    assert a is b
    assert counter["n"] == 1


def test_retry_policy_retries_then_succeeds() -> None:
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    policy = RetryPolicy(max_attempts=3, initial_delay_seconds=0.01, jitter=0.0)
    assert policy.run(flaky) == "ok"
    assert attempts["n"] == 3


def test_circuit_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker("demo", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=60))

    def boom():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        breaker.run(boom)
    with pytest.raises(ValueError):
        breaker.run(boom)
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.run(lambda: "never")


def test_metrics_fallback_render_contains_counters() -> None:
    metrics = get_metrics()
    metrics.observe_request("dms-service", "/chat", "200", 0.12)
    body = metrics.render()
    assert "rag_requests_total" in body
    assert "rag_request_latency_seconds" in body
    assert "rag_process_uptime_seconds" in body


def test_health_readiness_reports_critical_failure() -> None:
    registry = HealthRegistry("demo")
    registry.add("ok_dep", lambda: True, critical=True)
    registry.add("bad_dep", lambda: {"ok": False, "reason": "down"}, critical=True)
    registry.add("optional", lambda: False, critical=False)
    report = registry.readiness()
    assert report.ready is False
    assert report.status == "degraded"
    assert report.checks["ok_dep"]["ok"] is True
    assert report.checks["bad_dep"]["ok"] is False


def test_start_span_noop_works_without_otel() -> None:
    with start_span("unit-test", component="infra") as span:
        span.set_attribute("k", "v")
        assert span.trace_id
        assert span.span_id


def test_async_helpers() -> None:
    async def main():
        value = await run_sync(lambda: 41 + 1)
        assert value == 42
        results = await map_sync(lambda x: x * 2, [1, 2, 3], limit=2)
        assert results == [2, 4, 6]
        gathered = await gather_bounded([run_sync(lambda: "a"), run_sync(lambda: "b")], limit=1)
        assert gathered == ["a", "b"]

    asyncio.run(main())
    assert run_async(run_sync(lambda: "sync-from-async-helper")) == "sync-from-async-helper"


def test_bootstrap_registers_all_component_roles() -> None:
    from src.infrastructure.application_support.bootstrap import bootstrap_infra
    from src.infrastructure.application_support.interfaces import COMPONENT_ROLES

    loader = bootstrap_infra(service="unit-test", force=True)
    catalog = loader.catalog()
    registered_roles = {p["role"] for p in catalog["plugins"]}
    for role in COMPONENT_ROLES:
        assert role in registered_roles


def test_dms_infra_routes_exist() -> None:
    from tests.unit.weaviate_test_stubs import install_weaviate_stubs

    install_weaviate_stubs()
    from fastapi.testclient import TestClient

    from src.application.consumer_api.main import create_app

    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert "checks" in ready.json()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "rag_" in metrics.text or "process_uptime" in metrics.text

    plugins = client.get("/infra/plugins")
    assert plugins.status_code == 200
    assert "plugins" in plugins.json()
