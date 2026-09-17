"""Enterprise infrastructure: observability, DI, plugins, resilience."""

from src.infrastructure.application_support.async_exec import gather_bounded, map_sync, run_async, run_sync
from src.infrastructure.application_support.bootstrap import bootstrap_infra
from src.infrastructure.application_support.health import HealthRegistry, HealthReport, attach_infra_routes
from src.infrastructure.application_support.interfaces import (
    COMPONENT_ROLES,
    Chunker,
    DLP,
    Embedder,
    LLM,
    Moderation,
    NER,
    Parser,
    Retriever,
    RiskEngine,
    VectorDatabase,
)
from src.infrastructure.application_support.logging_json import ContextLoggerAdapter, JsonFormatter, configure_json_logging
from src.infrastructure.application_support.metrics import MetricsRegistry, get_metrics, reset_metrics
from src.infrastructure.application_support.registry import (
    ConfigDrivenModelLoader,
    PluginRegistry,
    ServiceContainer,
    get_container,
    get_model_loader,
    get_registry,
    reset_infra_globals,
)
from src.infrastructure.application_support.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryPolicy,
    get_circuit_breaker,
    reset_circuit_breakers,
)
from src.infrastructure.application_support.tracing import configure_tracing, start_span, tracing_enabled

__all__ = [
    "COMPONENT_ROLES",
    "Chunker",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "ConfigDrivenModelLoader",
    "ContextLoggerAdapter",
    "DLP",
    "Embedder",
    "HealthRegistry",
    "HealthReport",
    "JsonFormatter",
    "LLM",
    "MetricsRegistry",
    "Moderation",
    "NER",
    "Parser",
    "PluginRegistry",
    "Retriever",
    "RetryPolicy",
    "RiskEngine",
    "ServiceContainer",
    "VectorDatabase",
    "attach_infra_routes",
    "bootstrap_infra",
    "configure_json_logging",
    "configure_tracing",
    "gather_bounded",
    "get_circuit_breaker",
    "get_container",
    "get_metrics",
    "get_model_loader",
    "get_registry",
    "map_sync",
    "reset_circuit_breakers",
    "reset_infra_globals",
    "reset_metrics",
    "run_async",
    "run_sync",
    "start_span",
    "tracing_enabled",
]
