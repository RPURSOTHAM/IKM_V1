"""Bootstrap default plugins from existing production implementations."""

from __future__ import annotations

import logging
import os

from src.infrastructure.application_support.registry import (
    ConfigDrivenModelLoader,
    get_container,
    get_model_loader,
    get_registry,
    is_bootstrapped,
    mark_bootstrapped,
)

_logger = logging.getLogger(__name__)


def bootstrap_infra(*, service: str | None = None, force: bool = False) -> ConfigDrivenModelLoader:
    """Register default adapters once. Safe to call from multiple services."""
    if is_bootstrapped() and not force:
        return get_model_loader()

    from src.infrastructure.application_support.adapters import (
        ComplianceDLPAdapter,
        DLPRiskEngineAdapter,
        DocumentLoaderParserAdapter,
        GeneratorLLMAdapter,
        NERDetectorAdapter,
        OutputModerationAdapter,
        RetrievalServiceAdapter,
        SentenceTransformerEmbedderAdapter,
        StrategyChunkerAdapter,
        WeaviateVectorDatabaseAdapter,
    )
    from src.infrastructure.application_support.logging_json import configure_json_logging
    from src.infrastructure.application_support.metrics import get_metrics
    from src.infrastructure.application_support.tracing import configure_tracing

    service_name = service or os.getenv("SERVICE_NAME") or "rag-builder"
    configure_json_logging(service=service_name)
    configure_tracing(service_name)
    get_metrics().up.labels(service=service_name).set(1)

    registry = get_registry()
    registry.register("parser", "document_loader", DocumentLoaderParserAdapter, default=True)
    registry.register("chunker", "strategy_chunker", StrategyChunkerAdapter, default=True)
    registry.register(
        "embedder",
        "sentence_transformers",
        lambda: SentenceTransformerEmbedderAdapter(os.getenv("EMBEDDING_MODEL") or os.getenv("MODEL_DIR")),
        default=True,
        metadata={"preferred_model": "BAAI/bge-m3", "note": "Does not replace production model unless configured"},
    )
    registry.register("retriever", "retrieval_service", RetrievalServiceAdapter, default=True)
    registry.register("vector_database", "weaviate", WeaviateVectorDatabaseAdapter, default=True)
    registry.register("llm", "generator", GeneratorLLMAdapter, default=True)
    registry.register("ner", "ner_detector", NERDetectorAdapter, default=True)
    registry.register("dlp", "compliance_scanner", ComplianceDLPAdapter, default=True)
    registry.register("moderation", "output_moderation", OutputModerationAdapter, default=True)
    registry.register("risk_engine", "dlp_policy_engine", DLPRiskEngineAdapter, default=True)

    container = get_container()
    loader = get_model_loader()
    container.register_instance("plugin_registry", registry)
    container.register_instance("model_loader", loader)
    container.register_factory("metrics", get_metrics)

    mark_bootstrapped()
    _logger.info(
        "Enterprise infra bootstrapped",
        extra={"event": "infra_bootstrap", "component": "infra", "service": service_name},
    )
    return loader
