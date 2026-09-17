"""Deployment-level processor configuration for lightweight IKM instances.

Architecture
------------
Deployment Configuration (YAML / env)
        │
        ▼
ProcessorConfigurationProvider.get_enabled_processors()
        │
        ▼
ProcessorRegistry  (instantiate enabled only)
        │
        ▼
DynamicProcessingPipeline
        │
        ▼
Document processing / retrieval gates

Hot reload
----------
``configs/processors.yaml`` changes are detected by a background watcher
(``PROCESSORS_CONFIG_WATCH``, default on) or via
``POST /api/v1/system/processors/reload``. Environment variable overrides
still require an application restart.
"""

from __future__ import annotations

from src.features.document_processing.shared_processor.deployment.bootstrap import (
    bootstrap_deployment_processors,
    format_processor_startup_banner,
    log_processor_startup_banner,
)
from src.features.document_processing.shared_processor.deployment.config_provider import (
    ProcessorConfigurationError,
    ProcessorConfigurationProvider,
)
from src.features.document_processing.shared_processor.deployment.ids import (
    CANONICAL_DEPLOYMENT_PROCESSOR_IDS,
    DEPLOYMENT_PROCESSOR_DISPLAY_NAMES,
    DEPLOYMENT_PROCESSOR_ORDER,
    DEPLOYMENT_TO_PROCESSOR_TYPES,
    EMBEDDING_DEPENDENT_PROCESSOR_TYPES,
    DeploymentProcessorId,
    display_name,
)
from src.features.document_processing.shared_processor.deployment.interface import DeploymentProcessor
from src.features.document_processing.shared_processor.deployment.pipeline import (
    DynamicProcessingPipeline,
    get_active_pipeline,
    rebuild_active_pipeline,
    reset_active_pipeline,
)
from src.features.document_processing.shared_processor.deployment.registry import (
    ProcessorRegistry,
    get_processor_registry,
    iter_enabled_processors,
)
from src.features.document_processing.shared_processor.deployment.reload import (
    ProcessorReloadResult,
    reload_deployment_processors,
    reload_deployment_processors_or_raise,
)
from src.features.document_processing.shared_processor.deployment.watcher import (
    start_processor_config_watcher,
    stop_processor_config_watcher,
)

__all__ = [
    "CANONICAL_DEPLOYMENT_PROCESSOR_IDS",
    "DEPLOYMENT_PROCESSOR_DISPLAY_NAMES",
    "DEPLOYMENT_PROCESSOR_ORDER",
    "DEPLOYMENT_TO_PROCESSOR_TYPES",
    "EMBEDDING_DEPENDENT_PROCESSOR_TYPES",
    "DeploymentProcessor",
    "DeploymentProcessorId",
    "DynamicProcessingPipeline",
    "ProcessorConfigurationError",
    "ProcessorConfigurationProvider",
    "ProcessorRegistry",
    "ProcessorReloadResult",
    "bootstrap_deployment_processors",
    "display_name",
    "format_processor_startup_banner",
    "get_active_pipeline",
    "get_processor_registry",
    "iter_enabled_processors",
    "log_processor_startup_banner",
    "rebuild_active_pipeline",
    "reload_deployment_processors",
    "reload_deployment_processors_or_raise",
    "reset_active_pipeline",
    "start_processor_config_watcher",
    "stop_processor_config_watcher",
    "is_deployment_processor_enabled",
]


def is_deployment_processor_enabled(processor_id: str | DeploymentProcessorId) -> bool:
    """Convenience gate used throughout ingest/retrieval call sites."""
    return ProcessorConfigurationProvider.is_enabled(processor_id)
