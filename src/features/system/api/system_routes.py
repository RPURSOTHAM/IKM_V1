"""System-level HTTP routes (deployment status, processor catalog)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from src.features.document_processing.shared_processor.deployment import ProcessorConfigurationProvider
from src.features.document_processing.shared_processor.deployment.pipeline import get_active_pipeline
from src.features.document_processing.shared_processor.deployment.reload import reload_deployment_processors

router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/processors",
    summary="List deployment-enabled and disabled processors",
    operation_id="listDeploymentProcessors",
)
def list_deployment_processors() -> dict[str, Any]:
    """Return processors enabled / disabled for this IKM deployment."""
    provider = ProcessorConfigurationProvider.instance()
    status = provider.status()
    pipeline = [p.id for p in get_active_pipeline().processors]
    return {
        "enabled": status["enabled"],
        "disabled": status["disabled"],
        "pipeline": pipeline,
    }


@router.post(
    "/processors/reload",
    summary="Reload deployment processor configuration from YAML",
    operation_id="reloadDeploymentProcessors",
)
def reload_deployment_processors_endpoint() -> dict[str, Any]:
    """Reload ``processors.yaml``, rebuild registry + pipeline, return active lists.

    Invalid configuration returns HTTP 400 and leaves the previous valid
    configuration active when one exists.
    """
    result = reload_deployment_processors(
        initialize=False,  # DMS does not preload heavy models
        keep_previous_on_error=True,
        reason="api",
    )
    if not result.ok:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_processor_configuration",
                "message": result.error or "Processor configuration reload failed.",
                "kept_previous_config": result.kept_previous_config,
                "enabled": result.enabled,
                "disabled": result.disabled,
                "pipeline": result.pipeline,
            },
        )
    return {
        "ok": True,
        "enabled": result.enabled,
        "disabled": result.disabled,
        "pipeline": result.pipeline,
        "newly_enabled": result.newly_enabled,
        "newly_disabled": result.newly_disabled,
        "config_path": result.config_path,
    }
