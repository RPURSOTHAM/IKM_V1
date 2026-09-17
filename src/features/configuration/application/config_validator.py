from __future__ import annotations

from typing import Any

from src.features.configuration.domain.config_exceptions import ValidationError
from src.features.configuration.application.schema_registry import ConfigKeySpec, get_spec


def coerce_value(spec: ConfigKeySpec, raw: Any) -> Any:
    if raw is None:
        if spec.default is None and spec.value_type != "secret_ref":
            raise ValidationError(f"Value for '{spec.namespace}.{spec.key}' cannot be null.")
        return spec.default

    value_type = spec.value_type
    try:
        if value_type == "string":
            text = str(raw).strip()
            if not text and spec.default is None:
                return None
            return text
        if value_type == "integer":
            return int(raw)
        if value_type == "float":
            return float(raw)
        if value_type == "boolean":
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            return text in {"1", "true", "yes", "on"}
        if value_type == "json":
            if isinstance(raw, (dict, list)):
                return raw
            raise ValidationError(f"'{spec.key}' must be a JSON object or array.")
        if value_type == "secret_ref":
            text = str(raw).strip()
            if not text:
                return None
            return text
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Invalid value for '{spec.namespace}.{spec.key}': expected {value_type}.",
            details={"error": str(exc)},
        ) from exc

    raise ValidationError(f"Unsupported value_type '{value_type}'.")


def validate_update(namespace: str, key: str, value: Any) -> tuple[ConfigKeySpec, Any]:
    spec = get_spec(namespace, key)
    if spec is None:
        raise ValidationError(
            f"Unknown configuration key '{namespace}.{key}'.",
            details={"namespace": namespace, "key": key},
        )
    coerced = coerce_value(spec, value)
    _validate_ranges(spec, coerced)
    return spec, coerced


def _validate_ranges(spec: ConfigKeySpec, value: Any) -> None:
    if value is None:
        return
    key = spec.key
    if spec.value_type == "integer":
        if key == "chunking.chunk_size" and not (200 <= int(value) <= 2000):
            raise ValidationError("processor.chunking.chunk_size must be between 200 and 2000.")
        if key == "validation.max_upload_bytes" and int(value) < 1:
            raise ValidationError("validation.max_upload_bytes must be positive.")
        if key == "validation.max_files_per_request" and int(value) < 1:
            raise ValidationError("validation.max_files_per_request must be at least 1.")
        if key == "search.max_top_k" and int(value) < 1:
            raise ValidationError("search.max_top_k must be at least 1.")
        if key == "hierarchy.max_depth" and int(value) < 1:
            raise ValidationError("hierarchy.max_depth must be at least 1.")
    if spec.value_type == "float":
        if key == "search.hybrid_alpha" and not (0.0 <= float(value) <= 1.0):
            raise ValidationError("search.hybrid_alpha must be between 0.0 and 1.0.")
