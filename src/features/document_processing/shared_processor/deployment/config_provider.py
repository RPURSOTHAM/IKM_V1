"""Single configuration provider for deployment-enabled processors.

Reads static YAML / env today. A future license service can replace the
loader behind ``ProcessorConfigurationProvider`` without changing the pipeline.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from src.features.document_processing.shared_processor.deployment.ids import (
    CANONICAL_DEPLOYMENT_PROCESSOR_IDS,
    DEPLOYMENT_PROCESSOR_ORDER,
    DeploymentProcessorId,
)

logger = logging.getLogger(__name__)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class ProcessorConfigurationError(RuntimeError):
    """Raised when deployment processor configuration is invalid."""


def _default_config_path() -> Path:
    env_path = (os.getenv("PROCESSORS_CONFIG_PATH") or "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    env_dir = (os.getenv("PROCESSORS_CONFIG_DIR") or "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve() / "processors.yaml"
    # repo_root/configs/processors.yaml — this file is src/shared/processor/deployment/
    return Path(__file__).resolve().parents[5] / "configs" / "processors.yaml"


def _parse_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "n", "disable", "disabled"}:
        return False
    raise ProcessorConfigurationError(
        f"Invalid boolean processor flag value: {value!r}. Expected true/false."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        # Minimal fallback for flat `key: true/false` maps under processors:
        data: dict[str, Any] = {}
        processors: dict[str, Any] = {}
        in_processors = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.strip() == "processors:":
                in_processors = True
                continue
            if in_processors and line.startswith(" ") and ":" in line:
                key, _, value = line.strip().partition(":")
                processors[key.strip()] = value.strip()
                continue
            if not line.startswith(" "):
                in_processors = False
        if processors:
            data["processors"] = processors
        return data
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ProcessorConfigurationError(
            f"Processor config at '{path}' must be a YAML mapping, got {type(loaded).__name__}."
        )
    return loaded


def _env_processor_overrides() -> dict[str, bool]:
    """Apply PROCESSOR_<ID>_ENABLED and list-style env overrides."""
    overrides: dict[str, bool] = {}

    enabled_list = (os.getenv("IKM_ENABLED_PROCESSORS") or "").strip()
    disabled_list = (os.getenv("IKM_DISABLED_PROCESSORS") or "").strip()

    if enabled_list:
        selected = {part.strip().lower() for part in enabled_list.split(",") if part.strip()}
        unknown = selected - CANONICAL_DEPLOYMENT_PROCESSOR_IDS
        if unknown:
            raise ProcessorConfigurationError(
                "IKM_ENABLED_PROCESSORS references unknown processor(s): "
                + ", ".join(sorted(unknown))
                + ". Known processors: "
                + ", ".join(pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER)
            )
        for pid in DeploymentProcessorId:
            overrides[pid.value] = pid.value in selected

    if disabled_list:
        selected = {part.strip().lower() for part in disabled_list.split(",") if part.strip()}
        unknown = selected - CANONICAL_DEPLOYMENT_PROCESSOR_IDS
        if unknown:
            raise ProcessorConfigurationError(
                "IKM_DISABLED_PROCESSORS references unknown processor(s): "
                + ", ".join(sorted(unknown))
                + ". Known processors: "
                + ", ".join(pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER)
            )
        for name in selected:
            overrides[name] = False

    for pid in DeploymentProcessorId:
        env_key = f"PROCESSOR_{pid.value.upper()}_ENABLED"
        raw = os.getenv(env_key)
        if raw is None or not str(raw).strip():
            # Also accept hyphen→underscore variants already covered by enum values.
            continue
        overrides[pid.value] = _parse_bool(raw)

    return overrides


class ProcessorConfigurationProvider:
    """Source of truth for which deployment processors are enabled.

    Future licensing can subclass or replace ``load`` without touching the
    registry or processing pipeline:

        enabled = ProcessorConfigurationProvider.get_enabled_processors()
    """

    _lock = threading.RLock()
    _instance: "ProcessorConfigurationProvider | None" = None

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _default_config_path()
        self._flags: dict[str, bool] | None = None
        self._config_present = False

    @property
    def config_path(self) -> Path:
        return self._config_path

    def has_loaded_flags(self) -> bool:
        return self._flags is not None

    @classmethod
    def instance(cls) -> "ProcessorConfigurationProvider":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test helper — clear cached singleton and flags."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def get_enabled_processors(cls) -> list[str]:
        return cls.instance().enabled_processors()

    @classmethod
    def get_disabled_processors(cls) -> list[str]:
        return cls.instance().disabled_processors()

    @classmethod
    def is_enabled(cls, processor_id: str | DeploymentProcessorId) -> bool:
        return cls.instance().is_processor_enabled(processor_id)

    def reload(self, *, keep_previous_on_error: bool = False) -> dict[str, bool]:
        """Reload YAML + env flags.

        When ``keep_previous_on_error`` is True and a prior valid flag set exists,
        invalid configs leave the previous flags in place (and still raise).
        """
        with self._lock:
            previous = dict(self._flags) if self._flags is not None else None
            previous_present = self._config_present
            try:
                loaded = self._load_flags()
                self._flags = loaded
                return loaded
            except ProcessorConfigurationError:
                if keep_previous_on_error and previous is not None:
                    self._flags = previous
                    self._config_present = previous_present
                    logger.error(
                        "Invalid processor configuration; previous valid configuration remains active."
                    )
                else:
                    # Leave flags unset / previous so callers can decide; do not cache bad data.
                    self._flags = previous
                    self._config_present = previous_present
                raise

    def _ensure_flags(self) -> dict[str, bool]:
        if self._flags is not None:
            return self._flags
        self._flags = self._load_flags()
        return self._flags

    def _load_flags(self) -> dict[str, bool]:
        # Default: all enabled (backward compatible when config is absent).
        flags: dict[str, bool] = {pid.value: True for pid in DeploymentProcessorId}
        path = self._config_path
        raw = _load_yaml(path)
        self._config_present = bool(raw) or path.is_file()

        processors_section: Mapping[str, Any] | None = None
        if "processors" in raw:
            section = raw.get("processors")
            if section is None:
                processors_section = {}
            elif isinstance(section, Mapping):
                processors_section = section
            else:
                raise ProcessorConfigurationError(
                    f"'processors' in '{path}' must be a mapping of processor_id → bool."
                )
        elif raw:
            # Allow a flat top-level map of processor flags.
            processors_section = raw

        if processors_section is not None:
            unknown = {
                str(key).strip().lower()
                for key in processors_section.keys()
                if str(key).strip().lower() not in CANONICAL_DEPLOYMENT_PROCESSOR_IDS
            }
            if unknown:
                known = ", ".join(pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER)
                message = (
                    f"Processor configuration references unknown processor(s): "
                    f"{', '.join(sorted(unknown))}. Known processors: {known}."
                )
                logger.error(message)
                raise ProcessorConfigurationError(message)

            for key, value in processors_section.items():
                name = str(key).strip().lower()
                flags[name] = _parse_bool(value)

        # Environment overrides always win over YAML.
        flags.update(_env_processor_overrides())
        return flags

    def is_processor_enabled(self, processor_id: str | DeploymentProcessorId) -> bool:
        name = (
            processor_id.value
            if isinstance(processor_id, DeploymentProcessorId)
            else str(processor_id).strip().lower()
        )
        if name not in CANONICAL_DEPLOYMENT_PROCESSOR_IDS:
            raise ProcessorConfigurationError(
                f"Unknown deployment processor '{processor_id}'. "
                f"Known: {', '.join(pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER)}"
            )
        return bool(self._ensure_flags().get(name, True))

    def enabled_processors(self) -> list[str]:
        flags = self._ensure_flags()
        return [pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER if flags.get(pid.value, True)]

    def disabled_processors(self) -> list[str]:
        flags = self._ensure_flags()
        return [pid.value for pid in DEPLOYMENT_PROCESSOR_ORDER if not flags.get(pid.value, True)]

    def status(self) -> dict[str, Any]:
        enabled = self.enabled_processors()
        disabled = self.disabled_processors()
        return {
            "enabled": enabled,
            "disabled": disabled,
            "config_path": str(self._config_path),
            "config_present": self._config_present or self._config_path.is_file(),
        }
