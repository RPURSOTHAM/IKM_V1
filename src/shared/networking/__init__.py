"""Deployment networking helpers — local vs Docker Compose service hostnames."""

from src.shared.networking.bind_paths import (
    bind_mount_paths_equivalent,
    ensure_host_bind_path_exists,
    host_repo_root,
    normalize_bind_mount_host_path,
    running_inside_container,
)
from src.shared.networking.compose_labels import compose_stack_labels, compose_stack_settings
from src.shared.networking.hosts import (
    DeploymentMode,
    detect_deployment_mode,
    infra_hosts_for_mode,
    resolve_deployment_mode,
)

__all__ = [
    "DeploymentMode",
    "bind_mount_paths_equivalent",
    "compose_stack_labels",
    "compose_stack_settings",
    "detect_deployment_mode",
    "ensure_host_bind_path_exists",
    "host_repo_root",
    "infra_hosts_for_mode",
    "normalize_bind_mount_host_path",
    "resolve_deployment_mode",
    "running_inside_container",
]
