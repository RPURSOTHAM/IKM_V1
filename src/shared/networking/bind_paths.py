"""Normalize host paths used for Docker bind mounts."""

from __future__ import annotations

import os
import re
from pathlib import Path


_MALFORMED_CONTAINER_WINDOWS_PREFIX = re.compile(r"^/app/([A-Za-z]:/.*)$")


def running_inside_container() -> bool:
    return os.path.exists("/.dockerenv") or os.getenv("RUNNING_IN_DOCKER", "").lower() in {
        "1",
        "true",
        "yes",
    }


def is_windows_host_path(path: str) -> bool:
    return len(path) >= 2 and path[1] == ":" and path[0].isalpha()


def host_repo_root() -> str | None:
    for key in ("HOST_REPO_ROOT", "REPO_ROOT"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value.replace("\\", "/").rstrip("/")
    return None


def repair_malformed_bind_mount_path(path: str) -> str:
    """Undo ``/app/C:/...`` paths produced by resolving Windows paths inside a container."""
    normalized = path.replace("\\", "/")
    match = _MALFORMED_CONTAINER_WINDOWS_PREFIX.match(normalized)
    if match:
        return match.group(1).rstrip("/")
    return normalized.rstrip("/")


def normalize_bind_mount_host_path(path: str | None, *, repo_root: str | None = None) -> str | None:
    """Return a host path suitable for Docker bind mounts.

    When the scheduler runs inside a container with a mounted Docker socket, host
    paths must be passed through to the daemon unchanged. Resolving them with
    ``Path.resolve()`` would anchor Windows drive paths under the container cwd
    (e.g. ``/app/C:/Users/...``).
    """
    if not path:
        return None
    text = repair_malformed_bind_mount_path(str(path).strip())
    if not text:
        return None

    if is_windows_host_path(text):
        return text

    if running_inside_container():
        root = (repo_root or host_repo_root() or "").strip().replace("\\", "/").rstrip("/")
        if root and not text.startswith("/"):
            return f"{root}/{text.lstrip('/')}"
        return text

    return str(Path(text).expanduser().resolve())


def bind_mount_paths_equivalent(left: str | None, right: str | None) -> bool:
    left_norm = normalize_bind_mount_host_path(left)
    right_norm = normalize_bind_mount_host_path(right)
    if not left_norm or not right_norm:
        return not left_norm and not right_norm
    return left_norm.casefold() == right_norm.casefold()


def ensure_host_bind_path_exists(path: str | None) -> None:
    """Create a directory when the scheduler runs on the host filesystem."""
    if not path or running_inside_container():
        return
    Path(path).expanduser().mkdir(parents=True, exist_ok=True)
