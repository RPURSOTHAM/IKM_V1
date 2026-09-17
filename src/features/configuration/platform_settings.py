from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSUMER_API_ROOT = REPO_ROOT / "src" / "dms_service" / "consumer_api_service"

# Single stack env (Docker + local).
load_dotenv(REPO_ROOT / "deploy" / "application" / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(REPO_ROOT / "src" / "processor_service" / ".env", override=False)
load_dotenv(CONSUMER_API_ROOT / ".env", override=True)

from src.shared.networking.hosts import detect_deployment_mode, infra_hosts_for_mode


def _resolve(key: str, default: Any = None) -> Any:
    value = os.environ.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, str):
        from src.features.configuration.application.schema_registry import strip_wrapping_quotes

        return strip_wrapping_quotes(value)
    return value


def _env_bool(key: str, default: str = "false") -> bool:
    raw = str(_resolve(key, default)).strip().lower()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1].strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _collect_api_keys(*env_names: str) -> tuple[str, ...]:
    keys: list[str] = []
    for name in env_names:
        raw = _resolve(name)
        if raw is None:
            continue
        keys.extend(k.strip() for k in str(raw).split(",") if k.strip())
    return tuple(dict.fromkeys(keys))


_API_KEYS = _collect_api_keys("CONSUMER_API_KEYS", "RAG_API_KEYS", "CONSUMER_API_KEY", "RAG_API_KEY")
_ADMIN_API_KEYS = _collect_api_keys("CONSUMER_API_ADMIN_KEYS", "RAG_API_ADMIN_KEYS")


@dataclass(frozen=True)
class ApiSettings:
    host: str
    port: int
    upload_dir: str
    db_path: str
    repository_type: str
    scheduler_host: str
    scheduler_port: int
    scheduler_timeout_seconds: float
    auto_start_scheduler: bool
    scheduler_dir: str
    rabbitmq_host: str
    rabbitmq_port: int
    rabbitmq_user: str
    rabbitmq_pass: str
    rabbitmq_queue_name: str
    rabbitmq_management_url: str
    rabbitmq_management_user: str
    rabbitmq_management_pass: str
    rabbitmq_management_timeout: float
    default_collection_name: str
    default_tenant_id: str | None
    enable_retrieval: bool
    enable_repository_admin: bool
    enable_queue_admin: bool
    app_env: str
    weaviate_url: str
    weaviate_grpc_port: int
    weaviate_api_key: str | None
    retrieval_model_name: str
    retrieval_model_dir: str | None
    hybrid_alpha: float
    cors_allow_origins: str
    api_version: str
    api_prefix: str
    api_keys: tuple[str, ...]
    admin_api_keys: tuple[str, ...]
    api_require_key: bool
    expose_openapi: bool
    enable_hsts: bool
    allowed_extensions: tuple[str, ...]
    max_upload_bytes: int
    max_files_per_request: int


_settings_cache: ApiSettings | None = None
_settings_version: int = -1


def _bootstrap_api_settings() -> ApiSettings:
    """Cold-start settings from environment only (before MySQL provider is available)."""
    mode = detect_deployment_mode()
    hosts = infra_hosts_for_mode(mode)
    upload_dir = _resolve_path(str(_resolve("DOCUMENT_ROOT", "_documents")))
    rabbitmq_user = str(_resolve("RABBITMQ_USER", "rabbitmq_user"))
    return ApiSettings(
        host=str(_resolve("RAG_API_HOST", "0.0.0.0")),
        port=int(_resolve("RAG_API_PORT", "8088")),
        upload_dir=upload_dir,
        db_path=_resolve_path(str(_resolve("RAG_API_DB_PATH", "src/dms_service/consumer_api_service/data/rag_api.db"))),
        repository_type=str(_resolve("DOCUMENT_REPOSITORY_TYPE", "local")),
        scheduler_host=str(_resolve("SCHEDULER_HOST") or hosts["scheduler"]),
        scheduler_port=int(_resolve("SCHEDULER_TCP_PORT", "3200")),
        scheduler_timeout_seconds=float(_resolve("SCHEDULER_TIMEOUT_SECONDS", "30")),
        auto_start_scheduler=_env_bool("RAG_API_AUTO_START_SCHEDULER", "false"),
        scheduler_dir=_resolve_path(str(_resolve("SCHEDULER_DIR", "src/scheduler_service"))),
        rabbitmq_host=str(_resolve("RABBITMQ_HOST") or hosts["rabbitmq"]),
        rabbitmq_port=int(_resolve("RABBITMQ_PORT", "5672")),
        rabbitmq_user=rabbitmq_user,
        rabbitmq_pass=str(_resolve("RABBITMQ_PASS", "rabbitmq_password")),
        rabbitmq_queue_name=str(_resolve("RABBITMQ_QUEUE_NAME", "document_processing_queue")),
        rabbitmq_management_url=str(_resolve("RABBITMQ_MANAGEMENT_URL") or hosts["rabbitmq_management_url"]),
        rabbitmq_management_user=str(_resolve("RABBITMQ_MANAGEMENT_USER", rabbitmq_user)),
        rabbitmq_management_pass=str(_resolve("RABBITMQ_MANAGEMENT_PASS", _resolve("RABBITMQ_PASS", "rabbitmq_password"))),
        rabbitmq_management_timeout=float(_resolve("RABBITMQ_MANAGEMENT_TIMEOUT", "30")),
        default_collection_name=str(_resolve("WEAVIATE_COLLECTION", "DocumentChunk")),
        default_tenant_id=_resolve("DEFAULT_TENANT_ID"),
        enable_retrieval=_env_bool("RAG_API_ENABLE_RETRIEVAL", "false"),
        enable_repository_admin=_env_bool("RAG_API_ENABLE_REPOSITORY_ADMIN", "true"),
        enable_queue_admin=_env_bool("RAG_API_ENABLE_QUEUE_ADMIN", "true"),
        app_env=str(_resolve("RAG_API_ENV", _resolve("APP_ENV", "development"))).strip().lower(),
        weaviate_url=_resolve_weaviate_url_env(),
        weaviate_grpc_port=int(_resolve("WEAVIATE_GRPC_PORT", "50051")),
        weaviate_api_key=_resolve_weaviate_api_key_env(),
        retrieval_model_name=str(
            _resolve("EMBEDDING_MODEL", _resolve("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
        ),
        retrieval_model_dir=_resolve("MODEL_DIR"),
        hybrid_alpha=float(_resolve("HYBRID_ALPHA", "0.75")),
        cors_allow_origins=str(_resolve("CORS_ALLOW_ORIGINS", "*")),
        api_version=str(_resolve("CONSUMER_API_VERSION", "v1")),
        api_prefix=str(_resolve("CONSUMER_API_PREFIX", "/api/v1")),
        api_keys=_API_KEYS,
        admin_api_keys=_ADMIN_API_KEYS,
        api_require_key=_resolve_require_api_key_env(_API_KEYS),
        expose_openapi=_env_bool(
            "CONSUMER_API_EXPOSE_DOCS",
            "true"
            if str(_resolve("RAG_API_ENV", _resolve("APP_ENV", "development"))).strip().lower()
            in {"development", "dev", "local"}
            else "false",
        ),
        enable_hsts=_env_bool("CONSUMER_API_ENABLE_HSTS", "false"),
        allowed_extensions=(".docx", ".pdf", ".txt", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".mkv", ".webm"),
        max_upload_bytes=209_715_200,
        max_files_per_request=50,
    )


def _resolve_weaviate_api_key_env() -> str | None:
    direct = _resolve("WEAVIATE_API_KEY")
    if direct is not None:
        s = str(direct).strip()
        if s:
            return s
    allowed = _resolve("WEAVIATE_AUTH_APIKEY_ALLOWED_KEYS")
    if allowed is None:
        return None
    first = str(allowed).split(",")[0].strip()
    return first or None


def _resolve_weaviate_url_env() -> str:
    url = _resolve("WEAVIATE_URL")
    if url is not None:
        s = str(url).strip()
        if s:
            return s
    hosts = infra_hosts_for_mode(detect_deployment_mode())
    if _resolve("WEAVIATE_HOST") is None:
        return hosts["weaviate_url"]
    host = str(_resolve("WEAVIATE_HOST", "localhost"))
    port = int(_resolve("WEAVIATE_PORT", "8086"))
    secure = _env_bool("WEAVIATE_HTTP_SECURE", "false")
    scheme = "https" if secure else "http"
    return f"{scheme}://{host}:{port}"


def _resolve_require_api_key_env(api_keys: tuple[str, ...]) -> bool:
    for key in ("CONSUMER_API_REQUIRE_API_KEY", "RAG_API_REQUIRE_API_KEY"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return _env_bool(key, str(raw))
    return bool(api_keys)


def build_api_settings() -> ApiSettings:
    """Build effective API settings from hot platform config with env/bootstrap fallbacks."""
    try:
        from src.features.configuration.application.config_provider import get_platform_config

        provider = get_platform_config()
        version = provider.config_version
        global_cfg = provider.namespace("global")
        dms_cfg = provider.namespace("dms")
        receiver_cfg = provider.namespace("dms.receiver")
        retrieval_cfg = provider.namespace("dms.retrieval")

        document_root = global_cfg.get_str("document_root", "_documents") or "_documents"
        upload_dir_raw = receiver_cfg.get_str("storage.upload_dir", document_root) or document_root
        extensions_raw = receiver_cfg.get(
            "validation.allowed_extensions",
            [
                ".docx",
                ".pdf",
                ".txt",
                ".pptx",
                ".ppt",
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".wav",
                ".mp3",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".mp4",
                ".mov",
                ".avi",
                ".mkv",
                ".webm",
            ],
        )
        if isinstance(extensions_raw, list):
            allowed = tuple(str(x) for x in extensions_raw)
        else:
            allowed = (
                ".docx",
                ".pdf",
                ".txt",
                ".pptx",
                ".ppt",
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".wav",
                ".mp3",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".mp4",
                ".mov",
                ".avi",
                ".mkv",
                ".webm",
            )

        cold = _bootstrap_api_settings()
        use_local_infra = detect_deployment_mode() == "local"
        if use_local_infra:
            rabbitmq_host = cold.rabbitmq_host
            rabbitmq_port = cold.rabbitmq_port
            rabbitmq_user = cold.rabbitmq_user
            rabbitmq_pass = cold.rabbitmq_pass
            rabbitmq_queue_name = cold.rabbitmq_queue_name
            weaviate_url = cold.weaviate_url
            weaviate_grpc_port = cold.weaviate_grpc_port
            weaviate_api_key = cold.weaviate_api_key
        else:
            rabbitmq_host = global_cfg.get_str("rabbitmq.host", cold.rabbitmq_host) or cold.rabbitmq_host
            rabbitmq_port = global_cfg.get_int("rabbitmq.port", cold.rabbitmq_port)
            rabbitmq_user = global_cfg.get_str("rabbitmq.user", cold.rabbitmq_user) or cold.rabbitmq_user
            rabbitmq_pass = global_cfg.get_str("rabbitmq.password", cold.rabbitmq_pass) or cold.rabbitmq_pass or ""
            rabbitmq_queue_name = (
                global_cfg.get_str("rabbitmq.queue_name", cold.rabbitmq_queue_name) or cold.rabbitmq_queue_name
            )
            weaviate_url = global_cfg.get_str("weaviate.url", cold.weaviate_url) or cold.weaviate_url
            weaviate_grpc_port = global_cfg.get_int("weaviate.grpc_port", cold.weaviate_grpc_port)
            weaviate_api_key = global_cfg.get_str("weaviate.api_key", cold.weaviate_api_key)
        return ApiSettings(
            host=dms_cfg.get_str("api.host", cold.host) or cold.host,
            port=int(dms_cfg.get_int("api.port", cold.port)),
            upload_dir=_resolve_path(upload_dir_raw),
            db_path=cold.db_path,
            repository_type=receiver_cfg.get_str("storage.repository_type", cold.repository_type) or cold.repository_type,
            scheduler_host=cold.scheduler_host,
            scheduler_port=cold.scheduler_port,
            scheduler_timeout_seconds=float(
                provider.namespace("scheduler").get_float("control.timeout_seconds", cold.scheduler_timeout_seconds)
            ),
            auto_start_scheduler=dms_cfg.get_bool("features.auto_start_scheduler", cold.auto_start_scheduler),
            scheduler_dir=cold.scheduler_dir,
            rabbitmq_host=rabbitmq_host,
            rabbitmq_port=rabbitmq_port,
            rabbitmq_user=rabbitmq_user,
            rabbitmq_pass=rabbitmq_pass,
            rabbitmq_queue_name=rabbitmq_queue_name,
            rabbitmq_management_url=cold.rabbitmq_management_url,
            rabbitmq_management_user=cold.rabbitmq_management_user,
            rabbitmq_management_pass=cold.rabbitmq_management_pass,
            rabbitmq_management_timeout=cold.rabbitmq_management_timeout,
            default_collection_name=global_cfg.get_str("weaviate.default_collection", cold.default_collection_name)
            or cold.default_collection_name,
            default_tenant_id=global_cfg.get_str("weaviate.default_tenant_id", cold.default_tenant_id),
            enable_retrieval=dms_cfg.get_bool("features.enable_retrieval", cold.enable_retrieval)
            or retrieval_cfg.get_bool("enabled", cold.enable_retrieval),
            enable_repository_admin=dms_cfg.get_bool("features.enable_repository_admin", cold.enable_repository_admin),
            enable_queue_admin=dms_cfg.get_bool("features.enable_queue_admin", cold.enable_queue_admin),
            app_env=global_cfg.get_str("app_env", cold.app_env) or cold.app_env,
            weaviate_url=weaviate_url,
            weaviate_grpc_port=weaviate_grpc_port,
            weaviate_api_key=weaviate_api_key,
            retrieval_model_name=retrieval_cfg.get_str("embedding.model_name", cold.retrieval_model_name)
            or cold.retrieval_model_name,
            retrieval_model_dir=retrieval_cfg.get_str("embedding.model_dir", cold.retrieval_model_dir),
            hybrid_alpha=retrieval_cfg.get_float("search.hybrid_alpha", cold.hybrid_alpha),
            cors_allow_origins=dms_cfg.get_str("api.cors_allow_origins", cold.cors_allow_origins) or cold.cors_allow_origins,
            api_version=cold.api_version,
            api_prefix=dms_cfg.get_str("api.prefix", cold.api_prefix) or cold.api_prefix,
            api_keys=cold.api_keys,
            admin_api_keys=cold.admin_api_keys,
            api_require_key=dms_cfg.get_bool("api.require_key", cold.api_require_key),
            expose_openapi=dms_cfg.get_bool("api.expose_openapi", cold.expose_openapi),
            enable_hsts=dms_cfg.get_bool("api.enable_hsts", cold.enable_hsts),
            allowed_extensions=allowed,
            max_upload_bytes=receiver_cfg.get_int("validation.max_upload_bytes", cold.max_upload_bytes),
            max_files_per_request=receiver_cfg.get_int("validation.max_files_per_request", cold.max_files_per_request),
        )
    except Exception:
        return _bootstrap_api_settings()


def get_settings() -> ApiSettings:
    global _settings_cache, _settings_version
    try:
        from src.features.configuration.application.config_provider import get_platform_config

        version = get_platform_config().config_version
    except Exception:
        version = -1
    if _settings_cache is None or _settings_version != version:
        _settings_cache = build_api_settings()
        _settings_version = version
    return _settings_cache


class _SettingsProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)


settings = _SettingsProxy()
