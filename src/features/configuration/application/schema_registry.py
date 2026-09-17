from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

ReloadPolicy = str  # hot | warm | cold
ValueType = str  # string | integer | float | boolean | json | secret_ref


def strip_wrapping_quotes(value: str) -> str:
    """Remove a single pair of surrounding quotes from dotenv / legacy seed values."""
    s = value.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        return s[1:-1]
    return s


def normalize_string_config_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return strip_wrapping_quotes(value)


@dataclass(frozen=True)
class ConfigKeySpec:
    namespace: str
    key: str
    value_type: ValueType
    default: Any
    reload_policy: ReloadPolicy
    description: str
    env_aliases: tuple[str, ...] = ()
    is_sensitive: bool = False
    env_resolver: Callable[[], Any] | None = None


def _env_bool(*keys: str, default: str = "false") -> bool:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            value = str(raw).strip().lower()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1].strip().lower()
            return value in {"1", "true", "yes", "on"}
    return default.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(*keys: str, default: Any = None) -> Any:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return strip_wrapping_quotes(str(raw))
    return default


def _env_int(*keys: str, default: int) -> int:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return int(raw)
    return default


def _env_float(*keys: str, default: float) -> float:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return float(raw)
    return default


def _resolve_weaviate_url() -> str:
    url = _env_str("WEAVIATE_URL")
    if url:
        return url
    host = str(_env_str("WEAVIATE_HOST", default="localhost"))
    port = _env_int("WEAVIATE_PORT", default=8086)
    secure = _env_bool("WEAVIATE_HTTP_SECURE", default="false")
    scheme = "https" if secure else "http"
    return f"{scheme}://{host}:{port}"


def _resolve_weaviate_api_key() -> str | None:
    direct = _env_str("WEAVIATE_API_KEY")
    if direct:
        return direct
    allowed = _env_str("WEAVIATE_AUTH_APIKEY_ALLOWED_KEYS")
    if allowed:
        first = str(allowed).split(",")[0].strip()
        return first or None
    return None


def _resolve_api_require_key() -> bool:
    for key in ("CONSUMER_API_REQUIRE_API_KEY", "RAG_API_REQUIRE_API_KEY"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return _env_bool(key, default=str(raw))
    keys_raw = _env_str("CONSUMER_API_KEYS", "RAG_API_KEYS", "CONSUMER_API_KEY", "RAG_API_KEY")
    return bool(keys_raw)


def _resolve_expose_openapi() -> bool:
    explicit = os.environ.get("CONSUMER_API_EXPOSE_DOCS")
    if explicit is not None and str(explicit).strip() != "":
        return _env_bool("CONSUMER_API_EXPOSE_DOCS", default=str(explicit))
    app_env = str(_env_str("RAG_API_ENV", "APP_ENV", default="development")).strip().lower()
    return app_env in {"development", "dev", "local"}


def _resolve_postgres_jobs_enabled() -> bool:
    return bool(
        _env_str(
            "DOCUMENT_JOBS_POSTGRES_HOST",
            "POSTGRES_HOST",
            "DOCUMENT_JOBS_MYSQL_HOST",
        )
    )

CONFIG_REGISTRY: list[ConfigKeySpec] = [
    # global
    ConfigKeySpec("global", "weaviate.url", "string", "http://localhost:8086", "hot", "Weaviate HTTP endpoint", ("WEAVIATE_URL",), env_resolver=_resolve_weaviate_url),
    ConfigKeySpec("global", "weaviate.grpc_port", "integer", 50051, "hot", "Weaviate gRPC port", ("WEAVIATE_GRPC_PORT",)),
    ConfigKeySpec("global", "weaviate.api_key", "secret_ref", None, "hot", "Weaviate client API key", ("WEAVIATE_API_KEY",), is_sensitive=True, env_resolver=_resolve_weaviate_api_key),
    ConfigKeySpec("global", "weaviate.default_collection", "string", "DocumentChunk", "hot", "Fallback Weaviate collection", ("WEAVIATE_COLLECTION",)),
    ConfigKeySpec("global", "weaviate.default_tenant_id", "string", None, "hot", "Optional default tenant", ("DEFAULT_TENANT_ID",)),
    ConfigKeySpec("global", "rabbitmq.host", "string", "localhost", "hot", "RabbitMQ host", ("RABBITMQ_HOST",)),
    ConfigKeySpec("global", "rabbitmq.port", "integer", 5672, "hot", "RabbitMQ AMQP port", ("RABBITMQ_PORT",)),
    ConfigKeySpec("global", "rabbitmq.user", "string", "rabbitmq_user", "hot", "RabbitMQ username", ("RABBITMQ_USER",)),
    ConfigKeySpec("global", "rabbitmq.password", "secret_ref", None, "hot", "RabbitMQ password", ("RABBITMQ_PASS",), is_sensitive=True),
    ConfigKeySpec("global", "rabbitmq.queue_name", "string", "document_processing_queue", "warm", "Primary processing queue", ("RABBITMQ_QUEUE_NAME",)),
    ConfigKeySpec("global", "postgres.document_jobs.enabled", "boolean", True, "cold", "PostgreSQL document job store enabled", env_resolver=_resolve_postgres_jobs_enabled),
    ConfigKeySpec("global", "mysql.document_jobs.enabled", "boolean", True, "cold", "Deprecated alias — use postgres.document_jobs.enabled", env_resolver=_resolve_postgres_jobs_enabled),
    ConfigKeySpec("global", "document_root", "string", "_documents", "warm", "Host path for uploaded originals", ("DOCUMENT_ROOT",)),
    ConfigKeySpec("global", "app_env", "string", "development", "hot", "Environment label", ("RAG_API_ENV", "APP_ENV")),
    # dms
    ConfigKeySpec("dms", "api.host", "string", "0.0.0.0", "cold", "HTTP bind host", ("RAG_API_HOST",)),
    ConfigKeySpec("dms", "api.port", "integer", 8088, "cold", "HTTP bind port", ("RAG_API_PORT",)),
    ConfigKeySpec("dms", "api.prefix", "string", "/api/v1", "hot", "API mount prefix", ("CONSUMER_API_PREFIX",)),
    ConfigKeySpec("dms", "api.require_key", "boolean", False, "hot", "Enforce consumer API key", env_resolver=_resolve_api_require_key),
    ConfigKeySpec("dms", "api.expose_openapi", "boolean", True, "hot", "Expose OpenAPI docs", env_resolver=_resolve_expose_openapi),
    ConfigKeySpec("dms", "api.cors_allow_origins", "string", "*", "hot", "CORS allow origins", ("CORS_ALLOW_ORIGINS",)),
    ConfigKeySpec("dms", "api.enable_hsts", "boolean", False, "hot", "Enable HSTS header", ("CONSUMER_API_ENABLE_HSTS",)),
    ConfigKeySpec("dms", "features.enable_retrieval", "boolean", False, "hot", "Master retrieval switch", ("RAG_API_ENABLE_RETRIEVAL",)),
    ConfigKeySpec("dms", "features.enable_repository_admin", "boolean", True, "hot", "Weaviate admin routes", ("RAG_API_ENABLE_REPOSITORY_ADMIN",)),
    ConfigKeySpec("dms", "features.enable_queue_admin", "boolean", True, "hot", "Queue admin routes", ("RAG_API_ENABLE_QUEUE_ADMIN",)),
    ConfigKeySpec("dms", "features.auto_start_scheduler", "boolean", False, "warm", "Auto-start scheduler from API", ("RAG_API_AUTO_START_SCHEDULER",)),
    # dms.receiver
    ConfigKeySpec("dms.receiver", "storage.repository_type", "string", "local", "warm", "Document storage backend type", ("DOCUMENT_REPOSITORY_TYPE",)),
    ConfigKeySpec("dms.receiver", "storage.upload_dir", "string", "_documents", "warm", "Upload directory", ("DOCUMENT_ROOT",)),
    ConfigKeySpec(
        "dms.receiver",
        "validation.allowed_extensions",
        "json",
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
        "hot",
        "Allowed upload extensions",
        ("DOCUMENT_ALLOWED_EXTENSIONS",),
    ),
    ConfigKeySpec("dms.receiver", "validation.max_upload_bytes", "integer", 209_715_200, "hot", "Max upload size in bytes"),
    ConfigKeySpec("dms.receiver", "validation.max_files_per_request", "integer", 50, "hot", "Max files per multipart upload request"),
    # dms.retrieval
    ConfigKeySpec("dms.retrieval", "enabled", "boolean", False, "hot", "Retrieval namespace enable flag", ("RAG_API_ENABLE_RETRIEVAL",)),
    ConfigKeySpec("dms.retrieval", "embedding.model_name", "string", "BAAI/bge-base-en", "warm", "Default query embedding model", ("EMBEDDING_MODEL", "MODEL_NAME")),
    ConfigKeySpec("dms.retrieval", "embedding.model_dir", "string", None, "warm", "Local model directory", ("MODEL_DIR",)),
    ConfigKeySpec("dms.retrieval", "search.hybrid_alpha", "float", 0.75, "hot", "Hybrid search alpha", ("HYBRID_ALPHA",)),
    ConfigKeySpec("dms.retrieval", "search.default_top_k", "integer", 10, "hot", "Default top_k"),
    ConfigKeySpec("dms.retrieval", "search.max_top_k", "integer", 50, "hot", "Maximum top_k"),
    ConfigKeySpec("dms.retrieval", "search.max_query_length", "integer", 2048, "hot", "Maximum query length"),
    ConfigKeySpec("dms.retrieval", "pipeline.enable_hybrid_rrf", "boolean", True, "hot", "Enable dense+BM25 RRF production pipeline", ("RETRIEVAL_ENABLE_HYBRID_RRF",)),
    ConfigKeySpec("dms.retrieval", "pipeline.candidate_k", "integer", 40, "hot", "Candidate pool before rerank", ("RETRIEVAL_CANDIDATE_K",)),
    ConfigKeySpec("dms.retrieval", "pipeline.rerank_top_k", "integer", 20, "hot", "Top-K candidates sent to reranker", ("RETRIEVAL_RERANK_TOP_K",)),
    ConfigKeySpec("dms.retrieval", "pipeline.rrf_k", "integer", 60, "hot", "RRF constant k", ("RETRIEVAL_RRF_K",)),
    ConfigKeySpec("dms.retrieval", "pipeline.reranker_model", "string", "BAAI/bge-reranker-v2-m3", "warm", "Preferred cross-encoder reranker", ("RETRIEVAL_RERANKER_MODEL", "RERANKER_MODEL")),
    ConfigKeySpec("dms.retrieval", "pipeline.prompt_guard_model", "string", "meta-llama/Llama-Guard-3-8B", "warm", "Preferred prompt guard model", ("RETRIEVAL_PROMPT_GUARD_MODEL",)),
    ConfigKeySpec("dms.retrieval", "pipeline.query_rewriter_model", "string", "Qwen/Qwen3-8B", "warm", "Preferred query rewriter model", ("RETRIEVAL_QUERY_REWRITER_MODEL",)),
    ConfigKeySpec("dms.retrieval", "pipeline.compression_model", "string", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", "warm", "Preferred context compression model", ("RETRIEVAL_COMPRESSION_MODEL",)),
    ConfigKeySpec("dms.generation", "provider", "string", "auto", "hot", "Generation provider (auto|openai|openai_compatible|extractive)", ("GENERATION_PROVIDER",)),
    ConfigKeySpec("dms.generation", "model_id", "string", "gpt-5.5", "hot", "Default generation model id", ("GENERATION_MODEL_ID",)),
    ConfigKeySpec("dms.generation", "base_url", "string", None, "warm", "OpenAI-compatible generation base URL", ("GENERATION_BASE_URL", "OPENAI_BASE_URL")),
    ConfigKeySpec("dms.generation", "temperature", "float", 0.0, "hot", "Generation temperature", ("GENERATION_TEMPERATURE",)),
    ConfigKeySpec("dms.generation", "max_output_tokens", "integer", 1200, "hot", "Max generation tokens", ("GENERATION_MAX_OUTPUT_TOKENS",)),
    ConfigKeySpec("dms.generation", "min_evidence_chunks", "integer", 1, "hot", "Minimum retrieved chunks required to answer", ("GENERATION_MIN_EVIDENCE_CHUNKS",)),
    ConfigKeySpec("dms.generation", "min_evidence_score", "float", 0.0, "hot", "Minimum chunk score required for grounding", ("GENERATION_MIN_EVIDENCE_SCORE",)),
    ConfigKeySpec("dms.moderation", "enable_regex", "boolean", True, "hot", "Regex output moderation (mandatory; disable attempts ignored)", ("MODERATION_ENABLE_REGEX",)),
    ConfigKeySpec("dms.moderation", "enable_presidio", "boolean", True, "hot", "Optional Presidio layer; regex/policy remain mandatory", ("MODERATION_ENABLE_PRESIDIO",)),
    ConfigKeySpec("dms.moderation", "enable_llama_guard", "boolean", True, "hot", "Optional Llama Guard layer; local LLM moderator remains mandatory", ("MODERATION_ENABLE_LLAMA_GUARD",)),
    ConfigKeySpec("dms.moderation", "enable_policy_engine", "boolean", True, "hot", "Policy engine output moderation (mandatory; disable attempts ignored)", ("MODERATION_ENABLE_POLICY_ENGINE",)),
    ConfigKeySpec("dms.moderation", "guard_model", "string", "meta-llama/Llama-Guard-3-8B", "warm", "Preferred output guard model", ("MODERATION_GUARD_MODEL",)),
    ConfigKeySpec("dms.moderation", "block_prompt_leakage", "boolean", True, "hot", "Block responses leaking system prompt", ("MODERATION_BLOCK_PROMPT_LEAKAGE",)),
    ConfigKeySpec("dms.moderation", "block_jailbreak", "boolean", True, "hot", "Block jailbreak-style responses", ("MODERATION_BLOCK_JAILBREAK",)),
    ConfigKeySpec("dms.moderation", "presidio_score_threshold", "float", 0.5, "hot", "Minimum Presidio confidence to mask", ("MODERATION_PRESIDIO_SCORE_THRESHOLD",)),
    ConfigKeySpec("infra", "json_logging", "boolean", True, "warm", "Emit structured JSON application logs", ("INFRA_JSON_LOGGING",)),
    ConfigKeySpec("infra", "otel_enabled", "boolean", False, "warm", "Enable OpenTelemetry tracing", ("INFRA_OTEL_ENABLED", "OTEL_SDK_DISABLED")),
    ConfigKeySpec("infra", "otel_service_name", "string", "rag-builder", "warm", "OpenTelemetry service name", ("OTEL_SERVICE_NAME", "SERVICE_NAME")),
    ConfigKeySpec("infra", "prometheus_enabled", "boolean", True, "warm", "Enable Prometheus /metrics endpoint", ("INFRA_PROMETHEUS_ENABLED",)),
    ConfigKeySpec("infra", "ready_strict", "boolean", False, "hot", "Return HTTP 503 from /ready when critical deps fail", ("INFRA_READY_STRICT",)),
    ConfigKeySpec("infra", "retry.max_attempts", "integer", 3, "hot", "Default retry attempts", ("INFRA_RETRY_MAX_ATTEMPTS",)),
    ConfigKeySpec("infra", "retry.initial_delay_seconds", "float", 0.2, "hot", "Retry initial delay seconds", ("INFRA_RETRY_INITIAL_DELAY_SECONDS",)),
    ConfigKeySpec("infra", "circuit.failure_threshold", "integer", 5, "hot", "Circuit breaker failure threshold", ("INFRA_CIRCUIT_FAILURE_THRESHOLD",)),
    ConfigKeySpec("infra", "circuit.recovery_timeout_seconds", "float", 30.0, "hot", "Circuit breaker recovery timeout", ("INFRA_CIRCUIT_RECOVERY_TIMEOUT_SECONDS",)),
    ConfigKeySpec("infra", "embedder_model", "string", None, "warm", "Preferred embedder model/backend id", ("INFRA_EMBEDDER_MODEL", "EMBEDDING_MODEL")),
    ConfigKeySpec("infra", "llm_model", "string", None, "warm", "Preferred LLM/generation model id", ("INFRA_LLM_MODEL", "GENERATION_MODEL_ID")),
    ConfigKeySpec("infra", "vector_db_backend", "string", "weaviate", "warm", "Preferred vector database plugin", ("INFRA_VECTOR_DB_BACKEND", "VECTOR_DB_BACKEND")),
    # scheduler
    ConfigKeySpec("scheduler", "pool.max_parallel_jobs", "integer", 5, "warm", "Processor pool size", ("SCHEDULER_MAX_PARALLEL_JOBS",)),
    ConfigKeySpec("scheduler", "pool.port_range_start", "integer", 3100, "cold", "Processor port range start", ("SCHEDULER_PORT_RANGE_START",)),
    ConfigKeySpec("scheduler", "pool.port_range_end", "integer", 3110, "cold", "Processor port range end", ("SCHEDULER_PORT_RANGE_END",)),
    ConfigKeySpec("scheduler", "pool.processor_image", "string", "rag-processor:latest", "warm", "Processor Docker image", ("PROCESSOR_IMAGE_NAME",)),
    ConfigKeySpec("scheduler", "pool.processor_pool", "string", "", "warm", "Processor pool slot map (type:count,...)", ("SCHEDULER_PROCESSOR_POOL",)),
    ConfigKeySpec("scheduler", "poll.queue_check_frequency_sec", "integer", 5, "hot", "RabbitMQ poll interval", ("QUEUE_CHECK_FREQUENCY",)),
    ConfigKeySpec("scheduler", "poll.container_polling_frequency_sec", "integer", 5, "hot", "Container reconcile interval", ("CONTAINER_POLLING_FREQUENCY",)),
    ConfigKeySpec("scheduler", "control.tcp_port", "integer", 3200, "cold", "Scheduler TCP control port", ("SCHEDULER_TCP_PORT",)),
    ConfigKeySpec("scheduler", "control.timeout_seconds", "float", 30.0, "hot", "Scheduler client timeout", ("SCHEDULER_TIMEOUT_SECONDS",)),
    ConfigKeySpec("scheduler", "volumes.document_host_path", "string", "_documents", "warm", "Document bind mount host path", ("SCHEDULER_DOCUMENT_HOST_PATH",)),
    ConfigKeySpec("scheduler", "volumes.document_container_path", "string", "/app/documents", "warm", "Document bind mount container path", ("SCHEDULER_DOCUMENT_CONTAINER_PATH",)),
    # processor
    ConfigKeySpec("processor", "server.port", "integer", 8082, "cold", "Processor HTTP port", ("PROCESSOR_PORT",)),
    ConfigKeySpec("processor", "chunking.chunk_size", "integer", 512, "warm", "Default chunk size", ("CHUNK_SIZE",)),
    ConfigKeySpec("processor", "chunking.chunk_overlap_sentences", "integer", 1, "warm", "Default chunk overlap", ("CHUNK_OVERLAP_SENTENCES",)),
    ConfigKeySpec("processor", "chunking.min_content_words", "integer", 60, "warm", "Minimum words per chunk", ("MIN_CONTENT_WORDS",)),
    ConfigKeySpec("processor", "embedding.model_name", "string", "BAAI/bge-base-en", "warm", "Default embedding model", ("MODEL_NAME",)),
    ConfigKeySpec("processor", "embedding.model_dir", "string", None, "warm", "Local model path", ("MODEL_DIR",)),
    ConfigKeySpec("processor", "embedding.models_root", "string", "src/models", "cold", "Model weights root", ("MODELS_ROOT",)),
    ConfigKeySpec("processor", "runtime.max_workers", "integer", 4, "warm", "Embed worker pool size", ("MAX_WORKERS",)),
    ConfigKeySpec("processor", "runtime.log_level", "string", "INFO", "hot", "Log level", ("LOG_LEVEL",)),
    ConfigKeySpec("processor", "limits.max_chunks_per_document", "integer", 10000, "hot", "Safety cap on chunks per document"),
    ConfigKeySpec("processor", "job.heartbeat_interval_sec", "float", 12.0, "hot", "Job heartbeat interval", ("PROCESSOR_JOB_HEARTBEAT_INTERVAL_SEC",)),
    # dms.document_types
    ConfigKeySpec("dms.document_types", "hierarchy.max_depth", "integer", 3, "hot", "Max document type hierarchy depth"),
    ConfigKeySpec("dms.document_types", "seed.base_document_on_startup", "boolean", True, "hot", "Seed Base Document Type on startup"),
    # dms.access
    ConfigKeySpec("dms.access", "subscription.default_expiry_days", "integer", None, "hot", "Optional subscription TTL days"),
    ConfigKeySpec("dms.access", "api_key.hash_algorithm", "string", "sha256", "cold", "API key hash algorithm"),
    ConfigKeySpec("dms.access", "api_key.show_once", "boolean", True, "hot", "Show plaintext API key only once"),
    # dms.security
    ConfigKeySpec("dms.security", "validation.enabled", "boolean", True, "hot", "Upload compliance scan (always enforced; false is ignored with audit warning)"),
    ConfigKeySpec("dms.security", "validation.blocked_categories", "json", [
        "Personally Identifiable Information (PII)",
        "Protected Health Information (PHI)",
        "Credit/Debit Card Number",
        "Social Security Number (SSN)",
        "Aadhaar Number",
        "PAN Number",
        "Passport Number",
        "Driver's License",
        "Confidential Business Information",
        "Confidential Legal Information",
        "Intellectual Property / Trade Secrets",
        "Authentication Credentials/API Keys",
        "Private Key",
        "Bank Account Number",
        "Bank Routing Number (IFSC)"
    ], "hot", "List of sensitive data categories blocked by policy"),
]

REGISTRY_BY_KEY: dict[tuple[str, str], ConfigKeySpec] = {
    (spec.namespace, spec.key): spec for spec in CONFIG_REGISTRY
}

NAMESPACES: tuple[str, ...] = tuple(dict.fromkeys(spec.namespace for spec in CONFIG_REGISTRY))


def get_spec(namespace: str, key: str) -> ConfigKeySpec | None:
    return REGISTRY_BY_KEY.get((namespace, key))


def namespace_chain(namespace: str) -> list[str]:
    if namespace == "global":
        return ["global"]
    parts = namespace.split(".")
    chain: list[str] = []
    for i in range(len(parts), 0, -1):
        chain.append(".".join(parts[:i]))
    if "global" not in chain:
        chain.append("global")
    return chain


def resolve_env_value(spec: ConfigKeySpec) -> Any:
    if spec.env_resolver is not None:
        return spec.env_resolver()
    for alias in spec.env_aliases:
        raw = os.environ.get(alias)
        if raw is not None and str(raw).strip() != "":
            if spec.value_type == "boolean":
                return _env_bool(alias)
            if spec.value_type == "integer":
                return int(raw)
            if spec.value_type == "float":
                return float(raw)
            if spec.value_type == "json":
                import json

                return json.loads(raw)
            return strip_wrapping_quotes(str(raw))
    return spec.default
