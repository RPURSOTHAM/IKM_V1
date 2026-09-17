"""Medical Literacy Research config — PubMed / ClinicalTrials / Azure OpenAI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# configuration/ → medical_literature → features → src → rag-builder
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PUBMED_ENV_CANDIDATES = (
    _REPO_ROOT / "deploy" / "application" / ".env",
    _REPO_ROOT / ".env",
)


def _load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_all_dotenv() -> None:
    for path in _PUBMED_ENV_CANDIDATES:
        _load_dotenv_file(path)


def load_config() -> MedicalLiteracyConfig:
    _load_all_dotenv()
    return MedicalLiteracyConfig.from_env()


def _env_first(*names: str, default: str = "") -> str:
    """Prefer the first set env var (PubMed names, then legacy MLR_* aliases)."""
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return default


def _env_bool(*names: str, default: bool) -> bool:
    raw = _env_first(*names, default="").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PubmedChatRetrievalSettings:
    initial_top_k: int
    detail_top_k: int
    max_expanded_queries: int
    use_rerank: bool
    expand_query: bool
    parallel_retrieval: bool
    single_retrieve_call: bool
    chunk_max_chars: int

    @classmethod
    def from_env(cls) -> PubmedChatRetrievalSettings:
        _load_all_dotenv()
        return cls(
            initial_top_k=max(
                1, int(_env_first("PUBMED_CHAT_INITIAL_TOP_K", "MLR_CHAT_INITIAL_TOP_K", default="6"))
            ),
            detail_top_k=max(
                1, int(_env_first("PUBMED_CHAT_DETAIL_TOP_K", "MLR_CHAT_DETAIL_TOP_K", default="8"))
            ),
            max_expanded_queries=max(
                1,
                int(
                    _env_first(
                        "PUBMED_CHAT_MAX_EXPANDED_QUERIES",
                        "MLR_CHAT_MAX_EXPANDED_QUERIES",
                        default="1",
                    )
                ),
            ),
            use_rerank=_env_bool("PUBMED_CHAT_USE_RERANK", "MLR_CHAT_USE_RERANK", default=True),
            expand_query=_env_bool("PUBMED_CHAT_EXPAND_QUERY", "MLR_CHAT_EXPAND_QUERY", default=True),
            parallel_retrieval=_env_bool(
                "PUBMED_CHAT_PARALLEL_RETRIEVAL", "MLR_CHAT_PARALLEL_RETRIEVAL", default=True
            ),
            single_retrieve_call=_env_bool(
                "PUBMED_CHAT_SINGLE_RETRIEVE_CALL", "MLR_CHAT_SINGLE_RETRIEVE_CALL", default=True
            ),
            chunk_max_chars=max(
                200,
                int(_env_first("PUBMED_CHAT_CHUNK_MAX_CHARS", "MLR_CHAT_CHUNK_MAX_CHARS", default="1500")),
            ),
        )


def load_chat_retrieval_settings() -> PubmedChatRetrievalSettings:
    return PubmedChatRetrievalSettings.from_env()


@dataclass(frozen=True)
class MedicalLiteracyConfig:
    pubmed_api_key: str = ""
    pubmed_email: str = ""
    pubmed_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    clinicaltrials_url: str = "https://clinicaltrials.gov/api/v2"
    pubmed_max_results: int = 25
    clinicaltrials_max_results: int = 25
    llm_provider: str = "azure_openai"
    llm_timeout_seconds: int = 180
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float | None = None
    absolute_max_results: int = 100
    pubmed_tool: str = "rag-builder-pubmed"

    @classmethod
    def from_env(cls) -> MedicalLiteracyConfig:
        _load_all_dotenv()
        provider = _env_first(
            "PUBMED_LLM_PROVIDER", "MLR_LLM_PROVIDER", default="azure_openai"
        ).lower()
        temp_raw = _env_first("PUBMED_LLM_TEMPERATURE", "MLR_LLM_TEMPERATURE", default="")
        if temp_raw:
            llm_temperature: float | None = float(temp_raw)
        elif provider == "openai":
            llm_temperature = 0.2
        else:
            llm_temperature = None
        return cls(
            pubmed_api_key=(os.getenv("PUBMED_API_KEY") or os.getenv("NCBI_API_KEY") or "").strip(),
            pubmed_email=(os.getenv("PUBMED_EMAIL") or os.getenv("NCBI_EMAIL") or "").strip(),
            pubmed_url=os.getenv(
                "PUBMED_URL",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            ).rstrip("/"),
            clinicaltrials_url=os.getenv(
                "CLINICALTRIALS_URL",
                "https://clinicaltrials.gov/api/v2",
            ).rstrip("/"),
            pubmed_max_results=int(
                _env_first("PUBMED_MAX_RESULTS", "MLR_PUBMED_MAX_RESULTS", default="25")
            ),
            clinicaltrials_max_results=int(
                _env_first(
                    "PUBMED_CLINICALTRIALS_MAX_RESULTS",
                    "MLR_CLINICALTRIALS_MAX_RESULTS",
                    default="25",
                )
            ),
            llm_provider=provider,
            llm_timeout_seconds=int(
                _env_first("PUBMED_LLM_TIMEOUT_SECONDS", "MLR_LLM_TIMEOUT_SECONDS", default="180")
            ),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/"),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip(),
            azure_openai_api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ).strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            llm_temperature=llm_temperature,
            absolute_max_results=int(os.getenv("EVIDENCE_ABSOLUTE_MAX_RESULTS", "100")),
            pubmed_tool=os.getenv("PUBMED_TOOL_NAME", "rag-builder-pubmed") or "rag-builder-pubmed",
        )

    def llm_configured(self) -> bool:
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return bool(
            self.azure_openai_api_key
            and self.azure_openai_endpoint
            and self.azure_openai_deployment
        )

    def resolve_pubmed_sources(self, repo_sources: dict | None) -> dict[str, str]:
        sources = dict(repo_sources or {})
        return {
            "api_key": str(sources.get("pubmed_api_key") or self.pubmed_api_key).strip(),
            "email": str(sources.get("pubmed_email") or self.pubmed_email).strip(),
            "base_url": str(sources.get("pubmed_url") or self.pubmed_url).rstrip("/"),
        }

    def resolve_clinicaltrials_url(self, repo_sources: dict | None) -> str:
        sources = dict(repo_sources or {})
        return str(sources.get("clinicaltrials_url") or self.clinicaltrials_url).rstrip("/")

    def clamp_max_results(self, requested: int | None) -> int:
        value = self.pubmed_max_results if requested is None else int(requested)
        if value < 1:
            raise ValueError("max_results must be >= 1")
        if value > self.absolute_max_results:
            raise ValueError(f"max_results must be <= {self.absolute_max_results}")
        return value

    def build_pubmed_client(self):
        from src.features.medical_literature.providers.pubmed_client import PubMedClient

        sources = self.resolve_pubmed_sources(None)
        return PubMedClient(
            base_url=sources["base_url"],
            api_key=sources["api_key"],
            email=sources["email"],
            tool=self.pubmed_tool,
        )

    def build_clinicaltrials_client(self):
        from src.features.medical_literature.providers.clinical_trials_client import (
            ClinicalTrialsClient,
        )

        return ClinicalTrialsClient(base_url=self.clinicaltrials_url)


# Back-compat name used by EvidenceService / tests.
EvidenceConfig = MedicalLiteracyConfig
