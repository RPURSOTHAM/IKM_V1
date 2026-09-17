"""Enterprise component interfaces.

Business logic depends on these contracts. Concrete backends are registered
via the plugin registry and may be swapped without changing call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class Parser(ABC):
    """Parse a source document into structured blocks/content."""

    name: str = "parser"

    @abstractmethod
    def parse(self, path: Path | str, **kwargs: Any) -> Any:
        raise NotImplementedError

    def supports(self, path: Path | str) -> bool:
        return True


class Chunker(ABC):
    """Split a parsed document into semantic chunks."""

    name: str = "chunker"

    @abstractmethod
    def chunk(self, document: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError


class Embedder(ABC):
    """Produce vectors for texts or chunks."""

    name: str = "embedder"

    @abstractmethod
    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return getattr(self, "_model_id", self.name)

    @property
    def dimensions(self) -> int | None:
        return getattr(self, "_dimensions", None)


class Retriever(ABC):
    """Retrieve candidates for a query."""

    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> Any:
        raise NotImplementedError


class VectorDatabase(ABC):
    """Persist and query vectors with metadata."""

    name: str = "vector_database"

    @abstractmethod
    def upsert(self, collection: str, objects: Iterable[Any], **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def search(self, collection: str, query: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"status": "unknown", "backend": self.name}


class LLM(ABC):
    """Large language model / generation backend."""

    name: str = "llm"

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return getattr(self, "_model_id", self.name)


class NER(ABC):
    """Named entity recognition."""

    name: str = "ner"

    @abstractmethod
    def detect(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError


class DLP(ABC):
    """Data loss prevention scanning / masking."""

    name: str = "dlp"

    @abstractmethod
    def scan(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    def mask(self, text: str, **kwargs: Any) -> str:
        return text


class Moderation(ABC):
    """Input or output moderation."""

    name: str = "moderation"

    @abstractmethod
    def moderate(self, text: str, **kwargs: Any) -> Any:
        raise NotImplementedError


class RiskEngine(ABC):
    """Aggregate detections into an allow / mask / block / review decision."""

    name: str = "risk_engine"

    @abstractmethod
    def evaluate(self, detections: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


# Re-export generation/retrieval contracts for a single discovery point.
try:
    from src.features.generation.domain.interfaces import Generator, GenerationPromptBuilder
except Exception:  # pragma: no cover - optional import during bootstrap
    Generator = ABC  # type: ignore[misc, assignment]
    GenerationPromptBuilder = ABC  # type: ignore[misc, assignment]

try:
    from src.features.retrieval.domain.interfaces import ModelRegistry, RetrievalStage
except Exception:  # pragma: no cover
    ModelRegistry = ABC  # type: ignore[misc, assignment]
    RetrievalStage = ABC  # type: ignore[misc, assignment]


COMPONENT_ROLES: tuple[str, ...] = (
    "parser",
    "chunker",
    "embedder",
    "retriever",
    "vector_database",
    "llm",
    "ner",
    "dlp",
    "moderation",
    "risk_engine",
)
