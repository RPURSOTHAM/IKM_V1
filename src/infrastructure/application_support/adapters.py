"""Adapters that wrap existing implementations behind enterprise interfaces.

Never reimplements business logic — only adapts call signatures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.infrastructure.application_support.interfaces import (
    Chunker,
    DLP,
    Embedder,
    LLM,
    Moderation,
    NER,
    Parser,
    Retriever,
    RiskEngine,
    VectorDatabase,
)


class DocumentLoaderParserAdapter(Parser):
    """Adapts processor DocumentLoader subclasses (pdfplumber / python-docx)."""

    name = "document_loader"

    def __init__(self, loader: Any | None = None) -> None:
        self._loader = loader

    def supports(self, path: Path | str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in {".pdf", ".docx", ".doc", ".txt", ".md"}

    def parse(self, path: Path | str, **kwargs: Any) -> Any:
        file_path = Path(path)
        if self._loader is not None:
            return self._loader.load(file_path, **kwargs)
        from src.features.documents.infrastructure.content.load import load_document_blocks

        return load_document_blocks(file_path)


class StrategyChunkerAdapter(Chunker):
    """Adapts chunk_document_with_strategy / semantic hierarchy chunking."""

    name = "strategy_chunker"

    def chunk(self, document: Any, **kwargs: Any) -> list[Any]:
        from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy

        strategy = kwargs.pop("strategy", None) or kwargs.pop("chunking_strategy", "semantic-hierarchy")
        blocks = document if isinstance(document, list) else kwargs.pop("blocks", document)
        doc_name = kwargs.pop("doc_name", None) or kwargs.pop("document_name", "document")
        chunk_size = int(kwargs.pop("chunk_size", 512))
        overlap_sentences = int(kwargs.pop("overlap_sentences", 1))
        min_content_words = int(kwargs.pop("min_content_words", 0))
        return list(
            chunk_document_with_strategy(
                blocks,
                doc_name,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap_sentences=overlap_sentences,
                min_content_words=min_content_words,
                **kwargs,
            )
        )


class SentenceTransformerEmbedderAdapter(Embedder):
    """Adapts existing processor embedding helpers; does not replace the model."""

    name = "sentence_transformers"

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id or "sentence-transformers/all-MiniLM-L6-v2"

    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        from src.features.embeddings.application.embedding_service import encode_texts

        model_name = kwargs.get("model_name") or self._model_id
        model_dir = kwargs.get("model_dir")
        vectors = encode_texts(texts, model_name, model_dir=model_dir)
        return [list(map(float, vector)) for vector in vectors]


class RetrievalServiceAdapter(Retriever):
    """Adapts DMS RetrievalService."""

    name = "retrieval_service"

    def __init__(self, service: Any | None = None) -> None:
        self._service = service

    def retrieve(self, query: str, **kwargs: Any) -> Any:
        service = self._service
        if service is None:
            from src.features.retrieval.application.retrieval_service import RetrievalService

            service = RetrievalService()
        request = kwargs.get("request")
        if request is not None:
            return service.retrieve(request)
        from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest

        payload = RetrieveRequest(query=query, **{k: v for k, v in kwargs.items() if k != "request"})
        return service.retrieve(payload)


class WeaviateVectorDatabaseAdapter(VectorDatabase):
    """Adapts existing Weaviate store helpers."""

    name = "weaviate"

    def upsert(self, collection: str, objects: Iterable[Any], **kwargs: Any) -> Any:
        import os

        from src.infrastructure.document_databases.weaviate_store import store_in_weaviate

        weaviate_url = kwargs.pop("weaviate_url", None) or os.getenv("WEAVIATE_URL") or ""
        return store_in_weaviate(
            list(objects),
            weaviate_url=weaviate_url,
            collection_name=collection,
            api_key=kwargs.pop("api_key", None) or os.getenv("WEAVIATE_API_KEY"),
            tenant_id=kwargs.pop("tenant_id", None),
            document_metadata=kwargs.pop("document_metadata", None),
        )

    def search(self, collection: str, query: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Use Retriever (RetrievalServiceAdapter) for search; Weaviate upsert is available here."
        )

    def health(self) -> dict[str, Any]:
        import os

        configured = bool(os.getenv("WEAVIATE_URL"))
        return {
            "ok": configured,
            "status": "configured" if configured else "unconfigured",
            "backend": self.name,
            "url": os.getenv("WEAVIATE_URL", ""),
        }


class GeneratorLLMAdapter(LLM):
    """Adapts shared Generator implementations (OpenAI-compatible / extractive)."""

    name = "generator"

    def __init__(self, generator: Any | None = None, model_id: str | None = None) -> None:
        self._generator = generator
        self._model_id = model_id or "auto"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        from src.features.generation.providers.llm_client import select_generator
        from src.features.generation.configuration.generation_config import GenerationConfig
        from src.features.generation.domain.models import GenerationRequest

        config = kwargs.get("config") or GenerationConfig.from_env()
        if kwargs.get("model_id"):
            config.model_id = kwargs["model_id"]
        generator = self._generator or select_generator(config)
        request = kwargs.get("request") or GenerationRequest(
            query=kwargs.get("query") or next((m["content"] for m in reversed(messages) if m.get("role") == "user"), ""),
            chunks=kwargs.get("chunks") or [],
            conversation=kwargs.get("conversation") or [],
            model_id=config.model_id,
            provider=config.provider,
        )
        return generator.generate(request, config, messages=messages)


class NERDetectorAdapter(NER):
    name = "ner_detector"

    def __init__(self, detector: Any | None = None) -> None:
        self._detector = detector

    def detect(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        detector = self._detector
        if detector is None:
            from src.features.security.dlp.ner_detector import NERDetector

            detector = NERDetector()
        return list(detector.detect_entities(text))


class ComplianceDLPAdapter(DLP):
    name = "compliance_scanner"

    def scan(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        from src.features.security.dlp.sensitive_data_detector import ComplianceScanner

        return list(ComplianceScanner().scan_text(text))

    def mask(self, text: str, **kwargs: Any) -> str:
        from src.features.security.dlp.sensitive_data_detector import mask_text_content

        return mask_text_content(text)


class OutputModerationAdapter(Moderation):
    name = "output_moderation"

    def moderate(self, text: str, **kwargs: Any) -> Any:
        from src.features.security.moderation.output_moderator import moderate_output

        return moderate_output(text, system_prompt=kwargs.get("system_prompt"), config=kwargs.get("config"))


class DLPRiskEngineAdapter(RiskEngine):
    name = "dlp_policy_engine"

    def evaluate(self, detections: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        from src.features.security.dlp.dlp_engine import DLPPolicyEngine

        return DLPPolicyEngine().evaluate_policy(
            detections=detections,
            doc_classification=kwargs.get("doc_classification") or {},
            topic_classification=kwargs.get("topic_classification") or {},
            similarity_results=kwargs.get("similarity_results") or {},
            moderation_results=kwargs.get("moderation_results") or {},
        )
