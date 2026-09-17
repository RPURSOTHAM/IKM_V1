"""Cross-feature dependency accessors used by delivery code."""

from src.features.repositories.application.repository_service import get_repository_service
from src.features.retrieval.application.retrieval_service import get_retrieval_service

__all__ = ["get_repository_service", "get_retrieval_service"]
