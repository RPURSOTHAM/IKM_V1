"""Generation interfaces. Adapter-first; do not replace existing chat guards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.features.generation.configuration.generation_config import GenerationConfig
from src.features.generation.domain.models import GenerationRequest, GenerationResult


class Generator(ABC):
    """Replaceable generation backend."""

    name: str = "generator"

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        config: GenerationConfig,
        *,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        raise NotImplementedError


class GenerationPromptBuilder(ABC):
    @abstractmethod
    def build_messages(
        self,
        request: GenerationRequest,
        config: GenerationConfig,
        *,
        citations: list[Any],
        public_context: str,
        public_metadata: dict[str, Any],
    ) -> list[dict[str, str]]:
        raise NotImplementedError
