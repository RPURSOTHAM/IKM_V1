"""Deprecated compatibility namespace for the pre-feature package layout."""

import importlib
import sys

_ALIASES = {
    "consumer_api_service": "src.application.consumer_api",
    "document_receiver_service": "src.features.documents.application.document_service",
    "document_review_service": "src.features.human_review.application.review_service",
    "document_type_service": "src.features.document_types.application.document_type_service",
    "evidence_service": "src.features.medical_literature.application.literature_research_service",
    "generation_service": "src.features.generation.application.generation_service",
    "platform_config": "src.features.configuration",
    "platform_config_service": "src.features.configuration.application.platform_config_service",
    "platform_security_service": "src.features.users.application.user_service",
    "queue_admin_service": "src.features.jobs.application.queue_admin_service",
    "repository_service": "src.features.repositories.application.repository_service",
    "retrieval_service": "src.features.retrieval.application.retrieval_service",
    "system_service": "src.features.system.api.system_routes",
    "weaviate_client": "src.infrastructure.vector_store.dms_weaviate_client",
    "errors": "src.shared.errors",
}

for _old, _new in _ALIASES.items():
    _module = importlib.import_module(_new)
    sys.modules[f"{__name__}.{_old}"] = _module
    setattr(sys.modules[__name__], _old, _module)
