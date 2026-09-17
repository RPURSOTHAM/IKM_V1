"""Deprecated compatibility namespace for the former processor service."""

import importlib
import sys

_ALIASES = {
    "security": "src.features.security",
    "indexing": "src.features.indexing.application",
    "databases": "src.infrastructure.document_databases",
    "core": "src.features.document_processing.core",
    "extractors": "src.features.document_processing.extractors",
    "key_fields": "src.features.document_processing.key_fields",
    "loaders": "src.features.document_processing.loaders",
    "metadata": "src.features.document_processing.metadata",
    "processors": "src.features.document_processing.processors",
    "services": "src.features.document_processing.services",
    "utils": "src.features.document_processing.utilities",
    "validation": "src.features.document_validation.application.validation_pipeline",
}

for _old, _new in _ALIASES.items():
    _module = importlib.import_module(_new)
    sys.modules[f"{__name__}.{_old}"] = _module
    setattr(sys.modules[__name__], _old, _module)
