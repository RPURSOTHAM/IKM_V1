"""Core PDF extraction: text, tables, images, and infographic clips."""

from .extract import extract_document
from .metadata_extraction import extract_metadata

__all__ = ["extract_document", "extract_metadata"]
