"""Utility package for src.processor_service.

Keep this module free of eager submodule imports. Several utility modules depend
on indexing dataclasses, while indexing modules import specific utilities. Eager
imports here create circular imports during API startup.
"""
