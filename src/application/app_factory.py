"""FastAPI application factory.

Keeping construction here lets deployments use a stable application-level
entrypoint while feature routers remain independent of delivery wiring.
"""

from .consumer_api.main import create_app

__all__ = ["create_app"]
