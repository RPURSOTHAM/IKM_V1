from __future__ import annotations

import sys
from pathlib import Path

# Allow `python main.py` from this directory: package root is the parent `src/`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from src.features.document_processing.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "src.workers.document_processor.api_server:app",
        host="0.0.0.0",
        port=settings.processor_port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
