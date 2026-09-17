from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("REFERENCE_EXTRACTION_PORT", os.getenv("PROCESSOR_PORT", "3110")))
    uvicorn.run(
        "src.workers.reference_processor.api_server:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
