"""Uvicorn entrypoint: `python -m rag_ingestion.main`."""

from __future__ import annotations

import uvicorn

from rag_ingestion.config import IngestionSettings


def main() -> None:
    settings = IngestionSettings()
    uvicorn.run(
        "rag_ingestion.api:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
