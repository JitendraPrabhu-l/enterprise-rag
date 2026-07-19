"""Uvicorn entrypoint for the retrieval service."""

from __future__ import annotations

import os

import uvicorn


def run() -> None:
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "rag_retrieval.api:app",
        host="0.0.0.0",
        port=port,
        log_config=None,  # structlog owns logging config; don't let uvicorn override it.
    )


if __name__ == "__main__":
    run()
