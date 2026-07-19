"""Uvicorn entrypoint for the generation service."""

from __future__ import annotations

import uvicorn

from rag_generation.config import GenerationSettings


def main() -> None:
    settings = GenerationSettings()
    uvicorn.run(
        "rag_generation.api:app",
        host="0.0.0.0",  # noqa: S104 - intentional bind-all inside a container
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
