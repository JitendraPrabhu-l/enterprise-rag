"""Uvicorn entrypoint for the rag-eval production sampling API."""

from __future__ import annotations

import uvicorn

from rag_eval.config import EvalSettings


def main() -> None:
    settings = EvalSettings()
    uvicorn.run(
        "rag_eval.api:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
