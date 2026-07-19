"""Cross-encoder reranking, stage two of the two-stage retrieval flow (ADR-005).

Wraps a self-hosted `sentence-transformers` CrossEncoder (default
`BAAI/bge-reranker-base`) behind a narrow `Reranker` interface so the wide
top_k (40-50) candidate set from hybrid+RRF can be re-scored jointly on
(query, passage) pairs — the cross-encoder attends over both texts at once,
which is far more precise than the bi-encoder cosine similarity used for the
initial wide retrieval, at the cost of being too slow to run over the full
corpus.
"""

from __future__ import annotations

import asyncio

import structlog
from rag_core.schemas import RetrievedChunk
from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)


class Reranker:
    """Cross-encoder reranker: scores (query, passage) pairs jointly."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        logger.info("reranker.loading", model=model_name)
        self._model = CrossEncoder(model_name)
        logger.info("reranker.loaded", model=model_name)

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Re-score `candidates` against `query` and return the top_n, best first.

        Sets `rerank_score` on each returned `RetrievedChunk` (via
        `model_copy`, since `RetrievedChunk` is an immutable-by-convention
        pydantic model passed around the pipeline). Candidates are scored
        using each chunk's child text, which is what the cross-encoder was
        fine-tuned to compare against a query.
        """
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        loop = asyncio.get_running_loop()
        raw_scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, convert_to_numpy=True),
        )

        scored = [
            candidate.model_copy(update={"rerank_score": float(score)})
            for candidate, score in zip(candidates, raw_scores, strict=True)
        ]
        scored.sort(key=lambda c: c.rerank_score or float("-inf"), reverse=True)
        return scored[:top_n]
