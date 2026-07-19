"""Query embedder used ONLY for semantic-cache keys (ADR-026).

Deliberately separate from retrieval's `QueryEmbedder`: the generation
service does not embed for search (it brokers retrieval over HTTP), so this
exists purely to turn a query into the vector the answer cache compares
against. Same model as retrieval/ingestion, so "semantically identical"
carries the identical meaning across services.

Synchronous model, thread-offloaded inference (sentence-transformers has no
async API) — one query per call, so the cost is negligible next to the
retrieval + generation it may let us skip.
"""

from __future__ import annotations

import asyncio

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class CacheKeyEmbedder:
    def __init__(self, model_name: str) -> None:
        logger.info("cache_embedder.loading", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("cache_embedder.loaded", model=model_name)

    async def embed(self, text: str) -> list[float]:
        vector = await asyncio.to_thread(
            self._model.encode, text, normalize_embeddings=True
        )
        return vector.tolist()
