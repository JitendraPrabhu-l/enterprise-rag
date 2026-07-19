"""Query embedding wrapper (ADR-004/ADR-005).

Wraps the same sentence-transformers model used at ingestion
(`BAAI/bge-small-en-v1.5` by default) so query vectors and document vectors
live in the same space and are directly comparable by Qdrant's cosine
distance. Model load is synchronous (sentence-transformers has no native
async API) and CPU/GPU-bound inference is offloaded to a thread so it never
blocks the event loop.

Optionally backed by the Redis embedding cache (ADR-013): repeated or
near-identical queries (a user refining a question, or the same question
asked by different users) skip the model entirely on a cache hit. Caching
is content-addressed and shared with the ingestion service's cache — same
model, same key scheme, one Redis instance.
"""

from __future__ import annotations

import asyncio

import structlog
from rag_core.embedding_cache import EmbeddingCache
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class QueryEmbedder:
    """Embeds query (or HyDE hypothetical-document) text for dense search."""

    def __init__(self, model_name: str, *, embedding_cache: EmbeddingCache | None = None) -> None:
        self._model_name = model_name
        self._embedding_cache = embedding_cache
        logger.info("embedder.loading", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("embedder.loaded", model=model_name)

    @property
    def dimension(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"model {self._model_name!r} did not report an embedding dimension")
        return int(dim)

    async def embed(self, text: str) -> list[float]:
        """Embed a single piece of text, returning a plain list[float] vector."""
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, consulting the cache first when configured.
        Only cache misses reach the model."""
        if not texts:
            return []

        cached: list[list[float] | None] = [None] * len(texts)
        if self._embedding_cache is not None:
            cached = await self._embedding_cache.get_many(self._model_name, texts)

        miss_indices = [i for i, vector in enumerate(cached) if vector is None]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            computed = await self._encode(miss_texts)
            for i, vector in zip(miss_indices, computed, strict=True):
                cached[i] = vector

            if self._embedding_cache is not None:
                await self._embedding_cache.set_many(
                    self._model_name, list(zip(miss_texts, computed, strict=True))
                )

        result: list[list[float]] = []
        for maybe_vector in cached:
            assert maybe_vector is not None  # every index was either a hit or filled above
            result.append(maybe_vector)
        return result

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        """Runs the model in a worker thread (sentence-transformers is sync/CPU-bound)."""
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
        )
        return [row.tolist() for row in embeddings]
