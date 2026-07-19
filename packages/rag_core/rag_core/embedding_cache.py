"""Redis-backed embedding cache (ADR-013).

Content-addressed, not request-addressed: the cache key is a hash of the
model name plus the text itself, so identical text embedded from two
different documents (or two different services — ingestion and retrieval
both embed) transparently shares one cache entry. Including the model name
in the key makes a model swap self-invalidating rather than silently
serving stale vectors computed by a different model.

Fails open by design: any Redis error (connection refused, timeout) is
logged and treated as a cache miss on read, or silently dropped on write —
callers always get a correct embedding, just not necessarily a cached one.
Caching is a performance optimization here, not a correctness dependency.
"""

from __future__ import annotations

import hashlib
import json

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # one week


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()
    return f"embcache:v1:{digest}"


class EmbeddingCache:
    """Async Redis client wrapper scoped to embedding-vector caching only."""

    def __init__(self, redis_url: str, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        # redis-py's `from_url` classmethod is untyped internally even though
        # the package ships py.typed — a known gap in its stub coverage, not
        # fixable from the call site.
        self._client: redis.Redis = redis.from_url(  # type: ignore[no-untyped-call]
            redis_url, decode_responses=True
        )
        self._ttl_seconds = ttl_seconds

    async def get_many(self, model_name: str, texts: list[str]) -> list[list[float] | None]:
        """Returns one entry per input text, in order; `None` where no cache hit exists."""
        if not texts:
            return []
        keys = [_cache_key(model_name, text) for text in texts]
        try:
            raw_values = await self._client.mget(keys)
        except redis.RedisError as exc:
            logger.warning("embedding_cache.get_failed", error=str(exc))
            return [None] * len(texts)

        results: list[list[float] | None] = []
        for raw in raw_values:
            if raw is None:
                results.append(None)
                continue
            try:
                results.append(json.loads(raw))
            except json.JSONDecodeError:
                results.append(None)
        return results

    async def set_many(self, model_name: str, items: list[tuple[str, list[float]]]) -> None:
        """Best-effort write-back; a failure here never raises."""
        if not items:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for text, vector in items:
                pipe.set(_cache_key(model_name, text), json.dumps(vector), ex=self._ttl_seconds)
            await pipe.execute()
        except redis.RedisError as exc:
            logger.warning("embedding_cache.set_failed", error=str(exc))

    async def close(self) -> None:
        await self._client.aclose()
