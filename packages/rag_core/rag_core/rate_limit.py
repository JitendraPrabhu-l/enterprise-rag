"""Redis-backed rate limiting (ADR-013), shared construction for every service.

Built on `pyrate_limiter.RedisBucket` — a Lua-script-backed sliding window
evaluated atomically inside Redis (not a naive read-then-increment, which
races under concurrent requests) — wrapped by `fastapi_limiter.RateLimiter`
as a per-route FastAPI dependency.

The underlying `Limiter.try_acquire_async` raises on a genuine Redis
connection error rather than degrading gracefully on its own, so
`FailOpenRateLimiter` wraps the dependency call to match the embedding
cache's fail-open posture (ADR-013): a Redis outage must never turn into a
500 on every request across every service — it degrades to "rate limiting
temporarily unavailable, allow the request."
"""

from __future__ import annotations

from typing import cast

import redis.asyncio as redis
import structlog
from fastapi import Request, Response
from fastapi_limiter.depends import RateLimiter
from pyrate_limiter import Duration, Limiter, Rate, RedisBucket

logger = structlog.get_logger(__name__)


class FailOpenRateLimiter:
    """Wraps a `RateLimiter` FastAPI dependency so a Redis error is logged
    and treated as "allow the request" rather than propagating as a 500."""

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    async def __call__(self, request: Request, response: Response) -> None:
        try:
            await self._limiter(request, response)
        except redis.RedisError as exc:
            logger.warning("rate_limit.backend_unavailable", error=str(exc))


async def build_route_limiter(
    redis_client: redis.Redis, *, requests_per_minute: int, bucket_key: str
) -> FailOpenRateLimiter:
    """Constructs one rate-limiting FastAPI dependency, scoped to `bucket_key`
    (distinct routes should use distinct keys so a burst on `/ingest` doesn't
    consume `/healthz`'s budget)."""
    rate = Rate(requests_per_minute, Duration.MINUTE)
    bucket = await RedisBucket.init([rate], redis_client, bucket_key)
    limiter = Limiter(bucket)
    return FailOpenRateLimiter(RateLimiter(limiter=limiter))


def get_redis_client(redis_url: str) -> redis.Redis:
    # Same untyped-classmethod stub gap as rag_core.embedding_cache.
    return cast(redis.Redis, redis.from_url(redis_url, decode_responses=False))  # type: ignore[no-untyped-call]
