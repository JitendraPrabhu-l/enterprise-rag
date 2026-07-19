from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from pyrate_limiter import Duration, InMemoryBucket, Rate

from rag_core.rate_limit import FailOpenRateLimiter, build_route_limiter


@pytest.mark.asyncio
class TestFailOpenRateLimiter:
    async def test_delegates_to_the_wrapped_limiter_on_success(self) -> None:
        inner = AsyncMock()
        wrapper = FailOpenRateLimiter(inner)
        request = MagicMock()
        response = MagicMock()

        await wrapper(request, response)

        inner.assert_awaited_once_with(request, response)

    async def test_redis_error_is_swallowed_not_raised(self) -> None:
        """Fail-open contract (ADR-013): a rate-limit backend outage must
        never turn into a 500 on every request — it degrades to
        'temporarily unavailable, allow the request.'"""
        inner = AsyncMock(side_effect=redis.RedisError("connection refused"))
        wrapper = FailOpenRateLimiter(inner)

        await wrapper(MagicMock(), MagicMock())  # must not raise

    async def test_non_redis_errors_still_propagate(self) -> None:
        """Only the Redis-backend failure mode is fail-open; a genuine bug
        inside the wrapped limiter (e.g. a real 429 HTTPException from
        fastapi_limiter's default_callback) must still surface normally."""
        inner = AsyncMock(side_effect=ValueError("not a redis error"))
        wrapper = FailOpenRateLimiter(inner)

        with pytest.raises(ValueError, match="not a redis error"):
            await wrapper(MagicMock(), MagicMock())


def _fake_bucket_init(rates: list[Rate]) -> InMemoryBucket:
    """`Limiter.__init__` does a real `isinstance(bucket, AbstractBucket)`
    check, so a bare MagicMock() fails it — use a real (in-memory, not
    Redis) bucket to keep the test isolated from an actual Redis connection
    while still exercising real pyrate_limiter construction code."""
    return InMemoryBucket(rates)


@pytest.mark.asyncio
class TestBuildRouteLimiter:
    async def test_constructs_a_bucket_scoped_to_the_given_key(self) -> None:
        redis_client = MagicMock()
        captured_args: tuple[object, ...] = ()

        async def fake_init(*args: object) -> InMemoryBucket:
            nonlocal captured_args
            captured_args = args
            rates = args[0]
            assert isinstance(rates, list)
            return _fake_bucket_init(rates)

        with patch("rag_core.rate_limit.RedisBucket.init", side_effect=fake_init):
            await build_route_limiter(
                redis_client, requests_per_minute=42, bucket_key="ingest-route"
            )

        rates, passed_client, bucket_key = captured_args
        assert passed_client is redis_client
        assert bucket_key == "ingest-route"
        assert isinstance(rates, list) and len(rates) == 1
        assert rates[0].limit == 42
        assert rates[0].interval == Duration.MINUTE

    async def test_returns_a_fail_open_wrapper(self) -> None:
        redis_client = MagicMock()

        async def fake_init(*args: object) -> InMemoryBucket:
            rates = args[0]
            assert isinstance(rates, list)
            return _fake_bucket_init(rates)

        with patch("rag_core.rate_limit.RedisBucket.init", side_effect=fake_init):
            result = await build_route_limiter(redis_client, requests_per_minute=10, bucket_key="k")
        assert isinstance(result, FailOpenRateLimiter)
