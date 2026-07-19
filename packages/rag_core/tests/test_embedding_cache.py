from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis

from rag_core.embedding_cache import EmbeddingCache, _cache_key


def _mock_pipeline() -> MagicMock:
    """redis.asyncio pipelines queue commands synchronously (`.set(...)`
    returns the pipeline itself, not a coroutine) and only `.execute()` is
    actually async — a plain AsyncMock() for the whole object would make
    every attribute awaitable, which doesn't match the real client."""
    pipe = MagicMock()
    pipe.execute = AsyncMock()
    return pipe


class TestCacheKey:
    def test_same_model_and_text_produce_same_key(self) -> None:
        assert _cache_key("model-a", "hello") == _cache_key("model-a", "hello")

    def test_different_models_produce_different_keys(self) -> None:
        assert _cache_key("model-a", "hello") != _cache_key("model-b", "hello")

    def test_different_text_produces_different_keys(self) -> None:
        assert _cache_key("model-a", "hello") != _cache_key("model-a", "goodbye")

    def test_key_is_namespaced(self) -> None:
        assert _cache_key("model-a", "hello").startswith("embcache:v1:")


@pytest.mark.asyncio
class TestGetMany:
    async def test_empty_input_returns_empty_list_without_calling_redis(self) -> None:
        cache = EmbeddingCache("redis://localhost:6379/0")
        with patch.object(cache._client, "mget", new=AsyncMock()) as mock_mget:
            result = await cache.get_many("model-a", [])
        assert result == []
        mock_mget.assert_not_awaited()

    async def test_all_misses_returns_none_for_each_text(self) -> None:
        cache = EmbeddingCache("redis://localhost:6379/0")
        with patch.object(cache._client, "mget", new=AsyncMock(return_value=[None, None])):
            result = await cache.get_many("model-a", ["a", "b"])
        assert result == [None, None]

    async def test_hit_deserializes_the_stored_vector(self) -> None:
        cache = EmbeddingCache("redis://localhost:6379/0")
        stored = json.dumps([0.1, 0.2, 0.3])
        with patch.object(cache._client, "mget", new=AsyncMock(return_value=[stored, None])):
            result = await cache.get_many("model-a", ["a", "b"])
        assert result == [[0.1, 0.2, 0.3], None]

    async def test_redis_error_on_read_is_treated_as_all_misses(self) -> None:
        """Fail-open contract (ADR-013): a cache backend outage must never
        surface to the caller as an error — it degrades to always-recompute."""
        cache = EmbeddingCache("redis://localhost:6379/0")
        with patch.object(
            cache._client, "mget", new=AsyncMock(side_effect=redis.RedisError("down"))
        ):
            result = await cache.get_many("model-a", ["a", "b", "c"])
        assert result == [None, None, None]

    async def test_corrupt_stored_value_is_treated_as_a_miss_not_an_error(self) -> None:
        cache = EmbeddingCache("redis://localhost:6379/0")
        with patch.object(cache._client, "mget", new=AsyncMock(return_value=["not valid json{{"])):
            result = await cache.get_many("model-a", ["a"])
        assert result == [None]


@pytest.mark.asyncio
class TestSetMany:
    async def test_empty_input_does_not_touch_redis(self) -> None:
        cache = EmbeddingCache("redis://localhost:6379/0")
        with patch.object(cache._client, "pipeline") as mock_pipeline:
            await cache.set_many("model-a", [])
        mock_pipeline.assert_not_called()

    async def test_writes_go_through_a_pipeline_with_ttl(self) -> None:
        cache = EmbeddingCache("redis://localhost:6379/0", ttl_seconds=123)
        mock_pipe = _mock_pipeline()
        with patch.object(cache._client, "pipeline", return_value=mock_pipe) as mock_pipeline:
            await cache.set_many("model-a", [("hello", [0.1, 0.2])])

        mock_pipeline.assert_called_once_with(transaction=False)
        mock_pipe.set.assert_called_once()
        args, kwargs = mock_pipe.set.call_args
        assert args[0] == _cache_key("model-a", "hello")
        assert json.loads(args[1]) == [0.1, 0.2]
        assert kwargs["ex"] == 123
        mock_pipe.execute.assert_awaited_once()

    async def test_redis_error_on_write_does_not_raise(self) -> None:
        """Fail-open contract (ADR-013): a write failure is swallowed, not
        propagated — caching is a performance optimization, not a
        correctness dependency, so callers must never see this fail."""
        cache = EmbeddingCache("redis://localhost:6379/0")
        mock_pipe = _mock_pipeline()
        mock_pipe.execute.side_effect = redis.RedisError("down")
        with patch.object(cache._client, "pipeline", return_value=mock_pipe):
            await cache.set_many("model-a", [("hello", [0.1, 0.2])])  # must not raise
