"""Tests for `SemanticAnswerCache` (ADR-026).

The Redis client is faked with an in-memory stand-in exposing exactly the
subset of the redis-py async API this module calls (`lrange`, `pipeline`
with `lpush`/`ltrim`/`expire`/`execute`) — real Redis semantics for those
calls, no network. This exercises the real similarity/scoping logic while
staying fast and dependency-free.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_core.schemas import Citation, GenerationResponse
from rag_core.semantic_cache import SemanticAnswerCache, _cosine, _scope_key


class _FakePipeline:
    def __init__(self, store: dict[str, list[str]]) -> None:
        self._store = store
        self._ops: list[tuple[str, tuple]] = []

    def lpush(self, key: str, value: str) -> _FakePipeline:
        self._ops.append(("lpush", (key, value)))
        return self

    def ltrim(self, key: str, start: int, end: int) -> _FakePipeline:
        self._ops.append(("ltrim", (key, start, end)))
        return self

    def expire(self, key: str, ttl: int) -> _FakePipeline:
        self._ops.append(("expire", (key, ttl)))
        return self

    async def execute(self) -> None:
        for op, args in self._ops:
            if op == "lpush":
                key, value = args
                self._store.setdefault(key, []).insert(0, value)
            elif op == "ltrim":
                key, start, end = args
                self._store[key] = self._store.get(key, [])[start : end + 1]
            # expire is a no-op for the fake — TTL expiry isn't under test here.


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return self.store.get(key, [])[start : end + 1 if end != -1 else None]

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store)

    async def aclose(self) -> None:
        pass


def _make_cache(threshold: float = 0.95) -> tuple[SemanticAnswerCache, _FakeRedis]:
    cache = SemanticAnswerCache("redis://fake", similarity_threshold=threshold)
    fake = _FakeRedis()
    cache._client = fake  # type: ignore[assignment]
    return cache, fake


def _response(answer: str = "Revenue was $12.4M.", flagged: bool = False) -> GenerationResponse:
    return GenerationResponse(
        request_id=uuid4(),
        answer=answer,
        citations=[Citation(parent_id="p1", document_id="d1", page_number=3)],
        model="test-model",
        guardrail_flagged=flagged,
    )


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        assert _cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self) -> None:
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_mismatched_dimensions_scores_below_any_threshold(self) -> None:
        """A dimension mismatch (e.g. embedding model swap mid-flight) must
        never accidentally register as a hit — it's an unambiguous miss."""
        assert _cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == -1.0

    def test_zero_vector_does_not_divide_by_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == -1.0


class TestScopeKeyIsolation:
    """The security-critical property: a cached answer must only be
    servable to a caller who could have retrieved the same documents."""

    def test_different_tenants_get_different_scope_keys(self) -> None:
        a = _scope_key("tenant-a", ["public"], None)
        b = _scope_key("tenant-b", ["public"], None)
        assert a != b

    def test_different_principals_get_different_scope_keys(self) -> None:
        a = _scope_key("t", ["user:alice"], None)
        b = _scope_key("t", ["user:bob"], None)
        assert a != b

    def test_different_source_domains_get_different_scope_keys(self) -> None:
        a = _scope_key("t", ["public"], ["sec-filings"])
        b = _scope_key("t", ["public"], ["arxiv-cs"])
        assert a != b

    def test_principal_order_does_not_matter(self) -> None:
        """The caller's group memberships may arrive in any order — the
        scope must not fragment across equivalent principal sets."""
        a = _scope_key("t", ["group:eng", "user:alice"], None)
        b = _scope_key("t", ["user:alice", "group:eng"], None)
        assert a == b

    def test_domain_order_does_not_matter(self) -> None:
        a = _scope_key("t", ["public"], ["a", "b"])
        b = _scope_key("t", ["public"], ["b", "a"])
        assert a == b


@pytest.mark.asyncio
class TestLookupAndStore:
    async def test_store_then_lookup_with_identical_embedding_hits(self) -> None:
        cache, _ = _make_cache()
        response = _response()
        vector = [1.0, 0.0, 0.0]

        await cache.store(
            query_embedding=vector, response=response,
            tenant_id="t", principals=["public"], source_domains=None,
        )
        hit = await cache.lookup(
            query_embedding=vector, tenant_id="t", principals=["public"], source_domains=None
        )

        assert hit is not None
        assert hit.answer == response.answer
        assert hit.citations == response.citations

    async def test_lookup_with_similar_but_not_identical_embedding_hits(self) -> None:
        """The whole point: near-paraphrase queries (cosine >= threshold)
        must hit, not just byte-identical ones."""
        cache, _ = _make_cache(threshold=0.95)
        response = _response()
        await cache.store(
            query_embedding=[1.0, 0.0, 0.0], response=response,
            tenant_id="t", principals=["public"], source_domains=None,
        )

        # A small perturbation staying above the 0.95 cosine threshold.
        hit = await cache.lookup(
            query_embedding=[0.99, 0.05, 0.0],
            tenant_id="t", principals=["public"], source_domains=None,
        )
        assert hit is not None

    async def test_lookup_below_threshold_misses(self) -> None:
        cache, _ = _make_cache(threshold=0.95)
        await cache.store(
            query_embedding=[1.0, 0.0, 0.0], response=_response(),
            tenant_id="t", principals=["public"], source_domains=None,
        )

        # An unrelated direction — cosine well under threshold.
        hit = await cache.lookup(
            query_embedding=[0.0, 1.0, 0.0],
            tenant_id="t", principals=["public"], source_domains=None,
        )
        assert hit is None

    async def test_lookup_on_empty_cache_misses(self) -> None:
        cache, _ = _make_cache()
        hit = await cache.lookup(
            query_embedding=[1.0, 0.0], tenant_id="t", principals=["public"], source_domains=None
        )
        assert hit is None

    async def test_lookup_does_not_cross_tenant_scope(self) -> None:
        """The load-bearing security property: a byte-identical query vector
        stored under one tenant must NOT be servable to a different tenant,
        even at perfect similarity — scope partitioning, not just
        similarity, gates every hit (ADR-024/ADR-026)."""
        cache, _ = _make_cache()
        vector = [1.0, 0.0, 0.0]
        await cache.store(
            query_embedding=vector, response=_response(),
            tenant_id="tenant-a", principals=["public"], source_domains=None,
        )

        hit = await cache.lookup(
            query_embedding=vector, tenant_id="tenant-b",
            principals=["public"], source_domains=None,
        )
        assert hit is None

    async def test_lookup_does_not_cross_principal_scope(self) -> None:
        """Same property, for principals: an answer cached for one caller's
        ACL scope must not leak to a caller with different principals —
        this is what actually prevents a cache hit from crossing an ACL
        boundary the retrieval layer would have enforced."""
        cache, _ = _make_cache()
        vector = [1.0, 0.0, 0.0]
        await cache.store(
            query_embedding=vector, response=_response(),
            tenant_id="t", principals=["user:alice"], source_domains=None,
        )

        hit = await cache.lookup(
            query_embedding=vector, tenant_id="t",
            principals=["user:bob"], source_domains=None,
        )
        assert hit is None

    async def test_guardrail_flagged_answers_are_never_cached(self) -> None:
        """A flagged/ungrounded answer must not be replayed to a future
        caller as if it were trustworthy — store() refuses it outright."""
        cache, fake = _make_cache()
        await cache.store(
            query_embedding=[1.0, 0.0], response=_response(flagged=True),
            tenant_id="t", principals=["public"], source_domains=None,
        )
        assert fake.store == {}

    async def test_lookup_failure_fails_open_not_raises(self) -> None:
        """Redis being unreachable must degrade to a miss, never propagate
        as an exception — the cache is a cost optimization, never a
        correctness dependency."""

        class _BrokenRedis:
            async def lrange(self, *a, **kw):
                raise ConnectionError("redis unreachable")

        cache = SemanticAnswerCache("redis://fake")
        cache._client = _BrokenRedis()  # type: ignore[assignment]

        hit = await cache.lookup(
            query_embedding=[1.0, 0.0], tenant_id="t", principals=["public"], source_domains=None
        )
        assert hit is None

    async def test_store_failure_fails_open_not_raises(self) -> None:
        class _BrokenRedis:
            def pipeline(self):
                raise ConnectionError("redis unreachable")

        cache = SemanticAnswerCache("redis://fake")
        cache._client = _BrokenRedis()  # type: ignore[assignment]

        # Must not raise.
        await cache.store(
            query_embedding=[1.0, 0.0], response=_response(),
            tenant_id="t", principals=["public"], source_domains=None,
        )
