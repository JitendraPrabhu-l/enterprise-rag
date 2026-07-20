"""Redis-backed semantic answer cache (ADR-026).

Production RAG's highest-leverage cost lever after reranking: when an
incoming query is near-identical in embedding space to one already answered,
serve the stored answer instead of re-running retrieval + generation.
Published production workloads report 30-50% LLM-call reduction from this
alone. The whole hybrid-search + rerank + generate path is skipped on a hit.

How it works, and why each choice:

- Keyed by (tenant_id, principals, source_domains, acl_policy_version) BEFORE
  similarity is ever considered — a cached answer is only eligible for a
  caller who could have retrieved the same documents. This is the
  load-bearing correctness rule: serving a cached answer across an ACL/tenant
  boundary would leak content the second caller may not see (ADR-024). The
  scope is hashed into the Redis key namespace, so cross-scope hits are
  structurally impossible, not merely filtered out.
- The acl_policy_version (ADR-035) closes a subtler leak the scope key alone
  does not: a caller keeps the SAME principals string after an authorization
  change (e.g. a document's allowed_principals is tightened, or a group's
  membership is revoked upstream), so the scope key is unchanged and a
  now-stale cached answer — computed when the caller could still see the
  document — would keep being served. Folding a monotonically-bumped ACL
  policy version into the key means any authorization change atomically
  invalidates every answer cached under the old policy: the new version
  hashes to a fresh namespace, and the stale entries simply expire unread.
- Within a scope, candidate query embeddings are compared by cosine
  similarity; a hit requires >= `similarity_threshold` (default 0.95, the
  value production writeups converge on — high enough that paraphrases match
  but distinct questions don't).
- Fails open like the embedding cache (ADR-013): any Redis error is a miss,
  never an error surfaced to the caller. The cache is never a correctness
  dependency, only a cost/latency optimization.
- Answers carry a TTL (default 1h) so corpus updates can't serve a stale
  answer indefinitely; feedback-driven or reindex-driven invalidation can
  shorten this per deployment.

Storage layout: one Redis sorted-effort list per scope holding recent
(embedding, answer) entries, capped at `max_entries_per_scope` to bound
memory and keep the linear similarity scan cheap. This is a pragmatic
approximate-NN over a small hot set — not a second vector database — which
is the right size for "did we just answer this exact question."
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import redis.asyncio as redis
import structlog

from rag_core.schemas import GenerationResponse

logger = structlog.get_logger(__name__)

_DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour
_DEFAULT_THRESHOLD = 0.95
_DEFAULT_MAX_ENTRIES = 128


def _scope_key(
    tenant_id: str,
    principals: list[str],
    source_domains: list[str] | None,
    acl_policy_version: str,
) -> str:
    """Namespace a cache bucket by everything that governs what a caller may
    retrieve. Principals and domains are sorted so equivalent sets collapse
    to one bucket regardless of order. The acl_policy_version (ADR-035) is
    part of the scope so any authorization change bumps the namespace and
    strands every answer cached under the old policy (see module docstring)."""
    scope = json.dumps(
        {
            "t": tenant_id,
            "p": sorted(set(principals)),
            "d": sorted(set(source_domains)) if source_domains else None,
            "av": acl_policy_version,
        },
        separators=(",", ":"),
    )
    digest = hashlib.sha256(scope.encode()).hexdigest()[:24]
    return f"anscache:v1:{digest}"


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


class SemanticAnswerCache:
    """Scope-partitioned, similarity-gated answer cache over Redis."""

    def __init__(
        self,
        redis_url: str,
        *,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_entries_per_scope: int = _DEFAULT_MAX_ENTRIES,
        acl_policy_version: str = "1",
    ) -> None:
        self._client: redis.Redis = redis.from_url(  # type: ignore[no-untyped-call]
            redis_url, decode_responses=True
        )
        self._threshold = similarity_threshold
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries_per_scope
        # ADR-035: bump this (via SEMANTIC_CACHE_ACL_POLICY_VERSION) whenever an
        # authorization change lands that could invalidate previously cached
        # answers — it re-namespaces every scope, so stale entries are never
        # served again and just expire on their TTL.
        self._acl_policy_version = acl_policy_version

    async def lookup(
        self,
        *,
        query_embedding: list[float],
        tenant_id: str,
        principals: list[str],
        source_domains: list[str] | None,
    ) -> GenerationResponse | None:
        """Return a cached answer whose query is >= threshold-similar within
        this caller's scope, or None. Fails open (returns None) on any error."""
        key = _scope_key(tenant_id, principals, source_domains, self._acl_policy_version)
        try:
            # redis-py's stubs type lrange as returning `Awaitable[list] |
            # list` rather than always-awaitable — same untyped-stub gap as
            # the `from_url` call in __init__ below.
            raw_entries = await self._client.lrange(  # type: ignore[misc]
                key, 0, self._max_entries - 1
            )
        except Exception:
            logger.warning("semantic_cache.lookup_failed", exc_info=True)
            return None

        best_sim = -1.0
        best_answer: dict[str, Any] | None = None
        for raw in raw_entries:
            try:
                entry = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            sim = _cosine(query_embedding, entry.get("embedding", []))
            if sim > best_sim:
                best_sim = sim
                best_answer = entry.get("response")

        if best_answer is not None and best_sim >= self._threshold:
            try:
                response = GenerationResponse.model_validate(best_answer)
            except Exception:
                logger.warning("semantic_cache.decode_failed", exc_info=True)
                return None
            logger.info("semantic_cache.hit", similarity=round(best_sim, 4))
            return response
        return None

    async def store(
        self,
        *,
        query_embedding: list[float],
        response: GenerationResponse,
        tenant_id: str,
        principals: list[str],
        source_domains: list[str] | None,
    ) -> None:
        """Cache `response` under the caller's scope. Best-effort: any error
        is logged and dropped, never raised. Guardrail-flagged answers are
        NOT cached — a flagged/ungrounded answer must not be replayed to
        other callers as if it were trustworthy."""
        if response.guardrail_flagged:
            return
        key = _scope_key(tenant_id, principals, source_domains, self._acl_policy_version)
        entry = json.dumps(
            {"embedding": query_embedding, "response": response.model_dump(mode="json")},
            separators=(",", ":"),
        )
        try:
            pipe = self._client.pipeline()
            pipe.lpush(key, entry)
            pipe.ltrim(key, 0, self._max_entries - 1)  # keep newest N, bound the scan
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()
        except Exception:
            logger.warning("semantic_cache.store_failed", exc_info=True)

    async def close(self) -> None:
        await self._client.aclose()
