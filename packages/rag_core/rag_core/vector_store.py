"""Thin Qdrant wrapper implementing ADR-003 (sharded collections, quantization)
and the hard pre-filter from ADR-010 (tenancy scoping before vector search).
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient, models
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_core.schemas import ChunkRecord, ParentContext

QUANTIZATION_THRESHOLD_VECTORS = 50_000_000


def collection_name(source_domain: str) -> str:
    """One Qdrant collection per source domain (ADR-003 sharding)."""
    return f"rag_{source_domain}"


def build_acl_filter(
    *, tenant_id: str, principals: list[str] | None, include_superseded: bool = False
) -> models.Filter:
    """The hard pre-filter every dense search runs under (ADR-010 + ADR-024 + ADR-034).

    Fail-closed by construction: an empty/None principals list degrades to
    the "public" sentinel — the caller sees only un-ACL'd documents, never
    everything. MatchAny gives OR-semantics over the caller's groups, which
    intersected with each chunk's allowed_principals array is exactly the
    "does the caller hold ANY principal this document allows" check the
    document-level RBAC pattern prescribes.

    Currency (ADR-034): unless `include_superseded`, superseded documents are
    excluded here — before HNSW distance is ever computed — so a stale-but-
    similar version can never out-rank the version that replaced it. Same
    backward-compat shape as the ACL branch: points written before ADR-034
    carry no is_current field and stay visible (their pre-versioning
    behavior), so this can never hide a document nobody ever marked stale.
    """
    effective = [p for p in (principals or []) if p.strip()] or ["public"]
    must: list[Any] = [
        models.FieldCondition(
            key="metadata.tenant_id", match=models.MatchValue(value=tenant_id)
        ),
        # Points indexed before ADR-024 carry no allowed_principals field
        # at all — those stay visible tenant-wide (exactly their pre-ACL
        # behavior) rather than vanishing behind a filter they never
        # opted into. Points written since always have the field, so the
        # IsEmpty branch can never weaken an ACL someone actually set.
        models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.allowed_principals",
                    match=models.MatchAny(any=effective),
                ),
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key="metadata.allowed_principals")
                ),
            ]
        ),
    ]
    if not include_superseded:
        # Keep a document if it is explicitly current OR predates the
        # is_current field entirely (IsEmpty). Only a document a writer
        # actively marked is_current=false is filtered out — never a legacy
        # point that never had the field.
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.is_current", match=models.MatchValue(value=True)
                    ),
                    models.IsEmptyCondition(
                        is_empty=models.PayloadField(key="metadata.is_current")
                    ),
                ]
            )
        )
    return models.Filter(must=must)


class VectorStore:
    def __init__(self, url: str, api_key: str | None, embedding_dim: int) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._embedding_dim = embedding_dim

    async def ensure_collection(self, source_domain: str, *, quantize: bool = False) -> None:
        name = collection_name(source_domain)
        if await self._client.collection_exists(name):
            return
        quantization_config = (
            models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            )
            if quantize
            else None
        )
        await self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self._embedding_dim, distance=models.Distance.COSINE
            ),
            quantization_config=quantization_config,
        )
        for field in (
            "document_id",
            "tenant_id",
            "access_role",
            "source_domain",
            "allowed_principals",  # ADR-024 ACL pre-filter
        ):
            await self._client.create_payload_index(
                collection_name=name,
                field_name=f"metadata.{field}",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        # ADR-034 currency pre-filter: bool index so is_current filtering is a
        # cheap payload lookup, not a full scan.
        await self._client.create_payload_index(
            collection_name=name,
            field_name="metadata.is_current",
            field_schema=models.PayloadSchemaType.BOOL,
        )

    async def enable_quantization_if_due(self, source_domain: str) -> bool:
        """Turn on scalar INT8 quantization for a domain that has grown past
        `QUANTIZATION_THRESHOLD_VECTORS` (ADR-003) and isn't quantized yet.

        Qdrant fixes a collection's quantization config at creation time —
        `ensure_collection` only ever runs once per domain, at its FIRST
        ingest, when the domain has by definition zero prior vectors, so
        there is no size to self-decide from at that moment (a hardcoded
        "0 vectors so far" check there would be quantization config that
        can never actually fire — exactly the dead-scaffolding problem this
        method exists to fix). Real vector counts only exist for a domain
        AFTER points have been upserted, so this is a separate, idempotent,
        call-anytime operation: check the domain's real point count via
        Qdrant's own collection info, and if it has crossed the threshold
        without quantization already configured, apply it in place via
        `update_collection` (Qdrant supports enabling quantization on an
        existing collection without recreating it or its indexed points).

        Intended to be called periodically (e.g. once per ingestion job, or
        from a scheduled sweep) rather than on every single upsert — the
        collection-info lookup is one extra API call, cheap relative to an
        ingestion job but wasteful to repeat per-chunk.

        Returns True if quantization was newly enabled this call, False if
        the domain doesn't exist yet, is under threshold, or was already
        quantized — all three are legitimate no-ops, not errors.
        """
        name = collection_name(source_domain)
        if not await self._client.collection_exists(name):
            return False

        info = await self._client.get_collection(name)
        already_quantized = info.config.quantization_config is not None
        if already_quantized:
            return False

        vector_count = info.points_count or 0
        if vector_count < QUANTIZATION_THRESHOLD_VECTORS:
            return False

        await self._client.update_collection(
            collection_name=name,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            ),
        )
        return True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
    async def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        parents: dict[str, ParentContext] | None = None,
    ) -> None:
        """Upsert child chunks, embedding the full parent passage in each point's payload.

        `parents` maps `parent_id -> ParentContext`. Storing the parent text alongside
        the child vector (rather than in a separate store) means a single Qdrant query
        returns everything `search()` needs to reconstruct both `ChunkRecord` and
        `ParentContext` (ADR-002) without a second round-trip to a parent datastore.
        """
        if not chunks:
            return
        parents = parents or {}
        by_domain: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            by_domain.setdefault(chunk.metadata.source_domain, []).append(chunk)

        for domain, domain_chunks in by_domain.items():
            await self.ensure_collection(domain)
            points = []
            for c in domain_chunks:
                if c.embedding is None:
                    continue
                parent = parents.get(c.parent_id)
                points.append(
                    models.PointStruct(
                        id=c.chunk_id,
                        vector=c.embedding,
                        payload={
                            "parent_id": c.parent_id,
                            "document_id": c.document_id,
                            "text": c.text,
                            "context_prefix": c.context_prefix,
                            "modality": c.modality.value,
                            "metadata": c.metadata.model_dump(mode="json"),
                            "parent": parent.model_dump(mode="json") if parent else None,
                        },
                    )
                )
            await self._client.upsert(collection_name=collection_name(domain), points=points)

    async def search(
        self,
        *,
        query_vector: list[float],
        source_domains: list[str],
        tenant_id: str,
        top_k: int,
        principals: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Hard pre-filter on tenancy (ADR-010), principal ACLs (ADR-024), and
        currency (ADR-034), all evaluated before HNSW distance search — a
        chunk the caller may not see, or a superseded version, is never even
        scored, so no post-filter bug can leak it or let it out-rank current
        content."""
        must_filter = build_acl_filter(
            tenant_id=tenant_id, principals=principals, include_superseded=include_superseded
        )
        results: list[dict[str, Any]] = []
        for domain in source_domains:
            name = collection_name(domain)
            if not await self._client.collection_exists(name):
                continue
            hits = await self._client.query_points(
                collection_name=name,
                query=query_vector,
                query_filter=must_filter,
                limit=top_k,
                with_payload=True,
            )
            for hit in hits.points:
                results.append({"score": hit.score, "payload": hit.payload, "id": hit.id})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
