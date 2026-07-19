"""OpenSearch BM25 sparse search + indexing (ADR-004 hybrid half #2, ADR-020).

One OpenSearch index per source domain, mirroring Qdrant's per-domain
collection sharding (ADR-003) so the two rankers stay aligned. Tenancy
(ADR-010) is enforced as a `bool.filter` clause — filter context in OpenSearch
does not affect scoring and is *not* optional/best-effort like a `should`
clause would be, so a tenant can never see another tenant's documents by
virtue of a fuzzy/optional match.

This module lives in rag_core (not the retrieval service) because BOTH sides
of the sparse index must share one definition of the index name and mapping:
retrieval searches it, ingestion writes it (ADR-020). Keeping them in separate
services is how the original write-path gap happened — retrieval politely
skipped "not yet ingested" indices that no code path could ever create.

Requires the `rag-core[opensearch]` extra; generation/eval never import this
module and stay free of the opensearch-py dependency.

Two entry points:

- `SparseSearchClient` — long-lived search client for the retrieval service.
- `SparseIndexer` — write path for the ingestion pipeline. Opens and closes a
  fresh connection per `index_chunks` call: the Celery worker runs each task
  under its own short-lived event loop (`asyncio.run`), and aiohttp sessions
  bind to the loop they first run on, so a persistent async client would die
  with "Event loop is closed" on the second task of a worker process.
"""

from __future__ import annotations

from typing import Any

import structlog
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError
from opensearchpy.helpers import async_bulk
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_core.schemas import ChunkRecord

logger = structlog.get_logger(__name__)

_RETRYABLE = (ConnectionError, TimeoutError)


def index_name(prefix: str, source_domain: str) -> str:
    """One OpenSearch index per source domain, analogous to `collection_name`."""
    return f"{prefix}_{source_domain}"


INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "standard"},
            "modality": {"type": "keyword"},
            "tenant_id": {"type": "keyword"},
            "access_role": {"type": "keyword"},
            "source_domain": {"type": "keyword"},
            "allowed_principals": {"type": "keyword"},  # ADR-024 ACL pre-filter
        }
    },
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
        }
    },
}


async def _ensure_index(client: AsyncOpenSearch, prefix: str, source_domain: str) -> None:
    """Create the per-domain index with an explicit text mapping if absent."""
    name = index_name(prefix, source_domain)
    exists = await client.indices.exists(index=name)
    if exists:
        return
    logger.info("sparse_search.creating_index", index=name)
    await client.indices.create(index=name, body=INDEX_MAPPING)


def chunk_to_sparse_doc(chunk: ChunkRecord) -> dict[str, Any]:
    """Project a ChunkRecord onto the sparse index mapping — exactly the
    fields in `INDEX_MAPPING`, nothing more (embeddings stay in Qdrant)."""
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "document_id": chunk.document_id,
        # searchable_text, not text (ADR-023): BM25 must see the same
        # context-situated form the dense embedding was computed from, or
        # the two hybrid legs rank against different documents.
        "text": chunk.searchable_text,
        "modality": chunk.modality.value,
        "tenant_id": chunk.metadata.tenant_id,
        "access_role": chunk.metadata.access_role.value,
        "source_domain": chunk.metadata.source_domain,
        "allowed_principals": chunk.metadata.allowed_principals,  # ADR-024
    }


class SparseIndexer:
    """Bulk write path for the ingestion pipeline (ADR-020).

    A failed sparse write fails the ingestion job (after retries) rather than
    degrading silently: a document that exists dense-only would give hybrid
    search inconsistent halves, which is the exact condition this class was
    introduced to eliminate. Consistency over availability on the write path.
    """

    def __init__(self, url: str, *, index_prefix: str = "rag", verify_certs: bool = True) -> None:
        self._url = url
        self._index_prefix = index_prefix
        self._verify_certs = verify_certs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def index_chunks(self, chunks: list[ChunkRecord], *, source_domain: str) -> int:
        """Bulk-index chunks into the domain's index; returns the count written.

        Uses `chunk_id` as the OpenSearch `_id`, so re-ingesting a document
        overwrites its chunks idempotently instead of duplicating them —
        matching `VectorStore.upsert_chunks` semantics on the dense side.
        """
        if not chunks:
            return 0

        name = index_name(self._index_prefix, source_domain)
        client = AsyncOpenSearch(hosts=[self._url], verify_certs=self._verify_certs)
        try:
            await _ensure_index(client, self._index_prefix, source_domain)
            actions = [
                {"_index": name, "_id": chunk.chunk_id, "_source": chunk_to_sparse_doc(chunk)}
                for chunk in chunks
            ]
            # raise_on_error (the default) surfaces per-document failures as
            # BulkIndexError — a partial write must fail the job, not pass.
            success_count, _ = await async_bulk(client, actions)
            logger.info("sparse_search.chunks_indexed", index=name, count=success_count)
            return int(success_count)
        finally:
            await client.close()


class SparseSearchClient:
    """Thin OpenSearch wrapper: index lifecycle + tenant-scoped BM25 search."""

    def __init__(
        self,
        url: str,
        *,
        index_prefix: str = "rag",
        verify_certs: bool = True,
    ) -> None:
        self._client = AsyncOpenSearch(hosts=[url], verify_certs=verify_certs)
        self._index_prefix = index_prefix

    async def close(self) -> None:
        await self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def ensure_index(self, source_domain: str) -> None:
        """Create the per-domain index with an explicit text mapping if absent."""
        await _ensure_index(self._client, self._index_prefix, source_domain)

    @staticmethod
    def _build_query(
        query: str, tenant_id: str, principals: list[str] | None = None
    ) -> dict[str, Any]:
        """Build the OpenSearch query body with tenancy (ADR-010) and principal
        ACLs (ADR-024) as hard `bool.filter` clauses.

        `filter` context (not `should`) is what makes these hard pre-filters:
        they participate in document selection but not scoring, and —
        critically — are mandatory, unlike `should` clauses which only
        influence ranking and can be satisfied by zero matches.

        The ACL clause mirrors rag_core.vector_store.build_acl_filter exactly:
        caller-holds-any-allowed-principal, with documents indexed before the
        field existed staying tenant-visible (their pre-ACL behavior), and an
        empty principals list degrading to the "public" sentinel (fail-closed).
        """
        effective = [p for p in (principals or []) if p.strip()] or ["public"]
        acl_clause = {
            "bool": {
                "should": [
                    {"terms": {"allowed_principals": effective}},
                    {"bool": {"must_not": [{"exists": {"field": "allowed_principals"}}]}},
                ],
                "minimum_should_match": 1,
            }
        }
        return {
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": query}}}],
                    "filter": [{"term": {"tenant_id": tenant_id}}, acl_clause],
                }
            }
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def search(
        self,
        *,
        query: str,
        source_domains: list[str],
        tenant_id: str,
        top_k: int,
        principals: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 search across the given source domains, hard-filtered by
        tenant_id and principal ACLs (ADR-024).

        Returns a list of dicts with `id`, `score`, and `source` (the indexed
        document body), sorted by BM25 score descending, truncated to top_k.
        Missing indices (domain never ingested yet) are skipped, not errors.
        """
        body = self._build_query(query, tenant_id, principals)
        results: list[dict[str, Any]] = []

        for domain in source_domains:
            name = index_name(self._index_prefix, domain)
            try:
                response = await self._client.search(index=name, body=body, size=top_k)
            except NotFoundError:
                logger.debug("sparse_search.index_missing", index=name)
                continue

            for hit in response["hits"]["hits"]:
                results.append(
                    {
                        "id": hit["_id"],
                        "score": hit["_score"],
                        "source": hit["_source"],
                    }
                )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
