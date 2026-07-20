from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from rag_core.schemas import ChunkRecord, DocumentMetadata, ParentContext
from rag_core.vector_store import VectorStore, collection_name


def _chunk(
    chunk_id: str, parent_id: str, metadata: DocumentMetadata, *, embedding: list[float] | None
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        parent_id=parent_id,
        document_id=metadata.document_id,
        text="child chunk text",
        token_count=3,
        embedding=embedding,
        metadata=metadata,
    )


class TestCollectionName:
    def test_prefixes_with_rag(self) -> None:
        assert collection_name("sec-filings") == "rag_sec-filings"


@pytest.mark.asyncio
class TestUpsertChunksPersistsParentContext:
    """Regression coverage for the parent-passage persistence fix: earlier,
    `upsert_chunks` only stored the child chunk's own text, so retrieval could
    never recover the true ~1024-token parent passage (ADR-002). It now
    accepts a `parent_id -> ParentContext` map and embeds each parent's full
    text in the corresponding point's payload.
    """

    async def test_upserted_payload_includes_full_parent_context(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)

        parent = ParentContext(
            parent_id="parent-1",
            document_id=sample_metadata.document_id,
            text="This is the full ~1024-token parent passage, not just the child sliver.",
            page_number=2,
        )
        chunk = _chunk("child-1", "parent-1", sample_metadata, embedding=[0.1, 0.2, 0.3])

        with (
            patch.object(store, "ensure_collection", new=AsyncMock()),
            patch.object(store._client, "upsert", new=AsyncMock()) as mock_upsert,
        ):
            await store.upsert_chunks([chunk], parents={"parent-1": parent})

        mock_upsert.assert_awaited_once()
        _, kwargs = mock_upsert.call_args
        points = kwargs["points"]
        assert len(points) == 1

        payload = points[0].payload
        assert payload["parent_id"] == "parent-1"
        assert payload["parent"] is not None
        assert payload["parent"]["text"] == parent.text
        assert payload["parent"]["page_number"] == 2

    async def test_missing_parent_mapping_persists_none_not_a_crash(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        # A chunk whose parent_id has no entry in `parents` (e.g. caller forgot
        # to pass it, or partial data) must not raise — the payload's "parent"
        # key is simply None, and downstream readers fall back to child text.
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        chunk = _chunk("child-1", "parent-missing", sample_metadata, embedding=[0.1, 0.2, 0.3])

        with (
            patch.object(store, "ensure_collection", new=AsyncMock()),
            patch.object(store._client, "upsert", new=AsyncMock()) as mock_upsert,
        ):
            await store.upsert_chunks([chunk])  # no `parents` argument at all

        _, kwargs = mock_upsert.call_args
        assert kwargs["points"][0].payload["parent"] is None

    async def test_chunks_without_embedding_are_skipped(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        embedded = _chunk("child-1", "parent-1", sample_metadata, embedding=[0.1, 0.2, 0.3])
        unembedded = _chunk("child-2", "parent-1", sample_metadata, embedding=None)

        with (
            patch.object(store, "ensure_collection", new=AsyncMock()),
            patch.object(store._client, "upsert", new=AsyncMock()) as mock_upsert,
        ):
            await store.upsert_chunks([embedded, unembedded])

        _, kwargs = mock_upsert.call_args
        point_ids = [p.id for p in kwargs["points"]]
        assert point_ids == ["child-1"]

    async def test_empty_chunk_list_is_a_noop(self) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        with patch.object(store._client, "upsert", new=AsyncMock()) as mock_upsert:
            await store.upsert_chunks([])
        mock_upsert.assert_not_awaited()

    async def test_chunks_from_different_domains_route_to_different_collections(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        other_metadata = sample_metadata.model_copy(update={"source_domain": "other-domain"})
        chunk_a = _chunk("child-a", "parent-a", sample_metadata, embedding=[0.1, 0.2, 0.3])
        chunk_b = _chunk("child-b", "parent-b", other_metadata, embedding=[0.4, 0.5, 0.6])

        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        with (
            patch.object(store, "ensure_collection", new=AsyncMock()),
            patch.object(store._client, "upsert", new=AsyncMock()) as mock_upsert,
        ):
            await store.upsert_chunks([chunk_a, chunk_b])

        called_collections = {
            call.kwargs["collection_name"] for call in mock_upsert.await_args_list
        }
        assert called_collections == {
            collection_name(sample_metadata.source_domain),
            collection_name("other-domain"),
        }


@pytest.mark.asyncio
class TestSearchAppliesTenantHardFilter:
    async def test_search_applies_tenant_filter_and_skips_missing_collections(self) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)

        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=False)),
            patch.object(store._client, "query_points", new=AsyncMock()) as mock_query,
        ):
            results = await store.search(
                query_vector=[0.1, 0.2, 0.3],
                source_domains=["missing-domain"],
                tenant_id="tenant-a",
                top_k=10,
            )

        assert results == []
        mock_query.assert_not_awaited()

    async def test_search_passes_tenant_filter_to_query_points(self) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)

        fake_hit = type("Hit", (), {"score": 0.9, "payload": {"parent_id": "p1"}, "id": "c1"})()
        fake_response = type("Resp", (), {"points": [fake_hit]})()

        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=True)),
            patch.object(
                store._client, "query_points", new=AsyncMock(return_value=fake_response)
            ) as mock_query,
        ):
            results = await store.search(
                query_vector=[0.1, 0.2, 0.3],
                source_domains=["demo-corpus"],
                tenant_id="tenant-a",
                top_k=10,
            )

        assert len(results) == 1
        assert results[0]["id"] == "c1"

        _, kwargs = mock_query.call_args
        query_filter: Any = kwargs["query_filter"]
        # Three hard `must` (AND) conditions now (ADR-010 tenant + ADR-024 ACL
        # + ADR-034 currency): a `should` for any would let out-of-scope or
        # superseded docs rank in, just lower, rather than excluding them.
        # First is tenant scoping.
        assert len(query_filter.must) == 3
        tenant_condition = query_filter.must[0]
        assert tenant_condition.key == "metadata.tenant_id"
        assert tenant_condition.match.value == "tenant-a"
        # Second is the principal-ACL clause: a should-group of
        # (holds-an-allowed-principal OR no-ACL-field), which with an empty
        # caller principals list degrades to the "public" sentinel.
        acl_group = query_filter.must[1]
        principal_match = acl_group.should[0]
        assert principal_match.key == "metadata.allowed_principals"
        assert principal_match.match.any == ["public"]
        # Third is the ADR-034 currency clause: is_current=true OR field absent.
        currency_group = query_filter.must[2]
        assert currency_group.should[0].key == "metadata.is_current"
        assert currency_group.should[0].match.value is True
