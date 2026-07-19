"""Tests for `ColPaliPageIndex` (ADR-029). `ColPaliEmbedder` (the real
model wrapper) is never constructed here — it would load multi-GB model
weights — tests use a `PageImageEmbedder`-shaped fake, exercising only
`ColPaliPageIndex`'s own storage/query logic against a mocked
`AsyncQdrantClient`. `ColPaliEmbedder` itself has no meaningful logic to
unit test beyond "call the real library correctly," which is exactly the
kind of thing this repo verifies live (like contextual_enrichment's utility
model, or the semantic cache's embedding model) rather than mocking a
third-party ML library's internals.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models

from rag_ingestion.colpali_index import ColPaliPageIndex, visual_collection_name


class _FakePageImageEmbedder:
    def __init__(self, patch_dimension: int = 4, patches_per_page: int = 3) -> None:
        self._patch_dimension = patch_dimension
        self._patches_per_page = patches_per_page
        self.embed_calls: list[bytes] = []

    @property
    def patch_dimension(self) -> int:
        return self._patch_dimension

    async def embed_page(self, image_bytes: bytes) -> list[list[float]]:
        self.embed_calls.append(image_bytes)
        return [[0.1 * i] * self._patch_dimension for i in range(self._patches_per_page)]


class TestVisualCollectionName:
    def test_uses_a_separate_namespace_from_the_dense_collection(self) -> None:
        """Must never collide with VectorStore's `rag_{domain}` collections
        - the two have incompatible vector configs (single dense vector vs.
        named multi-vector), so sharing a name would be a hard Qdrant
        schema conflict, not just a logical mixup."""
        assert visual_collection_name("sec-filings") == "rag_visual_sec-filings"
        assert visual_collection_name("sec-filings") != "rag_sec-filings"


@pytest.mark.asyncio
class TestIndexPage:
    async def test_successful_index_creates_collection_and_upserts(self) -> None:
        embedder = _FakePageImageEmbedder(patch_dimension=4, patches_per_page=3)
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=False)
        index._client.create_collection = AsyncMock()
        index._client.upsert = AsyncMock()

        result = await index.index_page(
            source_domain="sec-filings",
            document_id="doc-1",
            page_number=3,
            image_bytes=b"fake-png-bytes",
        )

        assert result is True
        index._client.create_collection.assert_awaited_once()
        _, create_kwargs = index._client.create_collection.await_args
        assert create_kwargs["collection_name"] == "rag_visual_sec-filings"

        index._client.upsert.assert_awaited_once()
        _, upsert_kwargs = index._client.upsert.await_args
        assert upsert_kwargs["collection_name"] == "rag_visual_sec-filings"
        [point] = upsert_kwargs["points"]
        assert point.id == "doc-1:p3"
        assert point.payload == {"document_id": "doc-1", "page_number": 3}

    async def test_existing_collection_is_not_recreated(self) -> None:
        embedder = _FakePageImageEmbedder()
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=True)
        index._client.create_collection = AsyncMock()
        index._client.upsert = AsyncMock()

        await index.index_page(
            source_domain="sec-filings", document_id="doc-1", page_number=1, image_bytes=b"x"
        )

        index._client.create_collection.assert_not_awaited()

    async def test_embedder_failure_returns_false_not_raises(self) -> None:
        """ADR-029's core reliability property, mirroring ADR-023's
        contextual enrichment: a per-page ColPali failure must be
        swallowed, never propagate and fail the document's ingest job -
        this is an additional retrieval signal, not a correctness
        dependency."""
        embedder = MagicMock()
        embedder.patch_dimension = 4
        embedder.embed_page = AsyncMock(side_effect=RuntimeError("model inference failed"))
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)

        result = await index.index_page(
            source_domain="sec-filings", document_id="doc-1", page_number=1, image_bytes=b"x"
        )

        assert result is False

    async def test_qdrant_upsert_failure_returns_false_not_raises(self) -> None:
        embedder = _FakePageImageEmbedder()
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=True)
        index._client.upsert = AsyncMock(side_effect=ConnectionError("qdrant unreachable"))

        result = await index.index_page(
            source_domain="sec-filings", document_id="doc-1", page_number=1, image_bytes=b"x"
        )

        assert result is False

    async def test_collection_uses_multivector_maxsim_config(self) -> None:
        embedder = _FakePageImageEmbedder(patch_dimension=128)
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=False)
        index._client.create_collection = AsyncMock()
        index._client.upsert = AsyncMock()

        await index.index_page(
            source_domain="sec-filings", document_id="doc-1", page_number=1, image_bytes=b"x"
        )

        _, kwargs = index._client.create_collection.await_args
        vector_params = kwargs["vectors_config"]["colpali_patches"]
        assert vector_params.size == 128
        assert vector_params.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM


@pytest.mark.asyncio
class TestSearch:
    async def test_missing_collection_returns_empty_list_not_error(self) -> None:
        """Mirrors VectorStore.search's handling of a never-ingested domain
        - a domain with no visual index yet is a normal, expected state,
        not an error condition."""
        embedder = _FakePageImageEmbedder()
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=False)

        results = await index.search(
            query_vectors=[[0.1, 0.2]], source_domain="never-ingested", top_k=5
        )

        assert results == []

    async def test_search_queries_the_named_multivector_field(self) -> None:
        embedder = _FakePageImageEmbedder()
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=True)

        fake_hit = MagicMock()
        fake_hit.score = 0.87
        fake_hit.payload = {"document_id": "doc-1", "page_number": 4}
        fake_response = MagicMock()
        fake_response.points = [fake_hit]
        index._client.query_points = AsyncMock(return_value=fake_response)

        results = await index.search(
            query_vectors=[[0.1, 0.2], [0.3, 0.4]], source_domain="sec-filings", top_k=5
        )

        assert results == [{"score": 0.87, "document_id": "doc-1", "page_number": 4}]
        _, kwargs = index._client.query_points.await_args
        assert kwargs["using"] == "colpali_patches"
        assert kwargs["query"] == [[0.1, 0.2], [0.3, 0.4]]
        assert kwargs["collection_name"] == "rag_visual_sec-filings"

    async def test_a_hit_with_no_payload_is_skipped_not_a_crash(self) -> None:
        """Qdrant types a point's payload as `dict | None` - every point
        this class writes always has one, but a point written some other
        way (or corrupted) could not. Must skip that point, not raise
        TypeError, and must not affect other hits in the same response."""
        embedder = _FakePageImageEmbedder()
        index = ColPaliPageIndex(url="http://qdrant:6333", api_key=None, embedder=embedder)
        index._client.collection_exists = AsyncMock(return_value=True)

        no_payload_hit = MagicMock()
        no_payload_hit.score = 0.5
        no_payload_hit.payload = None
        no_payload_hit.id = "malformed-point"

        good_hit = MagicMock()
        good_hit.score = 0.9
        good_hit.payload = {"document_id": "doc-1", "page_number": 1}

        fake_response = MagicMock()
        fake_response.points = [no_payload_hit, good_hit]
        index._client.query_points = AsyncMock(return_value=fake_response)

        results = await index.search(
            query_vectors=[[0.1, 0.2]], source_domain="sec-filings", top_k=5
        )

        assert results == [{"score": 0.9, "document_id": "doc-1", "page_number": 1}]
