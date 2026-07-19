"""Regression coverage for the ADR-023 wiring in `IngestionPipeline._embed_chunks`:
embedding MUST run on `searchable_text` (context-prefixed when contextual
enrichment ran), never on raw `.text` — otherwise the dense vector represents
different content than what BM25 indexes, silently breaking hybrid search's
premise that both legs search the same document.

Uses a spy embedder (records exactly what text list it was called with)
rather than mocking the whole embedding path, so the assertion is on the
real contract `_embed_chunks` must honor.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rag_core.schemas import AccessRole, ChunkRecord, ContentModality, DocumentMetadata, SourceType

from rag_ingestion.config import IngestionSettings
from rag_ingestion.page_classifier import HeuristicPageClassifier
from rag_ingestion.pipeline import IngestionPipeline


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        source_type=SourceType.PDF,
        source_domain="test-domain",
        tenant_id="tenant-a",
        access_role=AccessRole.INTERNAL,
        last_updated_epoch=1_700_000_000,
    )


def _chunk(chunk_id: str, *, text: str, context_prefix: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        parent_id="doc-1:p0",
        document_id="doc-1",
        text=text,
        context_prefix=context_prefix,
        modality=ContentModality.PROSE,
        token_count=5,
        metadata=_metadata(),
    )


class _SpyEmbedder:
    """Records every text list it's asked to embed; returns one fixed-length
    zero vector per input so downstream shape assumptions still hold."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[0.0, 0.0, 0.0] for _ in texts]


def _pipeline(embedder: _SpyEmbedder) -> IngestionPipeline:
    return IngestionPipeline(
        settings=IngestionSettings(),
        page_classifier=HeuristicPageClassifier(),
        vision_describer=MagicMock(),
        embedder=embedder,
        vector_store=MagicMock(),
        sparse_indexer=MagicMock(),
        embedding_cache=None,
    )


@pytest.mark.asyncio
class TestEmbedChunksUsesSearchableText:
    async def test_enriched_chunk_embeds_prefix_plus_text_not_raw_text_alone(self) -> None:
        embedder = _SpyEmbedder()
        pipeline = _pipeline(embedder)
        chunk = _chunk(
            "c-1", text="Cash reserves grew 12%.",
            context_prefix="From Acme's Q3 filing on liquidity.",
        )

        await pipeline._embed_chunks([chunk])

        assert embedder.embed_calls == [
            ["From Acme's Q3 filing on liquidity.\nCash reserves grew 12%."]
        ]

    async def test_unenriched_chunk_embeds_raw_text_unchanged(self) -> None:
        """Contextual enrichment disabled/failed (context_prefix=None) must
        produce byte-for-byte the pre-ADR-023 embedding input."""
        embedder = _SpyEmbedder()
        pipeline = _pipeline(embedder)
        chunk = _chunk("c-1", text="Cash reserves grew 12%.", context_prefix=None)

        await pipeline._embed_chunks([chunk])

        assert embedder.embed_calls == [["Cash reserves grew 12%."]]

    async def test_returned_chunk_keeps_its_context_prefix_after_embedding(self) -> None:
        """The prefix must survive into the ChunkRecord that gets indexed —
        it's what SparseIndexer.index_chunks and VectorStore.upsert_chunks
        both need to write into their respective stores."""
        embedder = _SpyEmbedder()
        pipeline = _pipeline(embedder)
        chunk = _chunk("c-1", text="Cash reserves grew 12%.", context_prefix="Situating sentence.")

        [result] = await pipeline._embed_chunks([chunk])

        assert result.context_prefix == "Situating sentence."
        assert result.embedding == [0.0, 0.0, 0.0]

    async def test_mixed_batch_each_chunk_embeds_its_own_searchable_text(self) -> None:
        """One document's chunks can have a mix of enriched and
        fallback-to-raw entries (per-chunk enrichment failures) — each must
        independently embed the correct form."""
        embedder = _SpyEmbedder()
        pipeline = _pipeline(embedder)
        chunks = [
            _chunk("c-1", text="alpha", context_prefix="ctx-a"),
            _chunk("c-2", text="beta", context_prefix=None),
        ]

        await pipeline._embed_chunks(chunks)

        assert embedder.embed_calls == [["ctx-a\nalpha", "beta"]]
