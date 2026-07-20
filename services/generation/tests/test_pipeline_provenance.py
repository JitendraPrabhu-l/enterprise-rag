"""Tests for the ADR-034 answer-provenance stamp (`_build_provenance`).

The stamp is built from the chunks an answer actually *cited* — not the whole
retrieved set — so it records the version/model behind the evidence the model
was attributed to. These test that pure assembly directly: which
embedding-model versions, document versions, and content hashes land in the
stamp given a set of citations and context chunks.
"""

from __future__ import annotations

from rag_core.schemas import (
    ChunkRecord,
    Citation,
    DocumentMetadata,
    ParentContext,
    RetrievedChunk,
    SourceType,
)

from rag_generation.pipeline import _build_provenance


def _chunk(
    *,
    parent_id: str,
    document_id: str,
    document_version: int,
    content_hash: str | None,
    embedding_model_version: str | None,
) -> RetrievedChunk:
    metadata = DocumentMetadata(
        document_id=document_id,
        source_type=SourceType.PDF,
        source_domain="d",
        last_updated_epoch=1,
        document_version=document_version,
        content_hash=content_hash,
        embedding_model_version=embedding_model_version,
    )
    parent = ParentContext(parent_id=parent_id, document_id=document_id, text="body")
    chunk = ChunkRecord(
        parent_id=parent_id,
        document_id=document_id,
        text="body",
        token_count=1,
        metadata=metadata,
    )
    return RetrievedChunk(chunk=chunk, parent=parent)


class TestBuildProvenance:
    def test_stamps_only_cited_chunks(self) -> None:
        """A retrieved-but-uncited chunk must not contribute to the stamp —
        provenance reflects the evidence the answer was attributed to."""
        cited = _chunk(
            parent_id="p1",
            document_id="doc-1",
            document_version=3,
            content_hash="hash-1",
            embedding_model_version="bge-small-v1.5",
        )
        uncited = _chunk(
            parent_id="p2",
            document_id="doc-2",
            document_version=9,
            content_hash="hash-2",
            embedding_model_version="other-model",
        )
        citations = [Citation(parent_id="p1", document_id="doc-1")]

        prov = _build_provenance(citations, [cited, uncited])

        assert prov.embedding_model_versions == ["bge-small-v1.5"]
        assert prov.content_hashes == ["hash-1"]
        assert prov.document_versions == {"doc-1": 3}

    def test_deduplicates_and_sorts_model_versions_and_hashes(self) -> None:
        a = _chunk(
            parent_id="p1",
            document_id="doc-1",
            document_version=1,
            content_hash="hash-b",
            embedding_model_version="model-y",
        )
        b = _chunk(
            parent_id="p2",
            document_id="doc-2",
            document_version=1,
            content_hash="hash-a",
            embedding_model_version="model-x",
        )
        # Same model reused by a third cited chunk — must dedupe.
        c = _chunk(
            parent_id="p3",
            document_id="doc-3",
            document_version=1,
            content_hash="hash-a",
            embedding_model_version="model-x",
        )
        citations = [
            Citation(parent_id="p1", document_id="doc-1"),
            Citation(parent_id="p2", document_id="doc-2"),
            Citation(parent_id="p3", document_id="doc-3"),
        ]

        prov = _build_provenance(citations, [a, b, c])

        assert prov.embedding_model_versions == ["model-x", "model-y"]
        assert prov.content_hashes == ["hash-a", "hash-b"]

    def test_pre_adr034_chunks_contribute_nothing_without_erroring(self) -> None:
        """A cited chunk whose version fields are None (legacy point) yields an
        empty-but-valid stamp for that document — a partially-migrated corpus
        still produces a valid provenance record."""
        legacy = _chunk(
            parent_id="p1",
            document_id="doc-1",
            document_version=1,
            content_hash=None,
            embedding_model_version=None,
        )
        citations = [Citation(parent_id="p1", document_id="doc-1")]

        prov = _build_provenance(citations, [legacy])

        assert prov.embedding_model_versions == []
        assert prov.content_hashes == []
        # document_version always present (defaults to 1), so the doc is still tracked.
        assert prov.document_versions == {"doc-1": 1}

    def test_citation_with_no_matching_chunk_is_skipped(self) -> None:
        """A citation naming a parent_id not in the context (ADR-028 catches
        this as ungrounded) must not crash provenance assembly."""
        cited = _chunk(
            parent_id="p1",
            document_id="doc-1",
            document_version=2,
            content_hash="h",
            embedding_model_version="m",
        )
        citations = [Citation(parent_id="ghost", document_id="doc-x")]

        prov = _build_provenance(citations, [cited])

        assert prov.document_versions == {}
        assert prov.embedding_model_versions == []
