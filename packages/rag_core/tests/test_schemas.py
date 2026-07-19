from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_core.schemas import (
    ChunkRecord,
    ContentModality,
    DocumentMetadata,
    ParentContext,
    RetrievedChunk,
    SourceType,
)


class TestSourceType:
    def test_all_expected_members_present(self) -> None:
        # Regression guard: TXT/MD were added to remove a "closest available
        # enum value" workaround in the ingestion service's suffix mapping.
        assert {m.value for m in SourceType} == {
            "pdf",
            "html",
            "docx",
            "image",
            "text",
            "markdown",
        }

    def test_constructs_from_raw_string(self) -> None:
        # Every service constructs these enums from raw strings pulled out of
        # JSON payloads (e.g. Qdrant payload, HTTP request bodies) — this must
        # keep working, which is why SourceType stays str+Enum, not StrEnum.
        assert SourceType("text") == SourceType.TEXT
        assert SourceType("markdown") == SourceType.MARKDOWN


class TestChunkRecordValidation:
    def test_empty_text_is_rejected(self, sample_metadata: DocumentMetadata) -> None:
        with pytest.raises(ValidationError, match="chunk text must not be empty"):
            ChunkRecord(
                parent_id="p-1",
                document_id="doc-1",
                text="   ",
                token_count=0,
                metadata=sample_metadata,
            )

    def test_whitespace_only_text_is_rejected(self, sample_metadata: DocumentMetadata) -> None:
        with pytest.raises(ValidationError):
            ChunkRecord(
                parent_id="p-1",
                document_id="doc-1",
                text="\n\t  ",
                token_count=0,
                metadata=sample_metadata,
            )

    def test_valid_chunk_constructs_with_defaults(self, sample_metadata: DocumentMetadata) -> None:
        chunk = ChunkRecord(
            parent_id="p-1",
            document_id="doc-1",
            text="Some real content.",
            token_count=3,
            metadata=sample_metadata,
        )
        assert chunk.chunk_id  # auto-generated UUID string, non-empty
        assert chunk.modality == ContentModality.PROSE
        assert chunk.embedding is None
        assert chunk.created_at.tzinfo is not None  # timezone-aware, not naive utcnow()

    def test_chunk_ids_are_unique_across_instances(self, sample_metadata: DocumentMetadata) -> None:
        kwargs = {"parent_id": "p-1", "document_id": "doc-1", "text": "x", "token_count": 1}
        a = ChunkRecord(**kwargs, metadata=sample_metadata)
        b = ChunkRecord(**kwargs, metadata=sample_metadata)
        assert a.chunk_id != b.chunk_id


class TestRetrievedChunkFinalScore:
    def _retrieved(
        self,
        sample_metadata: DocumentMetadata,
        *,
        dense_score: float | None = None,
        rrf_score: float | None = None,
        rerank_score: float | None = None,
    ) -> RetrievedChunk:
        chunk = ChunkRecord(
            parent_id="p-1",
            document_id="doc-1",
            text="content",
            token_count=1,
            metadata=sample_metadata,
        )
        parent = ParentContext(parent_id="p-1", document_id="doc-1", text="content")
        return RetrievedChunk(
            chunk=chunk,
            parent=parent,
            dense_score=dense_score,
            rrf_score=rrf_score,
            rerank_score=rerank_score,
        )

    def test_prefers_rerank_score_when_present(self, sample_metadata: DocumentMetadata) -> None:
        r = self._retrieved(sample_metadata, dense_score=0.1, rrf_score=0.5, rerank_score=0.9)
        assert r.final_score == 0.9

    def test_falls_back_to_rrf_score_when_no_rerank(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        r = self._retrieved(sample_metadata, dense_score=0.1, rrf_score=0.5, rerank_score=None)
        assert r.final_score == 0.5

    def test_falls_back_to_dense_score_when_only_dense_present(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        r = self._retrieved(sample_metadata, dense_score=0.3, rrf_score=None, rerank_score=None)
        assert r.final_score == 0.3

    def test_defaults_to_zero_when_no_score_present(
        self, sample_metadata: DocumentMetadata
    ) -> None:
        r = self._retrieved(sample_metadata)
        assert r.final_score == 0.0
