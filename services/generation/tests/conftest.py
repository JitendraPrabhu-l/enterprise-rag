"""Shared fixtures for generation-service tests."""

from __future__ import annotations

import pytest
from rag_core.schemas import (
    AccessRole,
    ChunkRecord,
    ContentModality,
    DocumentMetadata,
    ParentContext,
    RetrievedChunk,
    SourceType,
)


def make_metadata(document_id: str = "doc-1") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        source_type=SourceType.PDF,
        source_domain="test-domain",
        tenant_id="tenant-a",
        access_role=AccessRole.INTERNAL,
        last_updated_epoch=1_700_000_000,
        page_count=1,
    )


def make_retrieved_chunk(
    parent_text: str,
    *,
    parent_id: str = "parent-1",
    document_id: str = "doc-1",
    page_number: int | None = 1,
    modality: ContentModality = ContentModality.PROSE,
    chunk_text: str | None = None,
    rerank_score: float = 0.9,
) -> RetrievedChunk:
    metadata = make_metadata(document_id)
    parent = ParentContext(
        parent_id=parent_id,
        document_id=document_id,
        text=parent_text,
        page_number=page_number,
        modality=modality,
    )
    chunk = ChunkRecord(
        parent_id=parent_id,
        document_id=document_id,
        text=chunk_text or parent_text[:200] or "placeholder",
        modality=modality,
        token_count=max(1, len(parent_text.split())),
        metadata=metadata,
    )
    return RetrievedChunk(chunk=chunk, parent=parent, rerank_score=rerank_score)


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return make_metadata()
