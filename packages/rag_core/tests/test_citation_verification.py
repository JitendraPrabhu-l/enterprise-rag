"""Tests for `find_ungrounded_citations` (ADR-028) — the canonical location
for this check's logic, shared by the generation service's serve-time
guardrail (re-exported via `rag_generation.guardrails`, its own tests cover
the pipeline wiring) and the eval service's eval-time metric.
"""

from __future__ import annotations

from rag_core.citation_verification import find_ungrounded_citations
from rag_core.schemas import (
    AccessRole,
    ChunkRecord,
    Citation,
    ContentModality,
    DocumentMetadata,
    ParentContext,
    RetrievedChunk,
    SourceType,
)


def _chunk(parent_id: str, document_id: str = "doc-1", text: str = "content") -> RetrievedChunk:
    metadata = DocumentMetadata(
        document_id=document_id,
        source_type=SourceType.PDF,
        source_domain="test-domain",
        tenant_id="tenant-a",
        access_role=AccessRole.INTERNAL,
        last_updated_epoch=1_700_000_000,
    )
    parent = ParentContext(parent_id=parent_id, document_id=document_id, text=text, page_number=1)
    child = ChunkRecord(
        parent_id=parent_id,
        document_id=document_id,
        text=text[:50],
        modality=ContentModality.PROSE,
        token_count=5,
        metadata=metadata,
    )
    return RetrievedChunk(chunk=child, parent=parent, rerank_score=0.9)


class TestFindUngroundedCitations:
    def test_citation_matching_a_retrieved_chunk_is_grounded(self) -> None:
        context = [_chunk("p1")]
        citations = [Citation(parent_id="p1", document_id="doc-1", page_number=1)]
        assert find_ungrounded_citations(citations, context) == []

    def test_citation_naming_a_never_retrieved_parent_id_is_ungrounded(self) -> None:
        context = [_chunk("p1")]
        citations = [Citation(parent_id="fabricated", document_id="doc-1", page_number=1)]

        ungrounded = find_ungrounded_citations(citations, context)

        assert [c.parent_id for c in ungrounded] == ["fabricated"]

    def test_empty_citations_is_trivially_grounded(self) -> None:
        assert find_ungrounded_citations([], [_chunk("p1")]) == []

    def test_empty_context_makes_every_citation_ungrounded(self) -> None:
        citations = [Citation(parent_id="p1", document_id="doc-1", page_number=1)]
        assert find_ungrounded_citations(citations, []) == citations

    def test_mixed_grounded_and_ungrounded_returns_only_the_bad_ones(self) -> None:
        context = [_chunk("p1"), _chunk("p2")]
        citations = [
            Citation(parent_id="p1", document_id="doc-1", page_number=1),
            Citation(parent_id="fabricated", document_id="doc-1", page_number=2),
            Citation(parent_id="p2", document_id="doc-1", page_number=3),
        ]

        ungrounded = find_ungrounded_citations(citations, context)

        assert [c.parent_id for c in ungrounded] == ["fabricated"]
