"""Tests for `find_ungrounded_citations` (ADR-028): the citation-identifier
verification guardrail distinct from BOTH the uncited-answer check (an
answer with zero citations) and the RAG Triad's faithfulness judge (which
scores answer TEXT, never verifies citation identifiers against what was
actually retrieved). A citation naming a parent_id absent from the context
shown to the model is schema-valid, present, and still wrong — this is the
one check that catches exactly that.
"""

from __future__ import annotations

from rag_core.schemas import Citation

from rag_generation.guardrails import find_ungrounded_citations
from tests.conftest import make_retrieved_chunk


class TestFindUngroundedCitations:
    def test_citation_matching_a_retrieved_chunk_is_grounded(self) -> None:
        context = [make_retrieved_chunk("Revenue grew 12%.", parent_id="p1", document_id="d1")]
        citations = [Citation(parent_id="p1", document_id="d1", page_number=1)]

        assert find_ungrounded_citations(citations, context) == []

    def test_citation_naming_a_never_retrieved_parent_id_is_ungrounded(self) -> None:
        """The core case: a citation that looks completely valid (well-formed
        parent_id, document_id, page_number) but names content that was
        never in the context the model actually saw."""
        context = [make_retrieved_chunk("Revenue grew 12%.", parent_id="p1", document_id="d1")]
        citations = [Citation(parent_id="p999-fabricated", document_id="d1", page_number=1)]

        ungrounded = find_ungrounded_citations(citations, context)

        assert len(ungrounded) == 1
        assert ungrounded[0].parent_id == "p999-fabricated"

    def test_empty_citations_list_is_trivially_grounded(self) -> None:
        context = [make_retrieved_chunk("Revenue grew 12%.", parent_id="p1")]
        assert find_ungrounded_citations([], context) == []

    def test_empty_context_makes_every_citation_ungrounded(self) -> None:
        """If nothing was retrieved (e.g. guardrail-filtered every chunk),
        any citation at all is by definition ungrounded."""
        citations = [Citation(parent_id="p1", document_id="d1", page_number=1)]
        assert find_ungrounded_citations(citations, []) == citations

    def test_mixed_grounded_and_ungrounded_citations_returns_only_the_bad_ones(self) -> None:
        context = [
            make_retrieved_chunk("Revenue grew 12%.", parent_id="p1", document_id="d1"),
            make_retrieved_chunk("Headcount reached 240.", parent_id="p2", document_id="d1"),
        ]
        citations = [
            Citation(parent_id="p1", document_id="d1", page_number=1),
            Citation(parent_id="p999-fabricated", document_id="d1", page_number=2),
            Citation(parent_id="p2", document_id="d1", page_number=3),
        ]

        ungrounded = find_ungrounded_citations(citations, context)

        assert len(ungrounded) == 1
        assert ungrounded[0].parent_id == "p999-fabricated"

    def test_multiple_citations_to_the_same_grounded_parent_id_are_all_grounded(self) -> None:
        """A model citing the same passage twice (e.g. for two different
        claims) must not be flagged - repetition isn't the failure mode
        under test, a wrong identifier is."""
        context = [make_retrieved_chunk("Revenue grew 12%.", parent_id="p1", document_id="d1")]
        citations = [
            Citation(parent_id="p1", document_id="d1", page_number=1),
            Citation(parent_id="p1", document_id="d1", page_number=1),
        ]

        assert find_ungrounded_citations(citations, context) == []

    def test_matches_on_parent_id_alone_not_document_id(self) -> None:
        """A citation with the CORRECT parent_id but a mismatched
        document_id is not what this check catches (parent_id is the
        unique join key into the retrieved set; a document_id/parent_id
        mismatch on an otherwise-valid parent_id is a different kind of
        model error, and parent_id is globally unique in practice since it
        is generated including the document_id as a prefix)."""
        context = [make_retrieved_chunk("Revenue grew 12%.", parent_id="p1", document_id="d1")]
        citations = [Citation(parent_id="p1", document_id="wrong-doc", page_number=1)]

        assert find_ungrounded_citations(citations, context) == []
