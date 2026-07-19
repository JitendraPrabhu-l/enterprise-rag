from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_core.schemas import ChunkRecord, ContentModality, DocumentMetadata

from rag_ingestion.chunking import (
    _approx_token_count,
    _pack_children,
    _Sentence,
    chunk_document,
    split_sentences,
)


def _make_long_document(
    num_sentences: int, words_per_sentence: int = 10, sentences_per_topic: int = 6
) -> str:
    """Builds prose grouped into topic blocks: sentences within a block share a
    common vocabulary (so the fake bag-of-words embedder sees them as similar),
    while blocks use disjoint vocabulary (so a real semantic boundary exists at
    each block transition). This lets tests exercise multi-sentence parents
    *and* semantic-boundary splitting, unlike fully-disjoint-per-sentence text
    which would force a boundary after every single sentence."""
    sentences = []
    for i in range(num_sentences):
        topic = i // sentences_per_topic
        words = " ".join(f"topic{topic}word{j}" for j in range(words_per_sentence))
        sentences.append(f"{words.capitalize()}.")
    return " ".join(sentences)


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self) -> None:
        text = "This is one sentence. This is another! Is this a third? Yes it is."
        sentences = split_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "This is one sentence."

    def test_empty_text_returns_empty_list(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   \n\t  ") == []

    def test_no_boundary_returns_whole_text_as_one_sentence(self) -> None:
        text = "just some words with no terminal punctuation"
        assert split_sentences(text) == [text]

    def test_collapses_internal_whitespace(self) -> None:
        text = "Line one.\n\n   Line two continues here."
        sentences = split_sentences(text)
        assert sentences[0] == "Line one."
        assert "  " not in sentences[1]


class TestApproxTokenCount:
    def test_counts_words(self) -> None:
        assert _approx_token_count("one two three") == 3

    def test_empty_string_is_zero(self) -> None:
        assert _approx_token_count("") == 0


class TestPackChildren:
    def _sentence(self, text: str, tokens: int) -> _Sentence:
        return _Sentence(text=text, page_number=1, token_count=tokens)

    def test_single_oversized_sentence_taken_alone(self) -> None:
        sentences = [self._sentence("huge", 500)]
        children = _pack_children(sentences, child_chunk_tokens=128, overlap_ratio=0.15)
        assert len(children) == 1
        assert children[0] == sentences

    def test_no_sentences_returns_empty(self) -> None:
        assert _pack_children([], child_chunk_tokens=128, overlap_ratio=0.15) == []

    def test_produces_overlap_between_consecutive_children(self) -> None:
        # 20 sentences of 10 tokens each = 200 tokens; child budget 50 tokens
        # forces multiple children, each should share trailing sentences with
        # the next as overlap.
        sentences = [self._sentence(f"s{i}", 10) for i in range(20)]
        children = _pack_children(sentences, child_chunk_tokens=50, overlap_ratio=0.2)
        assert len(children) > 1
        for prev, nxt in zip(children, children[1:], strict=False):
            overlap = {id(s) for s in prev} & {id(s) for s in nxt}
            assert overlap, "expected shared sentences between consecutive children"

    def test_all_sentences_are_covered(self) -> None:
        sentences = [self._sentence(f"s{i}", 10) for i in range(20)]
        children = _pack_children(sentences, child_chunk_tokens=50, overlap_ratio=0.2)
        covered_texts = {s.text for group in children for s in group}
        assert covered_texts == {s.text for s in sentences}

    def test_terminates_when_overlap_would_consume_entire_child(self) -> None:
        # Regression test: many short sentences (as in table rows / itemized
        # lists) can pack into a child small enough that the *entire* child
        # falls within the overlap token budget. Before the len(current) - 1
        # cap, this froze `start` in place forever — same window re-packed
        # every iteration, `children` grew unbounded until MemoryError. This
        # reproduced deterministically on a real 110-page SEC filing.
        sentences = [self._sentence(f"s{i}", 4) for i in range(3)]
        children = _pack_children(sentences, child_chunk_tokens=10, overlap_ratio=0.9)
        covered_texts = {s.text for group in children for s in group}
        assert covered_texts == {s.text for s in sentences}

    def test_terminates_across_short_sentence_configs(self) -> None:
        # Broader sweep of the same failure family: short, uniform-length
        # sentences with high overlap ratios relative to the child budget.
        # Each call must return (not hang) and cover every input sentence.
        for num_sentences in (3, 5, 8, 15):
            for tokens in (1, 2, 4, 6):
                for budget in (5, 10, 20, 50):
                    for ratio in (0.5, 0.9, 0.99, 1.0):
                        sentences = [self._sentence(f"s{i}", tokens) for i in range(num_sentences)]
                        children = _pack_children(
                            sentences, child_chunk_tokens=budget, overlap_ratio=ratio
                        )
                        covered = {s.text for group in children for s in group}
                        assert covered == {s.text for s in sentences}


class TestChunkDocument:
    def test_empty_pages_produce_no_output(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        parents, chunks = chunk_document(
            document_id="doc-1",
            page_texts=[(1, ""), (2, "   ")],
            metadata=sample_metadata,
            embedder=fake_embedder,
        )
        assert parents == []
        assert chunks == []

    def test_child_chunks_reference_valid_parent_ids(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        text = _make_long_document(num_sentences=40, words_per_sentence=15)
        parents, chunks = chunk_document(
            document_id="doc-1",
            page_texts=[(1, text)],
            metadata=sample_metadata,
            embedder=fake_embedder,
            parent_chunk_tokens=200,
            child_chunk_tokens=40,
            chunk_overlap_ratio=0.15,
        )
        assert parents, "expected at least one parent"
        assert chunks, "expected at least one child chunk"

        parent_ids = {p.parent_id for p in parents}
        for chunk in chunks:
            assert chunk.parent_id in parent_ids
            assert chunk.document_id == "doc-1"

    def test_child_text_is_substring_of_its_parent_text(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        text = _make_long_document(num_sentences=30, words_per_sentence=12)
        parents, chunks = chunk_document(
            document_id="doc-1",
            page_texts=[(1, text)],
            metadata=sample_metadata,
            embedder=fake_embedder,
            parent_chunk_tokens=150,
            child_chunk_tokens=30,
            chunk_overlap_ratio=0.1,
        )
        parents_by_id = {p.parent_id: p for p in parents}
        for chunk in chunks:
            parent = parents_by_id[chunk.parent_id]
            # Every word in the child chunk must appear in its parent's text —
            # children are built exclusively from their parent's sentences.
            assert all(word in parent.text for word in chunk.text.split())

    def test_child_token_counts_stay_within_reasonable_bound(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        text = _make_long_document(num_sentences=50, words_per_sentence=8)
        _, chunks = chunk_document(
            document_id="doc-1",
            page_texts=[(1, text)],
            metadata=sample_metadata,
            embedder=fake_embedder,
            parent_chunk_tokens=300,
            child_chunk_tokens=64,
            chunk_overlap_ratio=0.15,
        )
        for chunk in chunks:
            # A child may modestly exceed the budget only when a single
            # sentence alone is larger than the budget; with 8-word sentences
            # here that shouldn't happen, so enforce a hard, generous bound.
            assert chunk.token_count <= 64 * 1.5

    def test_overlap_ratio_produces_shared_content_across_children_within_a_parent(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        text = _make_long_document(num_sentences=40, words_per_sentence=10)
        parents, chunks = chunk_document(
            document_id="doc-1",
            page_texts=[(1, text)],
            metadata=sample_metadata,
            embedder=fake_embedder,
            parent_chunk_tokens=1000,
            child_chunk_tokens=50,
            chunk_overlap_ratio=0.2,
        )
        # Group children by parent and confirm consecutive children share
        # at least one word (evidence of overlap), for any parent with >1 child.
        by_parent: dict[str, list[ChunkRecord]] = {}
        for c in chunks:
            by_parent.setdefault(c.parent_id, []).append(c)

        found_multi_child_parent = False
        for parent_chunks in by_parent.values():
            if len(parent_chunks) < 2:
                continue
            found_multi_child_parent = True
            for prev, nxt in zip(parent_chunks, parent_chunks[1:], strict=False):
                prev_words = set(prev.text.split())
                next_words = set(nxt.text.split())
                assert prev_words & next_words, "expected overlapping words between children"

        assert found_multi_child_parent, "test setup should produce at least one multi-child parent"

    def test_page_number_propagates_to_parent(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        text = "First sentence on page five. Second sentence on page five too."
        parents, _ = chunk_document(
            document_id="doc-1",
            page_texts=[(5, text)],
            metadata=sample_metadata,
            embedder=fake_embedder,
        )
        assert parents[0].page_number == 5

    def test_modality_and_source_ref_propagate_to_parent_and_children(
        self, fake_embedder, sample_metadata: DocumentMetadata
    ) -> None:
        text = "A table description sentence. Another descriptive sentence about the table."
        parents, chunks = chunk_document(
            document_id="doc-1",
            page_texts=[(1, text)],
            metadata=sample_metadata,
            embedder=fake_embedder,
            modality=ContentModality.TABLE,
            source_ref="blob://doc-1/page-1-table.png",
        )
        assert parents[0].modality == ContentModality.TABLE
        assert parents[0].source_ref == "blob://doc-1/page-1-table.png"
        assert all(c.modality == ContentModality.TABLE for c in chunks)


class TestChunkRecordEmptyTextRejection:
    """Exercises the pydantic validator on ChunkRecord directly (ADR contract),
    confirming the ingestion pipeline can rely on it to reject blank chunks."""

    def test_blank_text_raises_validation_error(self, sample_metadata: DocumentMetadata) -> None:
        with pytest.raises(ValidationError):
            ChunkRecord(
                parent_id="p1",
                document_id="doc-1",
                text="   ",
                token_count=0,
                metadata=sample_metadata,
            )

    def test_empty_string_raises_validation_error(self, sample_metadata: DocumentMetadata) -> None:
        with pytest.raises(ValidationError):
            ChunkRecord(
                parent_id="p1",
                document_id="doc-1",
                text="",
                token_count=0,
                metadata=sample_metadata,
            )

    def test_nonblank_text_is_accepted(self, sample_metadata: DocumentMetadata) -> None:
        record = ChunkRecord(
            parent_id="p1",
            document_id="doc-1",
            text="valid text",
            token_count=2,
            metadata=sample_metadata,
        )
        assert record.text == "valid text"
