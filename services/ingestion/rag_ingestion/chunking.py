"""Parent-child hierarchical chunking (ADR-002).

Pipeline: sentence-split -> greedily pack sentences into ~parent_chunk_tokens
passages, but cut a parent early wherever the semantic-boundary heuristic
detects a sharp topic shift between consecutive sentences (local embedding
cosine distance) -> within each parent, re-split into ~child_chunk_tokens
overlapping children. Both cuts are sentence-boundary aware so children never
split mid-sentence.

Token counting uses a whitespace-based approximation (`_approx_token_count`)
rather than a real tokenizer, matching the ADR's "no heavy NLP deps" posture
— it's consistent enough to drive relative budgets (1024 vs 128 tokens) even
if it doesn't match any specific model's BPE count exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_core.schemas import ChunkRecord, ContentModality, DocumentMetadata, ParentContext

from rag_ingestion.embeddings import Embedder, cosine_similarity

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD_SPLIT = re.compile(r"\S+")


def _approx_token_count(text: str) -> int:
    """~1 token per word is a coarse but stable proxy; good enough to size
    budgets that are themselves round numbers (1024 / 128)."""
    return len(_WORD_SPLIT.findall(text))


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter — deliberately simple (no spaCy/nltk dependency).

    Falls back to returning the whole text as one "sentence" if no boundary
    is found, so callers never have to special-case empty output.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(normalized) if s.strip()]
    return sentences or [normalized]


@dataclass(frozen=True)
class _Sentence:
    text: str
    page_number: int | None
    token_count: int


def _semantic_boundaries(
    sentences: list[_Sentence], embedder: Embedder, similarity_threshold: float
) -> set[int]:
    """Returns the set of sentence indices `i` where a boundary should be forced
    *before* sentence `i` (i.e. sentence i-1 and i are semantically distant).

    Embeds each sentence individually and compares consecutive cosine
    similarities — a real production version might use windowed embeddings,
    but per-sentence is a reasonable, cheap approximation and still gives a
    genuine (non-stubbed) semantic signal from the local embedding model.
    """
    if len(sentences) < 2:
        return set()

    vectors = embedder.embed([s.text for s in sentences])
    boundaries: set[int] = set()
    for i in range(1, len(vectors)):
        similarity = cosine_similarity(vectors[i - 1], vectors[i])
        if similarity < similarity_threshold:
            boundaries.add(i)
    return boundaries


def _pack_parents(
    sentences: list[_Sentence],
    *,
    parent_chunk_tokens: int,
    boundaries: set[int],
) -> list[list[_Sentence]]:
    """Greedily accumulate sentences into parent groups, cutting when either the
    token budget is exceeded or a forced semantic boundary is reached — but
    never producing an empty parent just because a boundary immediately
    follows a cut."""
    parents: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    current_tokens = 0

    for index, sentence in enumerate(sentences):
        starts_new_parent = index in boundaries and current
        exceeds_budget = current and current_tokens + sentence.token_count > parent_chunk_tokens

        if starts_new_parent or exceeds_budget:
            parents.append(current)
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence.token_count

    if current:
        parents.append(current)

    return parents


def _pack_children(
    sentences: list[_Sentence],
    *,
    child_chunk_tokens: int,
    overlap_ratio: float,
) -> list[list[_Sentence]]:
    """Sliding-window sentence packing for children, with `overlap_ratio` of the
    previous child's *trailing sentences* (by token budget) repeated at the
    start of the next child."""
    if not sentences:
        return []

    overlap_tokens_budget = max(0, round(child_chunk_tokens * overlap_ratio))
    children: list[list[_Sentence]] = []
    start = 0
    n = len(sentences)

    while start < n:
        current: list[_Sentence] = []
        tokens = 0
        i = start
        while i < n:
            next_tokens = sentences[i].token_count
            if current and tokens + next_tokens > child_chunk_tokens:
                break
            current.append(sentences[i])
            tokens += next_tokens
            i += 1

        if not current:
            # A single sentence alone exceeds the child budget; take it anyway
            # rather than infinite-looping or silently dropping content.
            current = [sentences[i]]
            i += 1

        children.append(current)

        if i >= n:
            break

        # Walk backward from the end of the current child to find how many
        # trailing sentences fit within the overlap token budget. Capped at
        # len(current) - 1 so at least one sentence of forward progress is
        # always made — an overlap consuming *every* sentence of `current`
        # would repeat the same window and never advance `start`.
        overlap_tokens = 0
        overlap_count = 0
        max_overlap_count = len(current) - 1
        for sentence in reversed(current):
            if overlap_count >= max_overlap_count:
                break
            if overlap_tokens + sentence.token_count > overlap_tokens_budget:
                break
            overlap_tokens += sentence.token_count
            overlap_count += 1

        start = i - overlap_count if overlap_count else i

    return children


def chunk_document(
    *,
    document_id: str,
    page_texts: list[tuple[int, str]],
    metadata: DocumentMetadata,
    embedder: Embedder,
    parent_chunk_tokens: int = 1024,
    child_chunk_tokens: int = 128,
    chunk_overlap_ratio: float = 0.15,
    semantic_similarity_threshold: float = 0.55,
    modality: ContentModality = ContentModality.PROSE,
    source_ref: str | None = None,
) -> tuple[list[ParentContext], list[ChunkRecord]]:
    """Produce `ParentContext` + `ChunkRecord` lists for one document (or one
    modality-homogeneous slice of it, e.g. a single vision-described table).

    `page_texts` is a list of (page_number, text) pairs in document order;
    sentence splitting happens per page so each sentence retains its
    originating page number for citation purposes.
    """
    sentences: list[_Sentence] = []
    for page_number, text in page_texts:
        for sentence_text in split_sentences(text):
            sentences.append(
                _Sentence(
                    text=sentence_text,
                    page_number=page_number,
                    token_count=_approx_token_count(sentence_text),
                )
            )

    if not sentences:
        return [], []

    boundaries = _semantic_boundaries(sentences, embedder, semantic_similarity_threshold)
    parent_groups = _pack_parents(
        sentences, parent_chunk_tokens=parent_chunk_tokens, boundaries=boundaries
    )

    parents: list[ParentContext] = []
    chunks: list[ChunkRecord] = []

    for group in parent_groups:
        parent_text = " ".join(s.text for s in group)
        if not parent_text.strip():
            continue
        parent_page_number = group[0].page_number
        parent = ParentContext(
            parent_id=f"{document_id}:p{len(parents)}",
            document_id=document_id,
            text=parent_text,
            page_number=parent_page_number,
            modality=modality,
            source_ref=source_ref,
        )
        parents.append(parent)

        child_groups = _pack_children(
            group, child_chunk_tokens=child_chunk_tokens, overlap_ratio=chunk_overlap_ratio
        )
        for child_group in child_groups:
            child_text = " ".join(s.text for s in child_group)
            if not child_text.strip():
                continue
            chunks.append(
                ChunkRecord(
                    parent_id=parent.parent_id,
                    document_id=document_id,
                    text=child_text,
                    modality=modality,
                    token_count=_approx_token_count(child_text),
                    metadata=metadata,
                )
            )

    return parents, chunks
