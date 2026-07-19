"""ADR-008: entropy/information-density-based context compression.

A real, deterministic heuristic — no external LLMLingua-style package, no LLM
call. For each parent-context passage we:

1. Split the text into sentence-like units, always preserving table-formatted
   lines and lines that are dense with numeric content verbatim (never split
   or scored for removal).
2. Score every *removable* sentence on two signals:
   - Rarity: the mean inverse document frequency of its distinct tokens
     against the token-frequency table built from the whole passage set.
     Sentences full of distinct, low-frequency ("rare") words score higher —
     they are assumed to carry more information than boilerplate filler.
   - Redundancy: cosine similarity (computed with plain bag-of-words count
     vectors, no embeddings) against the sentences already kept. A sentence
     that is nearly identical to something already kept is penalized, since
     keeping it adds tokens without adding information.
3. Greedily keep sentences highest-score-first until the running token count
   hits `target_ratio * original_token_count`, always keeping every
   protected (numeric/table) sentence regardless of score or budget.

Determinism: no randomness anywhere; ties are broken by original sentence
order, so the same input always produces the same output.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_core.schemas import RetrievedChunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.]*")
_TABLE_LINE_RE = re.compile(r"\|")
_MULTI_SPACE_COLUMNS_RE = re.compile(r"\S(?: {2,}|\t)\S")
_NUMERIC_TOKEN_RE = re.compile(r"\d")
_NUMERIC_DENSITY_THRESHOLD = 0.15
"""A sentence is "numeric-heavy" if at least this fraction of its tokens
contain a digit — such sentences are never removed (ADR-008 numeric-span
protection)."""


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _is_table_or_columnar(line: str) -> bool:
    """Detect markdown tables ('|') or whitespace-aligned columnar text."""
    if _TABLE_LINE_RE.search(line):
        return True
    return bool(_MULTI_SPACE_COLUMNS_RE.search(line))


def _is_numeric_heavy(tokens: list[str]) -> bool:
    if not tokens:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.search(t))
    return (numeric / len(tokens)) >= _NUMERIC_DENSITY_THRESHOLD


def _split_into_units(text: str) -> list[str]:
    """Split into line-respecting sentence units.

    Table/columnar lines are kept as their own atomic unit (never merged with
    surrounding prose, never re-split by sentence punctuation) so they can be
    protected wholesale.
    """
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip("\n")
        if not line.strip():
            continue
        if _is_table_or_columnar(line):
            units.append(line)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line.strip()):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


@dataclass(frozen=True)
class _Unit:
    text: str
    tokens: tuple[str, ...]
    protected: bool
    order: int


def _build_units(text: str) -> list[_Unit]:
    raw_units = _split_into_units(text)
    result: list[_Unit] = []
    for i, u in enumerate(raw_units):
        tokens = tuple(_tokenize(u))
        protected = _is_table_or_columnar(u) or _is_numeric_heavy(list(tokens))
        result.append(_Unit(text=u, tokens=tokens, protected=protected, order=i))
    return result


def _document_frequencies(all_units: list[_Unit]) -> Counter[str]:
    df: Counter[str] = Counter()
    for unit in all_units:
        for token in set(unit.tokens):
            df[token] += 1
    return df


def _rarity_score(unit: _Unit, doc_freq: Counter[str], n_docs: int) -> float:
    """Mean inverse-document-frequency of the unit's distinct tokens.

    Higher = more distinct, low-frequency ("rare") vocabulary => assumed more
    information-dense. Uses smoothed IDF so a token appearing in every unit
    scores near zero and a token appearing once scores highest.
    """
    distinct = set(unit.tokens)
    if not distinct:
        return 0.0
    total = 0.0
    for token in distinct:
        freq = doc_freq.get(token, 1)
        idf = math.log((n_docs + 1) / (freq + 0.5))
        total += idf
    return total / len(distinct)


def _bow_vector(tokens: tuple[str, ...]) -> Counter[str]:
    return Counter(tokens)


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _max_similarity_to_kept(unit: _Unit, kept_vectors: list[Counter[str]]) -> float:
    if not kept_vectors:
        return 0.0
    vec = _bow_vector(unit.tokens)
    return max((_cosine_similarity(vec, kv) for kv in kept_vectors), default=0.0)


def _estimate_tokens(units: list[_Unit]) -> int:
    """Cheap deterministic token proxy: word count. Good enough for a ratio-
    based budget; we don't need exact model tokenization here."""
    return sum(len(u.tokens) for u in units)


def compress_text(text: str, target_ratio: float) -> str:
    """Compress a single passage of text, preserving protected units verbatim.

    Deterministic: identical input always yields identical output. Sentences
    are re-emitted in their original order (not selection order) so the
    compressed passage still reads coherently.
    """
    if not text.strip():
        return text
    target_ratio = max(0.0, min(1.0, target_ratio))

    units = _build_units(text)
    if not units:
        return text

    total_tokens = _estimate_tokens(units)
    if total_tokens == 0:
        return text
    target_tokens = max(1, round(total_tokens * target_ratio))

    doc_freq = _document_frequencies(units)
    n_docs = len(units)

    protected = [u for u in units if u.protected]
    removable = [u for u in units if not u.protected]

    protected_tokens = sum(len(u.tokens) for u in protected)

    kept: dict[int, _Unit] = {u.order: u for u in protected}
    kept_vectors: list[Counter[str]] = [_bow_vector(u.tokens) for u in protected]
    running_tokens = protected_tokens

    # Score removable units once, up front, against the *original* kept set
    # (protected units) — this keeps scoring deterministic and independent of
    # selection order for the redundancy-vs-protected-content comparison.
    # Redundancy against *other selected* removable sentences is folded in
    # incrementally as we greedily add them below.
    scored: list[tuple[float, int, _Unit]] = []
    for u in removable:
        rarity = _rarity_score(u, doc_freq, n_docs)
        redundancy = _max_similarity_to_kept(u, kept_vectors)
        score = rarity * (1.0 - redundancy)
        scored.append((score, u.order, u))

    # Highest information density first; ties broken by original order for
    # determinism (Python's sort is stable, so sorting solely on -score with
    # a secondary explicit order key guarantees a total, reproducible order).
    scored.sort(key=lambda item: (-item[0], item[1]))

    for _score, _order, unit in scored:
        if running_tokens >= target_tokens:
            break
        # Recompute redundancy against everything kept so far (protected +
        # previously accepted removable units) before committing, so near-
        # duplicate sentences already queued for inclusion are still caught.
        redundancy_now = _max_similarity_to_kept(unit, kept_vectors)
        if redundancy_now > 0.9:
            continue
        kept[unit.order] = unit
        kept_vectors.append(_bow_vector(unit.tokens))
        running_tokens += len(unit.tokens)

    ordered = [kept[i] for i in sorted(kept)]
    return " ".join(u.text for u in ordered)


def compress_context(chunks: list[RetrievedChunk], target_ratio: float) -> list[RetrievedChunk]:
    """Apply `compress_text` to each chunk's parent context text.

    Returns new `RetrievedChunk` instances (does not mutate inputs) with
    `parent.text` replaced by its compressed form. `chunk.text` (the small
    child chunk used for retrieval) is left untouched — only the larger
    parent passage handed to the generator is compressed.
    """
    compressed: list[RetrievedChunk] = []
    for rc in chunks:
        new_parent = rc.parent.model_copy(
            update={"text": compress_text(rc.parent.text, target_ratio)}
        )
        compressed.append(rc.model_copy(update={"parent": new_parent}))
    return compressed
