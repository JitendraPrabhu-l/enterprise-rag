"""Citation-identifier grounding check (ADR-028), shared by the generation
service's serve-time guardrail and the eval service's eval-time metric.

Lives in rag_core (not rag_generation) because it is a pure function over
shared schemas (`Citation`, `RetrievedChunk`) with no generation-specific
dependency, and the eval service needs it too — duplicating it per-service
would be exactly the kind of split-ownership drift ADR-020 (sparse
indexing) already fixed once for a different pair of services.
"""

from __future__ import annotations

from rag_core.schemas import Citation, RetrievedChunk


def find_ungrounded_citations(
    citations: list[Citation], context_chunks: list[RetrievedChunk]
) -> list[Citation]:
    """Return the subset of `citations` whose `parent_id` was NOT among the
    chunks actually shown to the model.

    A citation naming a real-looking but never-retrieved `parent_id` is a
    distinct failure mode from an uncited answer (guarded against
    separately in `rag_generation.pipeline`) and from ungrounded answer
    TEXT (the RAG Triad's faithfulness judge, ADR-009) — this is the
    identifier-level check neither of those catches: schema-valid,
    present, and still wrong (a fabricated or training-data-recalled
    parent_id).

    Pure and dependency-free (no LLM call) precisely because it doesn't
    need one — "was this parent_id in the set we retrieved" is a set
    membership check, not a judgment call. That's what makes it usable
    BOTH as a synchronous serve-time guardrail (no added latency) and as
    a deterministic eval-time metric (no judge-model cost or flakiness),
    unlike every other axis in the RAG Triad.
    """
    valid_parent_ids = {chunk.parent.parent_id for chunk in context_chunks}
    return [c for c in citations if c.parent_id not in valid_parent_ids]
