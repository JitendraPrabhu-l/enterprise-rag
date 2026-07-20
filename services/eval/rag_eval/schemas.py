"""Pydantic models specific to the eval service.

`TriadScore` / `TriadResult` are the output contract of the three RAG Triad
judge functions (ADR-009). `SyntheticEvalItem` is one row of a repeatable,
synthetically generated eval dataset used by the CI/CD gate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TriadScore(BaseModel):
    """A single LLM-as-judge score with a one-line justification.

    `score` is always normalized to the 1.0 = best / 0.0 = worst convention
    used by every RAG Triad axis in ADR-009 (fully-grounded, fully-relevant,
    fully-precise respectively).
    """

    score: float = Field(ge=0.0, le=1.0)
    justification: str = Field(min_length=1)


class TriadResult(BaseModel):
    """The three RAG Triad axes for one generated answer (ADR-009)."""

    faithfulness: TriadScore
    answer_relevance: TriadScore
    context_precision: TriadScore


class SyntheticEvalItem(BaseModel):
    """One question/context/answer triplet in a synthetic eval dataset.

    Generated from a single source passage so the dataset is repeatable and
    traceable back to the document it was derived from.
    """

    question: str = Field(min_length=1)
    reference_context: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)


class GoldenRetrievalItem(BaseModel):
    """One query with its known-relevant chunk ids for the deterministic,
    LLM-free retrieval gate (ADR-037).

    Unlike SyntheticEvalItem (which judges the final *answer* via an LLM), this
    marks which chunk_ids retrieval *should* surface for a query, so
    recall@k / MRR / nDCG can be computed with no model in the loop. Seed it
    from production thumbs-down feedback (ADR-027) — real misses — as well as
    synthetic questions, so the gate hardens against failures that actually
    happened, not only ones imagined at design time.
    """

    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(
        min_length=1, description="chunk_ids a correct retrieval must surface for this query."
    )
    tenant_id: str = "public"
    source_domains: list[str] | None = None
