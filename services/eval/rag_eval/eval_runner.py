"""Core orchestration for running a synthetic eval dataset through the RAG
pipeline, scoring it on the RAG Triad, and aggregating pass/fail results.

Used by both the CI/CD gate CLI script (`scripts/run_eval_gate.py`) and any
programmatic caller (e.g. the production sampling API could reuse the
aggregation logic, though `/score` scores single interactions directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI
from rag_core.citation_verification import find_ungrounded_citations
from rag_core.schemas import Citation, QueryRequest

from rag_eval.judges import score_answer_relevance, score_context_precision, score_faithfulness
from rag_eval.pipeline_client import PipelineClient
from rag_eval.schemas import SyntheticEvalItem, TriadResult


@dataclass
class EvalItemResult:
    """The full result of running and scoring one dataset item."""

    question: str
    source_document_id: str
    answer: str
    retrieved_context: list[str]
    triad: TriadResult
    ungrounded_citations: list[Citation] = field(default_factory=list)
    """ADR-028: citations in `answer` whose parent_id was never in
    `retrieved_context` — deterministic, not part of the RAG Triad (which
    are all LLM-judge scores). Any non-empty list here is a hard pipeline
    bug (a fabricated citation reached the user), not a graded quality
    axis, which is why it gates `EvalGateResult.passed` directly rather
    than contributing to a mean-score threshold like the Triad axes."""


@dataclass
class AxisSummary:
    """Aggregated mean score and threshold comparison for one Triad axis."""

    mean_score: float
    threshold: float
    passed: bool


@dataclass
class EvalGateResult:
    """The full result of an eval gate run: per-item detail plus aggregates."""

    item_results: list[EvalItemResult]
    faithfulness: AxisSummary
    answer_relevance: AxisSummary
    context_precision: AxisSummary
    failed_items: list[str] = field(default_factory=list)
    """Human-readable descriptions of items that raised errors during scoring."""

    @property
    def passed(self) -> bool:
        """The gate passes iff every axis meets its threshold, no item
        errored, AND no item produced an ungrounded citation (ADR-028) —
        the last is a hard correctness bug (a fabricated citation reached
        the user), not a graded quality axis, so a single occurrence fails
        the gate regardless of how high the Triad means are."""
        return (
            self.faithfulness.passed
            and self.answer_relevance.passed
            and self.context_precision.passed
            and not self.failed_items
            and not any(r.ungrounded_citations for r in self.item_results)
        )


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty score list.")
    return sum(values) / len(values)


def summarize_axis(scores: list[float], threshold: float) -> AxisSummary:
    """Compute the mean of `scores` and compare it against `threshold`.

    The axis passes iff the mean score is greater than or equal to the
    threshold (inclusive) — matches the CI gate semantics described in the
    brief (e.g. "faithfulness >= 0.8").
    """
    mean_score = _mean(scores)
    return AxisSummary(mean_score=mean_score, threshold=threshold, passed=mean_score >= threshold)


async def run_pipeline_and_score_item(
    item: SyntheticEvalItem,
    *,
    pipeline_client: PipelineClient,
    judge_client: AsyncOpenAI,
    judge_model: str,
    tenant_id: str = "public",
    judge_max_retries: int = 3,
) -> EvalItemResult:
    """Run one dataset item through the real retrieval + generation pipeline,
    then score the resulting interaction on all three RAG Triad axes.

    Faithfulness is scored against the context actually returned by the
    pipeline (not the synthetic `reference_context`) — the gate is meant to
    catch real pipeline regressions, not just judge the synthetic data itself.
    """
    query_request = QueryRequest(query=item.question, tenant_id=tenant_id)

    retrieved_chunks = await pipeline_client.retrieve(query_request)
    generation_response = await pipeline_client.generate(query_request)

    retrieved_texts = [chunk.chunk.text for chunk in retrieved_chunks]

    faithfulness = await score_faithfulness(
        judge_client,
        model=judge_model,
        answer=generation_response.answer,
        context=retrieved_texts,
        max_retries=judge_max_retries,
    )
    answer_relevance = await score_answer_relevance(
        judge_client,
        model=judge_model,
        question=item.question,
        answer=generation_response.answer,
        max_retries=judge_max_retries,
    )
    context_precision = await score_context_precision(
        judge_client,
        model=judge_model,
        question=item.question,
        retrieved_chunks=retrieved_texts,
        max_retries=judge_max_retries,
    )

    # ADR-028: deterministic, non-LLM check — does every citation the
    # generator returned actually name a parent_id from THIS item's
    # retrieved set. Reuses `retrieved_chunks` (the same context the Triad
    # judges above score against) rather than introducing a second notion
    # of "the context" for this one check.
    ungrounded_citations = find_ungrounded_citations(
        generation_response.citations, retrieved_chunks
    )

    return EvalItemResult(
        question=item.question,
        source_document_id=item.source_document_id,
        answer=generation_response.answer,
        retrieved_context=retrieved_texts,
        triad=TriadResult(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
        ),
        ungrounded_citations=ungrounded_citations,
    )


async def run_eval_gate(
    dataset: list[SyntheticEvalItem],
    *,
    pipeline_client: PipelineClient,
    judge_client: AsyncOpenAI,
    judge_model: str,
    faithfulness_threshold: float,
    answer_relevance_threshold: float,
    context_precision_threshold: float,
    tenant_id: str = "public",
    judge_max_retries: int = 3,
) -> EvalGateResult:
    """Run every item in `dataset` through the pipeline, score it on the RAG
    Triad, and aggregate mean scores per axis against the given thresholds.

    An item that raises during pipeline execution or judging is recorded in
    `EvalGateResult.failed_items` and excluded from the mean-score
    computation; any recorded failure makes the overall gate fail regardless
    of the axis means, since a scoring failure means the pipeline could not
    be evaluated at all for that item (never silently dropped and ignored).
    """
    item_results: list[EvalItemResult] = []
    failed_items: list[str] = []

    for item in dataset:
        try:
            result = await run_pipeline_and_score_item(
                item,
                pipeline_client=pipeline_client,
                judge_client=judge_client,
                judge_model=judge_model,
                tenant_id=tenant_id,
                judge_max_retries=judge_max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - intentionally broad: any failure gates CI
            failed_items.append(f"{item.source_document_id!r} ({item.question!r}): {exc}")
            continue
        item_results.append(result)

    if not item_results:
        raise RuntimeError(
            "No dataset items were successfully scored — cannot compute a gate result. "
            f"Failures: {failed_items}"
        )

    faithfulness_scores = [r.triad.faithfulness.score for r in item_results]
    answer_relevance_scores = [r.triad.answer_relevance.score for r in item_results]
    context_precision_scores = [r.triad.context_precision.score for r in item_results]

    return EvalGateResult(
        item_results=item_results,
        faithfulness=summarize_axis(faithfulness_scores, faithfulness_threshold),
        answer_relevance=summarize_axis(answer_relevance_scores, answer_relevance_threshold),
        context_precision=summarize_axis(context_precision_scores, context_precision_threshold),
        failed_items=failed_items,
    )
