"""Tests for the ADR-028 citation-grounding check wired into eval_runner:
a deterministic, non-LLM axis distinct from the three RAG Triad judges.
Any ungrounded citation on any item is a hard pipeline bug (a fabricated
citation reached the user) and fails the gate outright, regardless of how
high the Triad means are.
"""

from __future__ import annotations

from uuid import uuid4

from rag_core.schemas import Citation, GenerationResponse, QueryRequest, RetrievedChunk

from rag_eval.eval_runner import EvalItemResult, run_eval_gate, run_pipeline_and_score_item
from tests.conftest import FakeAsyncOpenAI, make_score_message
from tests.test_eval_runner import _make_item, _make_retrieved_chunk


class FakePipelineClient:
    def __init__(
        self, *, retrieved_chunks: list[RetrievedChunk], generation_response: GenerationResponse
    ) -> None:
        self._retrieved_chunks = retrieved_chunks
        self._generation_response = generation_response

    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        return self._retrieved_chunks

    async def generate(self, request: QueryRequest) -> GenerationResponse:
        return self._generation_response


def _generation_response_with_citation(parent_id: str) -> GenerationResponse:
    return GenerationResponse(
        request_id=uuid4(),
        answer="An answer citing a specific passage.",
        citations=[Citation(parent_id=parent_id, document_id="doc-1")],
        model="claude-sonnet-5",
    )


class TestRunPipelineAndScoreItemCitationGrounding:
    async def test_citation_matching_retrieved_chunk_is_grounded(self) -> None:
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("Actual context.")],  # parent_id="parent-1"
            generation_response=_generation_response_with_citation("parent-1"),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(0.9, "f"),
                make_score_message(0.9, "a"),
                make_score_message(0.9, "c"),
            ]
        )

        result = await run_pipeline_and_score_item(
            _make_item(),
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
        )

        assert result.ungrounded_citations == []

    async def test_citation_naming_never_retrieved_parent_id_is_flagged(self) -> None:
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("Actual context.")],  # parent_id="parent-1"
            generation_response=_generation_response_with_citation("fabricated-parent-id"),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(0.9, "f"),
                make_score_message(0.9, "a"),
                make_score_message(0.9, "c"),
            ]
        )

        result = await run_pipeline_and_score_item(
            _make_item(),
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
        )

        assert len(result.ungrounded_citations) == 1
        assert result.ungrounded_citations[0].parent_id == "fabricated-parent-id"


class TestEvalGateResultGatingOnUngroundedCitations:
    async def test_gate_fails_when_an_item_has_an_ungrounded_citation_even_if_triad_passes(
        self,
    ) -> None:
        """The core property: a fabricated citation must fail the gate
        outright, even when every RAG Triad axis comfortably passes its
        threshold. This is what makes ADR-028 a hard gate, not a graded
        quality signal averaged in with the others."""
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("Actual context.")],  # parent_id="parent-1"
            generation_response=_generation_response_with_citation("fabricated-parent-id"),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(1.0, "f"),
                make_score_message(1.0, "a"),
                make_score_message(1.0, "c"),
            ]
        )

        result = await run_eval_gate(
            [_make_item()],
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
            faithfulness_threshold=0.5,
            answer_relevance_threshold=0.5,
            context_precision_threshold=0.5,
        )

        assert result.faithfulness.passed is True
        assert result.answer_relevance.passed is True
        assert result.context_precision.passed is True
        assert result.item_results[0].ungrounded_citations != []
        assert result.passed is False

    async def test_gate_passes_when_no_item_has_ungrounded_citations(self) -> None:
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("Actual context.")],  # parent_id="parent-1"
            generation_response=_generation_response_with_citation("parent-1"),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(1.0, "f"),
                make_score_message(1.0, "a"),
                make_score_message(1.0, "c"),
            ]
        )

        result = await run_eval_gate(
            [_make_item()],
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
            faithfulness_threshold=0.5,
            answer_relevance_threshold=0.5,
            context_precision_threshold=0.5,
        )

        assert result.passed is True

    def test_default_ungrounded_citations_is_empty_list(self) -> None:
        """Direct construction (as any pre-ADR-028 caller would) must default
        to an empty list, not require every caller to specify it."""
        from rag_eval.schemas import TriadResult, TriadScore

        score = TriadScore(score=1.0, justification="ok")
        triad = TriadResult(faithfulness=score, answer_relevance=score, context_precision=score)

        result = EvalItemResult(
            question="q",
            source_document_id="doc-1",
            answer="a",
            retrieved_context=["c"],
            triad=triad,
        )
        assert result.ungrounded_citations == []
