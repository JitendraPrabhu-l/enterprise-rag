"""Tests for rag_eval.eval_runner: pipeline orchestration and aggregation.

Mocks the pipeline_client HTTP calls (via a fake PipelineClient-shaped object
for unit tests of aggregation/threshold logic, and via respx for one
end-to-end check of PipelineClient itself talking HTTP). Verifies:
  - mean-score aggregation math across items is correct
  - threshold pass/fail comparison is correct with known inputs
  - a scoring failure on one item is recorded and fails the overall gate
"""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
import respx
from rag_core.schemas import (
    AccessRole,
    ChunkRecord,
    Citation,
    ContentModality,
    DocumentMetadata,
    GenerationResponse,
    ParentContext,
    QueryRequest,
    RetrievedChunk,
    SourceType,
)

from rag_eval.eval_runner import (
    run_eval_gate,
    run_pipeline_and_score_item,
    summarize_axis,
)
from rag_eval.pipeline_client import PipelineClient
from rag_eval.schemas import SyntheticEvalItem
from tests.conftest import FakeAsyncOpenAI, make_score_message


def _make_retrieved_chunk(text: str) -> RetrievedChunk:
    metadata = DocumentMetadata(
        document_id="doc-1",
        source_type=SourceType.HTML,
        source_domain="test-domain",
        tenant_id="public",
        access_role=AccessRole.PUBLIC,
        last_updated_epoch=0,
    )
    chunk = ChunkRecord(
        parent_id="parent-1",
        document_id="doc-1",
        text=text,
        modality=ContentModality.PROSE,
        token_count=10,
        metadata=metadata,
    )
    parent = ParentContext(parent_id="parent-1", document_id="doc-1", text=text)
    return RetrievedChunk(chunk=chunk, parent=parent, dense_score=0.9)


class FakePipelineClient:
    """Stand-in for PipelineClient used to unit test eval_runner orchestration
    without going over HTTP.
    """

    def __init__(
        self,
        *,
        retrieved_chunks: list[RetrievedChunk],
        generation_response: GenerationResponse,
        raise_on_retrieve: Exception | None = None,
        raise_on_generate: Exception | None = None,
    ) -> None:
        self._retrieved_chunks = retrieved_chunks
        self._generation_response = generation_response
        self._raise_on_retrieve = raise_on_retrieve
        self._raise_on_generate = raise_on_generate
        self.retrieve_calls: list[QueryRequest] = []
        self.generate_calls: list[QueryRequest] = []

    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        self.retrieve_calls.append(request)
        if self._raise_on_retrieve:
            raise self._raise_on_retrieve
        return self._retrieved_chunks

    async def generate(self, request: QueryRequest) -> GenerationResponse:
        self.generate_calls.append(request)
        if self._raise_on_generate:
            raise self._raise_on_generate
        return self._generation_response


def _make_generation_response(answer: str) -> GenerationResponse:
    return GenerationResponse(
        request_id=uuid4(),
        answer=answer,
        citations=[Citation(parent_id="parent-1", document_id="doc-1")],
        model="claude-sonnet-5",
    )


def _make_item(question: str = "What is X?", doc_id: str = "doc-1") -> SyntheticEvalItem:
    return SyntheticEvalItem(
        question=question,
        reference_context="X is a thing.",
        reference_answer="X is a thing.",
        source_document_id=doc_id,
    )


class TestSummarizeAxis:
    def test_mean_computation_is_correct(self) -> None:
        summary = summarize_axis([0.8, 0.9, 1.0], threshold=0.8)
        assert summary.mean_score == pytest.approx(0.9)

    def test_passes_when_mean_meets_threshold_exactly(self) -> None:
        summary = summarize_axis([0.8, 0.8, 0.8], threshold=0.8)
        assert summary.passed is True

    def test_passes_when_mean_exceeds_threshold(self) -> None:
        summary = summarize_axis([0.9, 0.95], threshold=0.8)
        assert summary.passed is True

    def test_fails_when_mean_below_threshold(self) -> None:
        summary = summarize_axis([0.5, 0.6], threshold=0.8)
        assert summary.passed is False
        assert summary.mean_score == pytest.approx(0.55)

    def test_single_item_mean(self) -> None:
        summary = summarize_axis([0.42], threshold=0.5)
        assert summary.mean_score == pytest.approx(0.42)
        assert summary.passed is False

    def test_empty_scores_raises(self) -> None:
        with pytest.raises(ValueError):
            summarize_axis([], threshold=0.8)


class TestRunPipelineAndScoreItem:
    async def test_scores_against_pipeline_returned_context_not_reference_context(self) -> None:
        """Faithfulness must be judged against the context the pipeline actually
        returned, not the synthetic dataset's reference_context.
        """
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("Actual pipeline context.")],
            generation_response=_make_generation_response("An answer."),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(0.9, "faithful"),
                make_score_message(0.8, "relevant"),
                make_score_message(1.0, "precise"),
            ]
        )
        item = _make_item()

        result = await run_pipeline_and_score_item(
            item,
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
        )

        assert result.retrieved_context == ["Actual pipeline context."]
        assert result.answer == "An answer."
        assert result.triad.faithfulness.score == pytest.approx(0.9)
        assert result.triad.answer_relevance.score == pytest.approx(0.8)
        assert result.triad.context_precision.score == pytest.approx(1.0)

        # faithfulness judge call must reference the pipeline's context, not reference_context
        faithfulness_call = judge_client.chat.completions.calls[0]
        assert "Actual pipeline context." in faithfulness_call["messages"][0]["content"]
        assert "reference_context" not in faithfulness_call["messages"][0]["content"]


class TestRunEvalGate:
    async def test_aggregation_across_multiple_items(self) -> None:
        """Verify mean scores are computed correctly across a multi-item dataset."""
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("context")],
            generation_response=_make_generation_response("answer"),
        )
        # 3 items x 3 judge calls each = 9 responses queued.
        # Faithfulness scores: 1.0, 0.8, 0.6 -> mean 0.8
        # Answer relevance: 0.9, 0.9, 0.9 -> mean 0.9
        # Context precision: 1.0, 1.0, 1.0 -> mean 1.0
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(1.0, "f1"),
                make_score_message(0.9, "a1"),
                make_score_message(1.0, "c1"),
                make_score_message(0.8, "f2"),
                make_score_message(0.9, "a2"),
                make_score_message(1.0, "c2"),
                make_score_message(0.6, "f3"),
                make_score_message(0.9, "a3"),
                make_score_message(1.0, "c3"),
            ]
        )
        dataset = [_make_item(f"Q{i}", f"doc-{i}") for i in range(3)]

        result = await run_eval_gate(
            dataset,
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
            faithfulness_threshold=0.8,
            answer_relevance_threshold=0.75,
            context_precision_threshold=0.7,
        )

        assert result.faithfulness.mean_score == pytest.approx(0.8)
        assert result.answer_relevance.mean_score == pytest.approx(0.9)
        assert result.context_precision.mean_score == pytest.approx(1.0)
        assert len(result.item_results) == 3
        assert result.failed_items == []

    async def test_gate_passes_when_all_axes_meet_threshold(self) -> None:
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("context")],
            generation_response=_make_generation_response("answer"),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(0.8, "f"),
                make_score_message(0.75, "a"),
                make_score_message(0.7, "c"),
            ]
        )

        result = await run_eval_gate(
            [_make_item()],
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
            faithfulness_threshold=0.8,
            answer_relevance_threshold=0.75,
            context_precision_threshold=0.7,
        )

        assert result.passed is True

    async def test_gate_fails_when_one_axis_misses_threshold(self) -> None:
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("context")],
            generation_response=_make_generation_response("answer"),
        )
        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(0.79, "f"),  # below 0.8 threshold
                make_score_message(0.9, "a"),
                make_score_message(0.9, "c"),
            ]
        )

        result = await run_eval_gate(
            [_make_item()],
            pipeline_client=pipeline,  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
            faithfulness_threshold=0.8,
            answer_relevance_threshold=0.75,
            context_precision_threshold=0.7,
        )

        assert result.passed is False
        assert result.faithfulness.passed is False
        assert result.answer_relevance.passed is True
        assert result.context_precision.passed is True

    async def test_item_scoring_failure_is_recorded_and_fails_gate(self) -> None:
        """A pipeline error on one item must not be silently dropped -- it should
        appear in failed_items and force the overall gate to fail, even if the
        successfully-scored items all pass their thresholds.
        """
        pipeline = FakePipelineClient(
            retrieved_chunks=[_make_retrieved_chunk("context")],
            generation_response=_make_generation_response("answer"),
            raise_on_generate=RuntimeError("generation service unreachable"),
        )
        judge_client = FakeAsyncOpenAI([])

        with pytest.raises(RuntimeError):
            # every item fails -> no item_results -> run_eval_gate raises
            await run_eval_gate(
                [_make_item()],
                pipeline_client=pipeline,  # type: ignore[arg-type]
                judge_client=judge_client,  # type: ignore[arg-type]
                judge_model="claude-haiku-4-5-20251001",
                faithfulness_threshold=0.8,
                answer_relevance_threshold=0.75,
                context_precision_threshold=0.7,
            )

    async def test_partial_item_failure_fails_gate_even_if_thresholds_met(self) -> None:
        class MixedPipelineClient:
            def __init__(self) -> None:
                self._calls = 0

            async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
                return [_make_retrieved_chunk("context")]

            async def generate(self, request: QueryRequest) -> GenerationResponse:
                self._calls += 1
                if self._calls == 1:
                    return _make_generation_response("good answer")
                raise RuntimeError("simulated failure on second item")

        judge_client = FakeAsyncOpenAI(
            [
                make_score_message(0.9, "f"),
                make_score_message(0.9, "a"),
                make_score_message(0.9, "c"),
            ]
        )

        result = await run_eval_gate(
            [_make_item("Q1", "doc-1"), _make_item("Q2", "doc-2")],
            pipeline_client=MixedPipelineClient(),  # type: ignore[arg-type]
            judge_client=judge_client,  # type: ignore[arg-type]
            judge_model="claude-haiku-4-5-20251001",
            faithfulness_threshold=0.8,
            answer_relevance_threshold=0.75,
            context_precision_threshold=0.7,
        )

        assert len(result.item_results) == 1
        assert len(result.failed_items) == 1
        assert result.faithfulness.passed is True  # the one scored item meets threshold
        assert result.passed is False  # but overall gate still fails due to the failed item


class TestPipelineClientHttp:
    """One respx-backed check that PipelineClient itself does the right HTTP calls
    and parses responses per the rag_core.schemas contract.
    """

    async def test_retrieve_and_generate_round_trip(self) -> None:
        chunk = _make_retrieved_chunk("hello world")
        gen_response = _make_generation_response("the answer")

        with respx.mock(assert_all_called=True) as mock_router:
            mock_router.post("http://retrieval.test/retrieve").mock(
                return_value=httpx.Response(200, json=[json.loads(chunk.model_dump_json())])
            )
            mock_router.post("http://generation.test/generate").mock(
                return_value=httpx.Response(200, json=json.loads(gen_response.model_dump_json()))
            )

            async with PipelineClient(
                retrieval_base_url="http://retrieval.test",
                generation_base_url="http://generation.test",
            ) as client:
                request = QueryRequest(query="hi")
                chunks = await client.retrieve(request)
                response = await client.generate(request)

        assert len(chunks) == 1
        assert chunks[0].chunk.text == "hello world"
        assert response.answer == "the answer"

    async def test_retrieve_raises_pipeline_response_error_on_5xx(self) -> None:
        from rag_eval.pipeline_client import PipelineResponseError

        with respx.mock(assert_all_called=True) as mock_router:
            mock_router.post("http://retrieval.test/retrieve").mock(
                return_value=httpx.Response(500, text="internal error")
            )

            async with PipelineClient(
                retrieval_base_url="http://retrieval.test",
                generation_base_url="http://generation.test",
            ) as client:
                with pytest.raises(PipelineResponseError):
                    await client.retrieve(QueryRequest(query="hi"))
