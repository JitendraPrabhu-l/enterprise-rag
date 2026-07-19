"""Tests for rag_eval.judges: the three RAG Triad LLM-as-judge functions.

Mocks AsyncOpenAI (Groq-compatible) chat.completions responses. Verifies each
judge correctly parses a well-formed JSON response into a TriadScore, and
raises clearly (JudgeResponseError) on malformed judge output -- never
silently defaulting to 0.0.
"""

from __future__ import annotations

import pytest

from rag_eval.judges import (
    JudgeResponseError,
    score_answer_relevance,
    score_context_precision,
    score_faithfulness,
)
from rag_eval.schemas import TriadScore
from tests.conftest import (
    FakeAsyncOpenAI,
    make_empty_choices_message,
    make_no_content_message,
    make_raw_message,
    make_score_message,
)

_MODEL = "openai/gpt-oss-120b"


class TestScoreFaithfulness:
    async def test_parses_well_formed_response(self) -> None:
        client = FakeAsyncOpenAI(
            [make_score_message(0.9, "All claims are supported by the context.")]
        )

        result = await score_faithfulness(
            client,  # type: ignore[arg-type]
            model=_MODEL,
            answer="Paris is the capital of France.",
            context=["France's capital is Paris."],
        )

        assert isinstance(result, TriadScore)
        assert result.score == pytest.approx(0.9)
        assert result.justification == "All claims are supported by the context."

    async def test_uses_json_object_response_format(self) -> None:
        client = FakeAsyncOpenAI([make_score_message(1.0, "Fully grounded.")])

        await score_faithfulness(
            client,  # type: ignore[arg-type]
            model=_MODEL,
            answer="Some answer.",
            context=["Some context."],
        )

        call = client.chat.completions.calls[0]
        assert call["model"] == _MODEL
        assert call["response_format"] == {"type": "json_object"}
        # the prompt must explicitly describe the expected JSON shape, since
        # basic JSON mode does not enforce a schema server-side.
        assert '"score"' in call["messages"][0]["content"]
        assert '"justification"' in call["messages"][0]["content"]

    async def test_raises_on_malformed_json(self) -> None:
        client = FakeAsyncOpenAI([make_raw_message("not valid json at all")])

        with pytest.raises(JudgeResponseError):
            await score_faithfulness(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                answer="answer",
                context=["context"],
            )

    async def test_raises_on_missing_required_field(self) -> None:
        client = FakeAsyncOpenAI([make_raw_message('{"score": 0.5}')])

        with pytest.raises(JudgeResponseError):
            await score_faithfulness(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                answer="answer",
                context=["context"],
            )

    async def test_raises_on_out_of_range_score(self) -> None:
        client = FakeAsyncOpenAI([make_raw_message('{"score": 1.5, "justification": "bad"}')])

        with pytest.raises(JudgeResponseError):
            await score_faithfulness(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                answer="answer",
                context=["context"],
            )

    async def test_raises_on_empty_choices(self) -> None:
        client = FakeAsyncOpenAI([make_empty_choices_message()])

        with pytest.raises(JudgeResponseError):
            await score_faithfulness(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                answer="answer",
                context=["context"],
            )

    async def test_raises_on_none_content(self) -> None:
        client = FakeAsyncOpenAI([make_no_content_message()])

        with pytest.raises(JudgeResponseError):
            await score_faithfulness(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                answer="answer",
                context=["context"],
            )

    async def test_does_not_default_to_zero_on_failure(self) -> None:
        """A malformed judge response must raise, never silently become score=0.0."""
        client = FakeAsyncOpenAI([make_raw_message("garbage")])

        try:
            await score_faithfulness(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                answer="answer",
                context=["context"],
            )
            pytest.fail("expected JudgeResponseError to be raised")
        except JudgeResponseError:
            pass  # correct: failure is surfaced, not swallowed into a default score


class TestScoreAnswerRelevance:
    async def test_parses_well_formed_response(self) -> None:
        client = FakeAsyncOpenAI([make_score_message(0.75, "Mostly addresses the question.")])

        result = await score_answer_relevance(
            client,  # type: ignore[arg-type]
            model=_MODEL,
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
        )

        assert result.score == pytest.approx(0.75)
        assert result.justification == "Mostly addresses the question."

    async def test_raises_on_malformed_json(self) -> None:
        client = FakeAsyncOpenAI([make_raw_message("{broken json")])

        with pytest.raises(JudgeResponseError):
            await score_answer_relevance(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                question="q",
                answer="a",
            )


class TestScoreContextPrecision:
    async def test_parses_well_formed_response(self) -> None:
        client = FakeAsyncOpenAI([make_score_message(0.5, "Chunk 2 was irrelevant.")])

        result = await score_context_precision(
            client,  # type: ignore[arg-type]
            model=_MODEL,
            question="What is the capital of France?",
            retrieved_chunks=["Paris is the capital of France.", "Bananas are yellow."],
        )

        assert result.score == pytest.approx(0.5)
        assert result.justification == "Chunk 2 was irrelevant."

    async def test_handles_empty_chunk_list_prompt_formatting(self) -> None:
        client = FakeAsyncOpenAI([make_score_message(0.0, "No chunks retrieved.")])

        result = await score_context_precision(
            client,  # type: ignore[arg-type]
            model=_MODEL,
            question="q",
            retrieved_chunks=[],
        )

        assert result.score == pytest.approx(0.0)
        call = client.chat.completions.calls[0]
        assert "no chunks were retrieved" in call["messages"][0]["content"]

    async def test_raises_on_wrong_type_for_justification(self) -> None:
        client = FakeAsyncOpenAI([make_raw_message('{"score": 0.5, "justification": 123}')])

        with pytest.raises(JudgeResponseError):
            await score_context_precision(
                client,  # type: ignore[arg-type]
                model=_MODEL,
                question="q",
                retrieved_chunks=["c"],
            )
