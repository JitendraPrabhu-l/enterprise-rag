"""Serve-time guardrail behavior of `GenerationPipeline._generate_and_validate`.

The uncited-answer check (ADR-010/ADR-009): a substantive answer carrying
zero citations must surface as `guardrail_flagged=True` — it's the cheap
serve-time groundedness signal — while a cited answer must pass unflagged.
The Groq generator is mocked; these tests exercise only the validation and
flagging logic downstream of the raw model output.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import structlog
from rag_core.schemas import QueryRequest

from rag_generation.pipeline import GenerationPipeline
from tests.conftest import make_retrieved_chunk


def _pipeline(raw_model_output: str) -> GenerationPipeline:
    generator = MagicMock()
    generator.generate_structured = AsyncMock(return_value=raw_model_output)
    return GenerationPipeline(
        retrieval_client=MagicMock(),
        generator=generator,
        system_prompt="answer with citations",
        compression_target_ratio=0.5,
        generation_model="test-model",
    )


def _log() -> structlog.stdlib.BoundLogger:
    return structlog.get_logger("test")


async def test_answer_without_citations_is_flagged() -> None:
    raw = json.dumps({"answer": "Revenue grew 12% in fiscal 2022.", "citations": []})
    pipeline = _pipeline(raw)

    response = await pipeline._generate_and_validate(
        [{"role": "user", "content": "q"}], QueryRequest(query="q"), False, _log(),
        context_chunks=[],
    )

    assert response.guardrail_flagged is True


async def test_answer_with_citations_is_not_flagged() -> None:
    """The cited parent_id must actually be in context_chunks for this to
    pass unflagged post-ADR-028 — matching make_retrieved_chunk's default
    parent_id ("parent-1") keeps this test's citation grounded."""
    raw = json.dumps(
        {
            "answer": "Revenue grew 12% in fiscal 2022.",
            "citations": [{"parent_id": "parent-1", "document_id": "doc-1", "page_number": 4}],
        }
    )
    pipeline = _pipeline(raw)

    response = await pipeline._generate_and_validate(
        [{"role": "user", "content": "q"}], QueryRequest(query="q"), False, _log(),
        context_chunks=[make_retrieved_chunk("Revenue detail.", parent_id="parent-1")],
    )

    assert response.guardrail_flagged is False


async def test_upstream_flag_is_preserved_alongside_citations(  # noqa: D103
) -> None:
    """A query/chunk-level injection flag must survive even when the answer
    itself is well-cited — the two flags are independent signals."""
    raw = json.dumps(
        {
            "answer": "Grounded answer.",
            "citations": [{"parent_id": "parent-1", "document_id": "doc-1"}],
        }
    )
    pipeline = _pipeline(raw)

    response = await pipeline._generate_and_validate(
        [{"role": "user", "content": "q"}], QueryRequest(query="q"), True, _log(),
        context_chunks=[make_retrieved_chunk("Revenue detail.", parent_id="parent-1")],
    )

    assert response.guardrail_flagged is True


async def test_citation_naming_an_unretrieved_parent_id_is_flagged() -> None:
    """ADR-028: schema-valid, present citation — just pointing at content
    that was never in the context shown to the model. Distinct from the
    zero-citations case above."""
    raw = json.dumps(
        {
            "answer": "Revenue grew 12%, per the fabricated source.",
            "citations": [
                {"parent_id": "never-retrieved", "document_id": "doc-1", "page_number": 1}
            ],
        }
    )
    pipeline = _pipeline(raw)

    response = await pipeline._generate_and_validate(
        [{"role": "user", "content": "q"}], QueryRequest(query="q"), False, _log(),
        context_chunks=[make_retrieved_chunk("Revenue detail.", parent_id="parent-1")],
    )

    assert response.guardrail_flagged is True
