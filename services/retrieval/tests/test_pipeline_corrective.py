"""Tests for the CRAG-style corrective retrieval loop (ADR-038).

The loop wraps `_retrieve_once`: run one pass, grade the top result's rerank
score against a floor, and if insufficient escalate the query strategy and
retry (bounded, keeping the best result). These tests mock `_retrieve_once`
so the loop's decision logic — grade, escalate, cap, best-of-N — is exercised
in isolation from the real hybrid-search machinery.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from rag_core.schemas import (
    ChunkRecord,
    DocumentMetadata,
    ParentContext,
    QueryRequest,
    RetrievedChunk,
    SourceType,
)
from rag_core.tracing import get_tracer

from rag_retrieval.pipeline import RetrievalPipeline


def _result(rerank_score: float | None, chunk_id: str = "c1") -> list[RetrievedChunk]:
    metadata = DocumentMetadata(
        document_id="d1", source_type=SourceType.PDF, source_domain="d", last_updated_epoch=1
    )
    parent = ParentContext(parent_id="p1", document_id="d1", text="body")
    chunk = ChunkRecord(
        chunk_id=chunk_id, parent_id="p1", document_id="d1", text="body", token_count=1,
        metadata=metadata,
    )
    return [RetrievedChunk(chunk=chunk, parent=parent, rerank_score=rerank_score)]


def _pipeline(
    *, enabled: bool, floor: float = 0.0, max_retries: int = 2
) -> RetrievalPipeline:
    return RetrievalPipeline(
        vector_store=MagicMock(),
        sparse_client=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        query_expander=MagicMock(),
        graph_client=None,
        rrf_k=60,
        tracer=get_tracer("test"),
        corrective_enabled=enabled,
        corrective_confidence_floor=floor,
        corrective_max_retries=max_retries,
    )


@pytest.mark.asyncio
class TestCorrectiveLoopDisabled:
    async def test_disabled_runs_exactly_one_pass(self) -> None:
        """With the loop off, retrieve == one _retrieve_once call, even on a
        weak result — the pre-ADR-038 behavior."""
        pipeline = _pipeline(enabled=False)
        pipeline._retrieve_once = AsyncMock(return_value=_result(-5.0))  # type: ignore[method-assign]

        await pipeline.retrieve(QueryRequest(query="q"), query_strategy="direct")

        pipeline._retrieve_once.assert_awaited_once()


@pytest.mark.asyncio
class TestCorrectiveLoopEnabled:
    async def test_sufficient_first_pass_does_not_retry(self) -> None:
        """A top score at/above the floor is sufficient — no escalation."""
        pipeline = _pipeline(enabled=True, floor=0.0)
        pipeline._retrieve_once = AsyncMock(return_value=_result(1.5))  # type: ignore[method-assign]

        await pipeline.retrieve(QueryRequest(query="q"), query_strategy="direct")

        pipeline._retrieve_once.assert_awaited_once()

    async def test_weak_first_pass_triggers_escalation(self) -> None:
        """A below-floor top score escalates to the next ladder strategy; the
        second pass being sufficient stops the loop."""
        pipeline = _pipeline(enabled=True, floor=0.0)
        pipeline._retrieve_once = AsyncMock(  # type: ignore[method-assign]
            side_effect=[_result(-2.0), _result(3.0)]
        )

        result = await pipeline.retrieve(QueryRequest(query="q"), query_strategy="direct")

        assert pipeline._retrieve_once.await_count == 2
        # Second call escalated off "direct" to the next ladder rung.
        second_call_strategy = pipeline._retrieve_once.await_args_list[1].args[1]
        assert second_call_strategy == "multi_query"
        assert result[0].rerank_score == 3.0

    async def test_retries_are_capped(self) -> None:
        """All passes weak → the loop bounds at max_retries+1 total passes and
        returns the best-scoring result seen, never loops forever."""
        pipeline = _pipeline(enabled=True, floor=0.0, max_retries=2)
        pipeline._retrieve_once = AsyncMock(  # type: ignore[method-assign]
            side_effect=[_result(-3.0), _result(-1.0), _result(-2.0)]
        )

        result = await pipeline.retrieve(QueryRequest(query="q"), query_strategy="direct")

        # 1 initial + 2 corrective retries = 3 passes, then stop.
        assert pipeline._retrieve_once.await_count == 3
        # Best-of-N: the -1.0 pass is the highest, so it's returned.
        assert result[0].rerank_score == -1.0

    async def test_empty_result_is_insufficient_and_triggers_retry(self) -> None:
        pipeline = _pipeline(enabled=True, floor=0.0)
        pipeline._retrieve_once = AsyncMock(  # type: ignore[method-assign]
            side_effect=[[], _result(2.0)]
        )

        result = await pipeline.retrieve(QueryRequest(query="q"), query_strategy="direct")

        assert pipeline._retrieve_once.await_count == 2
        assert result[0].rerank_score == 2.0
