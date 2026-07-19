"""Tests for `RetrievalPipeline._expand_query`'s strategy dispatch (ADR-025):
confirms "decompose" is wired to `QueryExpander.decompose` exactly the way
"multi_query" is wired to `expand_multi_query` — same return shape
(query_texts, no dense-vector override), so it flows into the existing
per-variant hybrid-search fan-out unchanged. `ALL_STRATEGIES` inclusion is
what the API layer's Literal validates callers against.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from rag_core.schemas import QueryRequest
from rag_core.tracing import get_tracer

from rag_retrieval.pipeline import ALL_STRATEGIES, RetrievalPipeline


def _pipeline(query_expander: MagicMock) -> RetrievalPipeline:
    return RetrievalPipeline(
        vector_store=MagicMock(),
        sparse_client=MagicMock(),
        embedder=MagicMock(),
        reranker=MagicMock(),
        query_expander=query_expander,
        graph_client=None,
        rrf_k=60,
        tracer=get_tracer("test"),
    )


class TestAllStrategiesIncludesDecompose:
    def test_decompose_is_a_recognized_strategy(self) -> None:
        assert "decompose" in ALL_STRATEGIES


@pytest.mark.asyncio
class TestExpandQueryDispatchesDecompose:
    async def test_decompose_strategy_calls_query_expander_decompose(self) -> None:
        expander = MagicMock()
        expander.decompose = AsyncMock(
            return_value=["original query", "sub-question A", "sub-question B"]
        )
        pipeline = _pipeline(expander)
        request = QueryRequest(query="original query")

        query_texts, vector_override = await pipeline._expand_query(request, "decompose")

        expander.decompose.assert_awaited_once_with("original query")
        assert query_texts == ["original query", "sub-question A", "sub-question B"]

    async def test_decompose_never_overrides_the_dense_vector(self) -> None:
        """Unlike hyde, decompose reuses the ordinary per-variant embed step
        (each sub-question gets its own real embedding) — it must not
        return a vector override, or every sub-question would search using
        the SAME embedding as the original query, defeating the point."""
        expander = MagicMock()
        expander.decompose = AsyncMock(return_value=["q", "sub-q"])
        pipeline = _pipeline(expander)
        request = QueryRequest(query="q")

        _texts, vector_override = await pipeline._expand_query(request, "decompose")

        assert vector_override is None

    async def test_single_fact_query_decompose_result_is_passed_through_unchanged(self) -> None:
        """A single-fact query where decompose() itself already collapsed to
        a no-op ([query] only) must flow through the pipeline exactly like
        "direct" — one variant, one hybrid search, no wasted extra calls
        downstream."""
        expander = MagicMock()
        expander.decompose = AsyncMock(return_value=["simple lookup"])
        pipeline = _pipeline(expander)
        request = QueryRequest(query="simple lookup")

        query_texts, _ = await pipeline._expand_query(request, "decompose")

        assert query_texts == ["simple lookup"]
