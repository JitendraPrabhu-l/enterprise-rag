"""Tests for `IngestionPipeline._maybe_enable_quantization` (ADR-003): the
ingest-time call site for `VectorStore.enable_quantization_if_due`. Only the
pipeline's OWN contract is under test here — that it calls the vector
store with the right domain, and that a failure there is swallowed rather
than propagated (an optimization must never fail an otherwise-successful
ingest job). `VectorStore`'s own decision logic (threshold/already-quantized/
missing-collection) is covered in packages/rag_core's
test_vector_store_quantization.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog

from rag_ingestion.config import IngestionSettings
from rag_ingestion.page_classifier import HeuristicPageClassifier
from rag_ingestion.pipeline import IngestionPipeline


def _pipeline(vector_store: MagicMock) -> IngestionPipeline:
    return IngestionPipeline(
        settings=IngestionSettings(),
        page_classifier=HeuristicPageClassifier(),
        vision_describer=MagicMock(),
        embedder=MagicMock(),
        vector_store=vector_store,
        sparse_indexer=MagicMock(),
    )


def _log() -> structlog.stdlib.BoundLogger:
    return structlog.get_logger("test")


@pytest.mark.asyncio
class TestMaybeEnableQuantization:
    async def test_calls_vector_store_with_the_right_domain(self) -> None:
        vector_store = MagicMock()
        vector_store.enable_quantization_if_due = AsyncMock(return_value=False)
        pipeline = _pipeline(vector_store)

        await pipeline._maybe_enable_quantization("sec-filings", _log())

        vector_store.enable_quantization_if_due.assert_awaited_once_with("sec-filings")

    async def test_a_failure_is_swallowed_not_propagated(self) -> None:
        """The load-bearing property: an ingest job that successfully
        upserted and sparse-indexed every chunk must still report success
        even if the quantization check itself blows up (network blip
        against Qdrant, etc.) - this is an optimization, not a correctness
        dependency, so it must never turn a good ingest into a failed one."""
        vector_store = MagicMock()
        vector_store.enable_quantization_if_due = AsyncMock(
            side_effect=RuntimeError("qdrant unreachable")
        )
        pipeline = _pipeline(vector_store)

        # Must not raise.
        await pipeline._maybe_enable_quantization("sec-filings", _log())

    async def test_true_result_does_not_raise_or_change_behavior(self) -> None:
        vector_store = MagicMock()
        vector_store.enable_quantization_if_due = AsyncMock(return_value=True)
        pipeline = _pipeline(vector_store)

        await pipeline._maybe_enable_quantization("sec-filings", _log())

        vector_store.enable_quantization_if_due.assert_awaited_once()
