"""Tests for `VectorStore.enable_quantization_if_due` (ADR-003 quantization
wiring). Qdrant fixes a collection's quantization config at creation time,
so this is a deliberately separate, idempotent, call-anytime operation from
`ensure_collection` — see the method's own docstring for why a domain's
FIRST ingest (when `ensure_collection` actually runs) can never have a real
vector count to self-decide from.

`AsyncQdrantClient` itself is mocked; these tests assert only the decision
logic (collection missing / already quantized / under threshold / due) and
that `update_collection` is called with the right shape when due.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from rag_core.vector_store import QUANTIZATION_THRESHOLD_VECTORS, VectorStore


def _collection_info(*, points_count: int, quantized: bool) -> MagicMock:
    info = MagicMock()
    info.points_count = points_count
    info.config.quantization_config = (
        MagicMock(spec=models.ScalarQuantization) if quantized else None
    )
    return info


@pytest.mark.asyncio
class TestEnableQuantizationIfDue:
    async def test_missing_collection_is_a_noop(self) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        with patch.object(store._client, "collection_exists", new=AsyncMock(return_value=False)):
            result = await store.enable_quantization_if_due("sec-filings")
        assert result is False

    async def test_under_threshold_is_a_noop(self) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        info = _collection_info(points_count=QUANTIZATION_THRESHOLD_VECTORS - 1, quantized=False)
        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=True)),
            patch.object(store._client, "get_collection", new=AsyncMock(return_value=info)),
            patch.object(store._client, "update_collection", new=AsyncMock()) as mock_update,
        ):
            result = await store.enable_quantization_if_due("sec-filings")
        assert result is False
        mock_update.assert_not_awaited()

    async def test_already_quantized_is_a_noop_even_if_over_threshold(self) -> None:
        """Must not re-issue update_collection every time this is called on
        an already-quantized domain — that would be a wasted API call on
        every single ingest job for that domain's entire remaining life."""
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        info = _collection_info(points_count=QUANTIZATION_THRESHOLD_VECTORS + 1, quantized=True)
        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=True)),
            patch.object(store._client, "get_collection", new=AsyncMock(return_value=info)),
            patch.object(store._client, "update_collection", new=AsyncMock()) as mock_update,
        ):
            result = await store.enable_quantization_if_due("sec-filings")
        assert result is False
        mock_update.assert_not_awaited()

    async def test_over_threshold_and_unquantized_enables_quantization(self) -> None:
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        info = _collection_info(points_count=QUANTIZATION_THRESHOLD_VECTORS + 1, quantized=False)
        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=True)),
            patch.object(store._client, "get_collection", new=AsyncMock(return_value=info)),
            patch.object(store._client, "update_collection", new=AsyncMock()) as mock_update,
        ):
            result = await store.enable_quantization_if_due("sec-filings")

        assert result is True
        mock_update.assert_awaited_once()
        _, kwargs = mock_update.await_args
        assert kwargs["collection_name"] == "rag_sec-filings"
        assert isinstance(kwargs["quantization_config"], models.ScalarQuantization)
        assert kwargs["quantization_config"].scalar.type == models.ScalarType.INT8

    async def test_exactly_at_threshold_enables_quantization(self) -> None:
        """>= threshold, not > — a domain landing exactly on the boundary
        should not have to wait for one more vector to get quantized."""
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        info = _collection_info(points_count=QUANTIZATION_THRESHOLD_VECTORS, quantized=False)
        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=True)),
            patch.object(store._client, "get_collection", new=AsyncMock(return_value=info)),
            patch.object(store._client, "update_collection", new=AsyncMock()) as mock_update,
        ):
            result = await store.enable_quantization_if_due("sec-filings")
        assert result is True
        mock_update.assert_awaited_once()

    async def test_points_count_none_is_treated_as_zero_not_a_crash(self) -> None:
        """Qdrant's points_count can be None for a brand-new/empty
        collection depending on server version — must not crash comparing
        None >= threshold."""
        store = VectorStore(url="http://qdrant:6333", api_key=None, embedding_dim=3)
        info = _collection_info(points_count=0, quantized=False)
        info.points_count = None
        with (
            patch.object(store._client, "collection_exists", new=AsyncMock(return_value=True)),
            patch.object(store._client, "get_collection", new=AsyncMock(return_value=info)),
            patch.object(store._client, "update_collection", new=AsyncMock()) as mock_update,
        ):
            result = await store.enable_quantization_if_due("sec-filings")
        assert result is False
        mock_update.assert_not_awaited()
