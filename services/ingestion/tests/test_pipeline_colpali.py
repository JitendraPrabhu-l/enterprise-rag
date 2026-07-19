"""Tests for ADR-029's wiring in `IngestionPipeline._resolve_page_texts`:
a table/figure-dense page gets the SAME rendered image handed to BOTH the
existing vision-description text path AND (when configured) the ColPali
index — additive, never a replacement — and a ColPali failure must not
affect the text path that already succeeded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import rag_ingestion.pipeline as pipeline_module
from rag_ingestion.config import IngestionSettings
from rag_ingestion.page_classifier import PageCategory
from rag_ingestion.parsing import PageLayout, ParsedDocument, ParsedPage
from rag_ingestion.pipeline import IngestionPipeline, IngestRequest


def _dense_page(page_number: int = 1) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        text="fallback text",
        layout=PageLayout(line_count=1, avg_line_length=1.0, non_alpha_ratio=0.5, char_count=10),
    )


def _pipeline(
    *, colpali_index=None, vision_describer=None
) -> IngestionPipeline:
    classifier = MagicMock()
    classifier.classify.return_value = PageCategory.TABLE_OR_FIGURE_DENSE
    return IngestionPipeline(
        settings=IngestionSettings(),
        page_classifier=classifier,
        vision_describer=vision_describer or _describer("a description"),
        embedder=MagicMock(),
        vector_store=MagicMock(),
        sparse_indexer=MagicMock(),
        colpali_index=colpali_index,
    )


def _describer(text: str) -> MagicMock:
    describer = MagicMock()
    describer.describe_page = AsyncMock(return_value=text)
    return describer


def _request() -> IngestRequest:
    return IngestRequest(
        document_id="doc-1",
        file_path="/tmp/x.pdf",
        source_domain="sec-filings",
        tenant_id="tenant-a",
    )


@pytest.mark.asyncio
class TestResolvePageTextsColpaliWiring:
    async def test_colpali_index_page_is_called_with_the_same_rendered_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pipeline_module, "_render_page_to_png", lambda path, num: b"rendered-png-bytes"
        )
        colpali = MagicMock()
        colpali.index_page = AsyncMock(return_value=True)
        pipeline = _pipeline(colpali_index=colpali)
        parsed = ParsedDocument(
            document_id="doc-1", source_path="/tmp/x.pdf", pages=[_dense_page(3)]
        )
        request = _request()

        await pipeline._resolve_page_texts(parsed, request, _log())

        colpali.index_page.assert_awaited_once_with(
            source_domain="sec-filings",
            document_id="doc-1",
            page_number=3,
            image_bytes=b"rendered-png-bytes",
        )

    async def test_no_colpali_index_configured_is_a_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default (colpali_index=None) must behave exactly like before
        ADR-029 existed - no call, no crash."""
        monkeypatch.setattr(
            pipeline_module, "_render_page_to_png", lambda path, num: b"rendered-png-bytes"
        )
        pipeline = _pipeline(colpali_index=None)
        parsed = ParsedDocument(
            document_id="doc-1", source_path="/tmp/x.pdf", pages=[_dense_page(1)]
        )
        request = _request()

        page_texts, describe_calls = await pipeline._resolve_page_texts(parsed, request, _log())

        assert describe_calls == 1
        assert page_texts == [(1, "a description")]

    async def test_colpali_failure_does_not_affect_the_text_description_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The load-bearing reliability property: the vision-description
        text path (already indexed the same page successfully) must not be
        undone or blocked by a ColPali failure on that same page."""
        monkeypatch.setattr(
            pipeline_module, "_render_page_to_png", lambda path, num: b"rendered-png-bytes"
        )
        colpali = MagicMock()
        colpali.index_page = AsyncMock(side_effect=RuntimeError("colpali crashed"))
        pipeline = _pipeline(
            colpali_index=colpali, vision_describer=_describer("good description")
        )
        parsed = ParsedDocument(
            document_id="doc-1", source_path="/tmp/x.pdf", pages=[_dense_page(1)]
        )
        request = _request()

        # Must not raise, and the text path's result must be intact.
        page_texts, describe_calls = await pipeline._resolve_page_texts(parsed, request, _log())

        assert describe_calls == 1
        assert page_texts == [(1, "good description")]


def _log():
    import structlog

    return structlog.get_logger("test")
