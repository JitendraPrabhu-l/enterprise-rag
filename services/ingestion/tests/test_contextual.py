"""Tests for `ContextualEnricher` (ADR-023 contextual retrieval).

The Groq client is faked with a stand-in exposing exactly the
`chat.completions.create(...)` surface `ContextualEnricher` calls, returning
a scripted `.choices[0].message.content` — the real HTTP/Groq wiring is
covered by ADR-012's existing generation-service tests; what's under test
here is the enrichment contract: which chunks get a prefix, what
`searchable_text` becomes once they do, and — the ingestion-reliability
property this whole module exists to guarantee — that a per-chunk LLM
failure degrades to the raw chunk rather than failing the ingest job.
"""

from __future__ import annotations

import pytest
from rag_core.schemas import (
    AccessRole,
    ChunkRecord,
    ContentModality,
    DocumentMetadata,
    ParentContext,
    SourceType,
)

from rag_ingestion.contextual import ContextualEnricher


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        source_type=SourceType.PDF,
        source_domain="test-domain",
        tenant_id="tenant-a",
        access_role=AccessRole.INTERNAL,
        last_updated_epoch=1_700_000_000,
    )


def _parent(parent_id: str = "doc-1:p0") -> ParentContext:
    return ParentContext(
        parent_id=parent_id,
        document_id="doc-1",
        text="Acme Corp's Q3 2024 filing discusses liquidity and cash reserves in detail.",
        page_number=3,
    )


def _chunk(
    chunk_id: str = "c-1", parent_id: str = "doc-1:p0", text: str = "Cash reserves grew 12%."
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        parent_id=parent_id,
        document_id="doc-1",
        text=text,
        modality=ContentModality.PROSE,
        token_count=5,
        metadata=_metadata(),
    )


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _ScriptedCompletions:
    """Returns each scripted reply in call order; a callable entry raises
    that callable's exception instead (models a transient failure)."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return _FakeCompletionResponse(reply)


class _ScriptedChat:
    def __init__(self, replies: list) -> None:
        self.completions = _ScriptedCompletions(replies)


class _ScriptedClient:
    def __init__(self, replies: list) -> None:
        self.chat = _ScriptedChat(replies)


@pytest.mark.asyncio
class TestEnrichHappyPath:
    async def test_successful_call_sets_context_prefix(self) -> None:
        client = _ScriptedClient(["From Acme Corp's Q3 2024 filing, discussing cash reserves."])
        enricher = ContextualEnricher(client, model="test-utility-model")

        chunk = _chunk()
        result = await enricher.enrich(
            [chunk], {chunk.parent_id: _parent()}, document_title="Acme 10-Q"
        )

        assert len(result) == 1
        expected_prefix = "From Acme Corp's Q3 2024 filing, discussing cash reserves."
        assert result[0].context_prefix == expected_prefix

    async def test_searchable_text_combines_prefix_and_raw_text(self) -> None:
        client = _ScriptedClient(["Situating sentence."])
        enricher = ContextualEnricher(client, model="test-utility-model")

        chunk = _chunk(text="Cash reserves grew 12%.")
        [enriched] = await enricher.enrich(
            [chunk], {chunk.parent_id: _parent()}, document_title=None
        )

        assert enriched.searchable_text == "Situating sentence.\nCash reserves grew 12%."
        # The raw .text is untouched — the generator must always see the
        # original passage, never the situating prefix baked in permanently.
        assert enriched.text == "Cash reserves grew 12%."

    async def test_preserves_order_and_count_across_multiple_chunks(self) -> None:
        client = _ScriptedClient(["ctx-1", "ctx-2", "ctx-3"])
        enricher = ContextualEnricher(client, model="test-utility-model")

        chunks = [_chunk(f"c-{i}", text=f"text {i}") for i in range(3)]
        parents = {c.parent_id: _parent() for c in chunks}
        result = await enricher.enrich(chunks, parents, document_title=None)

        assert [c.chunk_id for c in result] == ["c-0", "c-1", "c-2"]

    async def test_document_title_is_included_in_the_prompt(self) -> None:
        client = _ScriptedClient(["prefix"])
        enricher = ContextualEnricher(client, model="test-utility-model")
        chunk = _chunk()

        await enricher.enrich(
            [chunk], {chunk.parent_id: _parent()}, document_title="Acme 10-Q 2024"
        )

        sent_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
        assert "Acme 10-Q 2024" in sent_prompt

    async def test_empty_chunk_list_is_a_noop(self) -> None:
        client = _ScriptedClient(["should never be used"])
        enricher = ContextualEnricher(client, model="test-utility-model")

        result = await enricher.enrich([], {}, document_title=None)

        assert result == []
        assert client.chat.completions.calls == []


@pytest.mark.asyncio
class TestEnrichFailureFallsBackGracefully:
    """The ingestion-reliability property: enrichment failures must degrade
    to the raw chunk, never fail the ingest job — enrichment is an
    optimization layered on top of indexing, not a precondition for it."""

    async def test_api_error_falls_back_to_unenriched_chunk(self) -> None:
        client = _ScriptedClient([RuntimeError("Groq API unavailable")])
        enricher = ContextualEnricher(client, model="test-utility-model")
        chunk = _chunk()

        [result] = await enricher.enrich([chunk], {chunk.parent_id: _parent()}, document_title=None)

        assert result.context_prefix is None
        assert result.searchable_text == chunk.text  # degrades to raw text

    async def test_blank_response_falls_back_to_unenriched_chunk(self) -> None:
        client = _ScriptedClient(["   "])  # whitespace-only "content"
        enricher = ContextualEnricher(client, model="test-utility-model")
        chunk = _chunk()

        [result] = await enricher.enrich([chunk], {chunk.parent_id: _parent()}, document_title=None)

        assert result.context_prefix is None

    async def test_none_content_falls_back_to_unenriched_chunk(self) -> None:
        """Some providers return message.content=None for certain finish
        reasons (e.g. content filtering) — must not crash on .strip()."""
        client = _ScriptedClient([None])
        enricher = ContextualEnricher(client, model="test-utility-model")
        chunk = _chunk()

        [result] = await enricher.enrich([chunk], {chunk.parent_id: _parent()}, document_title=None)

        assert result.context_prefix is None

    async def test_one_chunk_failing_does_not_affect_others(self) -> None:
        """Failures are per-chunk, not per-batch — one bad call must not
        poison the whole document's enrichment."""
        client = _ScriptedClient(["ok-1", RuntimeError("transient"), "ok-3"])
        enricher = ContextualEnricher(client, model="test-utility-model")
        chunks = [_chunk(f"c-{i}", text=f"text {i}") for i in range(3)]
        parents = {c.parent_id: _parent() for c in chunks}

        result = await enricher.enrich(chunks, parents, document_title=None)

        assert result[0].context_prefix == "ok-1"
        assert result[1].context_prefix is None  # the failed one
        assert result[2].context_prefix == "ok-3"

    async def test_missing_parent_does_not_crash(self) -> None:
        """A chunk whose parent_id isn't in the parents map (should not
        happen in practice, but must not crash enrichment if it does) still
        gets a call with empty parent context rather than raising."""
        client = _ScriptedClient(["fallback prefix"])
        enricher = ContextualEnricher(client, model="test-utility-model")
        chunk = _chunk(parent_id="nonexistent-parent")

        [result] = await enricher.enrich([chunk], {}, document_title=None)

        assert result.context_prefix == "fallback prefix"
