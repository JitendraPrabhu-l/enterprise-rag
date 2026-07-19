"""Tests for `GenerationPipeline.run`'s semantic-cache integration (ADR-026):
a cache hit must skip retrieval + generation entirely and return the stored
answer re-stamped with the CALLER's request_id; a miss must run the normal
pipeline and then store the result — except when the answer is guardrail-
flagged, which must never be cached (see rag_core's own
test_semantic_cache.py for the cache module's own contract).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from rag_core.schemas import Citation, GenerationResponse, QueryRequest

from rag_generation.pipeline import GenerationPipeline
from tests.conftest import make_retrieved_chunk


def _cached_response() -> GenerationResponse:
    return GenerationResponse(
        request_id=uuid4(),  # a DIFFERENT request's id — the point under test
        answer="Cached answer text.",
        citations=[Citation(parent_id="p1", document_id="d1", page_number=2)],
        model="test-model",
        guardrail_flagged=False,
    )


def _pipeline(
    *,
    cache_hit: GenerationResponse | None,
    raw_model_output: str | None = None,
    retrieved_chunks: list | None = None,
) -> tuple[GenerationPipeline, MagicMock, MagicMock, MagicMock]:
    retrieval_client = MagicMock()
    retrieval_client.retrieve = AsyncMock(return_value=retrieved_chunks or [])

    generator = MagicMock()
    generator.generate_structured = AsyncMock(return_value=raw_model_output)

    cache = MagicMock()
    cache.lookup = AsyncMock(return_value=cache_hit)
    cache.store = AsyncMock()

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])

    pipeline = GenerationPipeline(
        retrieval_client=retrieval_client,
        generator=generator,
        system_prompt="answer with citations",
        compression_target_ratio=0.5,
        generation_model="test-model",
        semantic_cache=cache,
        cache_embedder=embedder,
    )
    return pipeline, cache, retrieval_client, generator


class TestCacheHitSkipsThePipeline:
    async def test_hit_returns_cached_answer_without_calling_retrieval(self) -> None:
        cached = _cached_response()
        pipeline, _cache, retrieval_client, _generator = _pipeline(cache_hit=cached)

        response = await pipeline.run(QueryRequest(query="what was revenue?"))

        assert response.answer == cached.answer
        retrieval_client.retrieve.assert_not_awaited()

    async def test_hit_returns_cached_answer_without_calling_generator(self) -> None:
        cached = _cached_response()
        pipeline, _cache, _retrieval_client, generator = _pipeline(cache_hit=cached)

        await pipeline.run(QueryRequest(query="what was revenue?"))

        generator.generate_structured.assert_not_awaited()

    async def test_hit_response_is_restamped_with_the_new_requests_id(self) -> None:
        """The cached GenerationResponse carries the ORIGINAL request's id —
        returning it verbatim would silently mislabel which request this
        answer belongs to (matters for tracing/audit correlation)."""
        cached = _cached_response()
        pipeline, _cache, _retrieval_client, _generator = _pipeline(cache_hit=cached)
        request = QueryRequest(query="what was revenue?")

        response = await pipeline.run(request)

        assert response.request_id == request.request_id
        assert response.request_id != cached.request_id

    async def test_hit_preserves_the_cached_citations(self) -> None:
        cached = _cached_response()
        pipeline, _cache, _retrieval_client, _generator = _pipeline(cache_hit=cached)

        response = await pipeline.run(QueryRequest(query="what was revenue?"))

        assert response.citations == cached.citations


class TestCacheMissRunsPipelineThenStores:
    async def test_miss_runs_retrieval_and_generation_normally(self) -> None:
        raw = json.dumps({"answer": "Fresh answer.", "citations": []})
        pipeline, _cache, retrieval_client, generator = _pipeline(
            cache_hit=None, raw_model_output=raw
        )

        response = await pipeline.run(QueryRequest(query="what was revenue?"))

        retrieval_client.retrieve.assert_awaited_once()
        generator.generate_structured.assert_awaited_once()
        assert response.answer == "Fresh answer."

    async def test_miss_stores_the_fresh_answer_in_the_cache(self) -> None:
        raw = json.dumps(
            {
                "answer": "Fresh answer.",
                "citations": [{"parent_id": "p1", "document_id": "d1", "page_number": 1}],
            }
        )
        pipeline, cache, _retrieval_client, _generator = _pipeline(
            cache_hit=None,
            raw_model_output=raw,
            # Citation "p1" must be grounded (ADR-028) for this answer to
            # pass unflagged and therefore be cache-eligible at all.
            retrieved_chunks=[
                make_retrieved_chunk("Revenue detail.", parent_id="p1", document_id="d1")
            ],
        )

        await pipeline.run(QueryRequest(query="what was revenue?"))

        cache.store.assert_awaited_once()
        _, kwargs = cache.store.await_args
        assert kwargs["response"].answer == "Fresh answer."

    async def test_guardrail_flagged_answer_is_never_stored(self) -> None:
        """An uncited/flagged answer must not be cached and later replayed
        to a different caller as if it were trustworthy (ADR-026's own
        store() also refuses this — this proves the pipeline actually calls
        store() with the flagged response rather than skipping the call
        entirely in a way that would mask a regression there)."""
        raw = json.dumps({"answer": "Ungrounded answer.", "citations": []})
        pipeline, cache, _retrieval_client, _generator = _pipeline(
            cache_hit=None, raw_model_output=raw
        )

        response = await pipeline.run(QueryRequest(query="what was revenue?"))

        assert response.guardrail_flagged is True
        cache.store.assert_not_awaited()


class TestCacheDisabledBehavesAsBefore:
    async def test_no_cache_configured_runs_pipeline_normally(self) -> None:
        """semantic_cache=None (the default) must behave exactly like
        before ADR-026 existed — no lookup, no embedding call, no crash."""
        raw = json.dumps({"answer": "Answer.", "citations": []})
        retrieval_client = MagicMock()
        retrieval_client.retrieve = AsyncMock(return_value=[])
        generator = MagicMock()
        generator.generate_structured = AsyncMock(return_value=raw)

        pipeline = GenerationPipeline(
            retrieval_client=retrieval_client,
            generator=generator,
            system_prompt="answer with citations",
            compression_target_ratio=0.5,
            generation_model="test-model",
        )

        response = await pipeline.run(QueryRequest(query="q"))

        retrieval_client.retrieve.assert_awaited_once()
        assert response.answer == "Answer."
