"""Orchestrates the full generation pipeline (ADR-007, ADR-008, ADR-010):

1. Call the retrieval service (`retrieval_client.RetrievalClient.retrieve`).
2. Guardrail-scan the user's query AND every retrieved chunk's text
   (`guardrails.scan_for_injection`) — flagged chunks are excluded from the
   prompt before it is ever built, never merely logged.
3. Compress the remaining (unflagged) chunks' parent-context text
   (`compression.compress_context`).
4. Build the chat `messages` array (`prompt_builder.build_prompt`).
5. Generate the structured answer (`generation.GroqGenerator`).
6. Validate the output (`guardrails.validate_output`); retry once with a
   stricter instruction on parse failure; raise cleanly if it still fails.
7. Return a `GenerationResponse`, with `guardrail_flagged=True` if the query
   itself or any chunk was filtered.
"""

from __future__ import annotations

import structlog
from rag_core.metrics import GUARDRAIL_FLAGS, SEMANTIC_CACHE
from rag_core.schemas import GenerationResponse, QueryRequest, RetrievedChunk
from rag_core.semantic_cache import SemanticAnswerCache
from rag_core.tracing import get_tracer

from rag_generation.cache_embedder import CacheKeyEmbedder
from rag_generation.compression import compress_context
from rag_generation.generation import GroqGenerator
from rag_generation.guardrails import (
    OutputValidationError,
    build_retry_instruction,
    find_ungrounded_citations,
    scan_for_injection,
    validate_output,
)
from rag_generation.prompt_builder import ChatMessage, build_prompt
from rag_generation.retrieval_client import RetrievalClient

logger = structlog.get_logger(__name__)
tracer = get_tracer(__name__)


class GenerationPipeline:
    def __init__(
        self,
        retrieval_client: RetrievalClient,
        generator: GroqGenerator,
        system_prompt: str,
        compression_target_ratio: float,
        generation_model: str,
        semantic_cache: SemanticAnswerCache | None = None,
        cache_embedder: CacheKeyEmbedder | None = None,
    ) -> None:
        self._retrieval_client = retrieval_client
        self._generator = generator
        self._system_prompt = system_prompt
        self._compression_target_ratio = compression_target_ratio
        self._generation_model = generation_model
        # Both-or-neither: the cache needs the query vector to key on
        # (ADR-026). The factory enforces this pairing; the guard here keeps
        # the pipeline correct if constructed directly in a test.
        self._semantic_cache = semantic_cache if cache_embedder is not None else None
        self._cache_embedder = cache_embedder

    async def run(self, request: QueryRequest) -> GenerationResponse:
        log = logger.bind(request_id=str(request.request_id))

        # ADR-026: semantic answer cache. On a hit within this caller's exact
        # tenant/principal/domain scope, the whole retrieve+generate path is
        # skipped. Scope partitioning (not just similarity) is what keeps a
        # hit from ever crossing an ACL boundary — see SemanticAnswerCache.
        query_embedding: list[float] | None = None
        if self._semantic_cache is not None and self._cache_embedder is not None:
            with tracer.start_as_current_span("pipeline.semantic_cache_lookup") as span:
                query_embedding = await self._cache_embedder.embed(request.query)
                cached = await self._semantic_cache.lookup(
                    query_embedding=query_embedding,
                    tenant_id=request.tenant_id,
                    principals=request.principals,
                    source_domains=request.source_domains,
                )
                span.set_attribute("cache_hit", cached is not None)
            if cached is not None:
                SEMANTIC_CACHE.labels(outcome="hit").inc()
                log.info("semantic_cache_hit")
                # Re-stamp request_id so the response corresponds to THIS
                # request, not the one that populated the cache.
                return cached.model_copy(update={"request_id": request.request_id})
            SEMANTIC_CACHE.labels(outcome="miss").inc()

        with tracer.start_as_current_span("pipeline.guardrail_scan_query") as span:
            query_flagged = scan_for_injection(request.query)
            span.set_attribute("flagged", query_flagged)
        if query_flagged:
            GUARDRAIL_FLAGS.labels(source="query").inc()
            log.warning("guardrail_query_flagged", query_preview=request.query[:200])

        with tracer.start_as_current_span("pipeline.retrieval") as span:
            log.info("retrieval_start")
            retrieved = await self._retrieval_client.retrieve(request)
            span.set_attribute("chunk_count", len(retrieved))
            log.info("retrieval_complete", chunk_count=len(retrieved))

        with tracer.start_as_current_span("pipeline.guardrail_scan_chunks") as span:
            safe_chunks, any_chunk_flagged = self._filter_flagged_chunks(retrieved, log)
            span.set_attribute("safe_chunk_count", len(safe_chunks))
            span.set_attribute("any_chunk_flagged", any_chunk_flagged)

        guardrail_flagged = query_flagged or any_chunk_flagged

        # If the query itself is flagged, we still answer (refusing outright
        # would be a worse UX for likely-benign matches), but the injected
        # text never reaches the model unescorted — it is passed through as
        # ordinary user content exactly like any other question, and the
        # response records that it was flagged for downstream auditing.
        with tracer.start_as_current_span("pipeline.compression") as span:
            compressed = compress_context(safe_chunks, self._compression_target_ratio)
            span.set_attribute("chunk_count", len(compressed))
            log.info("compression_complete", chunk_count=len(compressed))

        with tracer.start_as_current_span("pipeline.build_prompt"):
            messages = build_prompt(request.query, compressed, self._system_prompt)

        with tracer.start_as_current_span("pipeline.generate_and_validate"):
            response = await self._generate_and_validate(
                messages, request, guardrail_flagged, log, context_chunks=compressed
            )

        # ADR-026: cache the answer for near-identical future queries in this
        # scope. store() itself refuses to cache guardrail-flagged answers, so
        # a flagged/ungrounded response is never replayed to another caller.
        if (
            self._semantic_cache is not None
            and query_embedding is not None
            and not response.guardrail_flagged
        ):
            await self._semantic_cache.store(
                query_embedding=query_embedding,
                response=response,
                tenant_id=request.tenant_id,
                principals=request.principals,
                source_domains=request.source_domains,
            )
        return response

    def _filter_flagged_chunks(
        self, retrieved: list[RetrievedChunk], log: structlog.stdlib.BoundLogger
    ) -> tuple[list[RetrievedChunk], bool]:
        """Guardrail-scan every retrieved chunk's parent text; flagged chunks
        are dropped before compression/prompt-building ever sees them —
        exclusion happens here, not as a post-hoc log-only action."""
        safe: list[RetrievedChunk] = []
        any_flagged = False
        for chunk in retrieved:
            if scan_for_injection(chunk.parent.text) or scan_for_injection(chunk.chunk.text):
                any_flagged = True
                GUARDRAIL_FLAGS.labels(source="chunk").inc()
                log.warning(
                    "guardrail_chunk_flagged",
                    parent_id=chunk.parent.parent_id,
                    document_id=chunk.parent.document_id,
                )
                continue
            safe.append(chunk)
        return safe, any_flagged

    async def _generate_and_validate(
        self,
        messages: list[ChatMessage],
        request: QueryRequest,
        guardrail_flagged: bool,
        log: structlog.stdlib.BoundLogger,
        *,
        context_chunks: list[RetrievedChunk],
    ) -> GenerationResponse:
        raw_json = await self._generator.generate_structured(messages)

        try:
            response = validate_output(
                raw_json,
                request_id=request.request_id,
                model=self._generation_model,
                used_graph=request.use_graph,
            )
        except OutputValidationError as first_error:
            log.warning("output_validation_failed_retrying", error=str(first_error))
            retry_messages: list[ChatMessage] = [
                *messages,
                {"role": "assistant", "content": raw_json},
                {"role": "user", "content": build_retry_instruction()},
            ]
            raw_json_retry = await self._generator.generate_structured(retry_messages)
            try:
                response = validate_output(
                    raw_json_retry,
                    request_id=request.request_id,
                    model=self._generation_model,
                    used_graph=request.use_graph,
                )
            except OutputValidationError as second_error:
                log.error("output_validation_failed_permanently", error=str(second_error))
                raise

        # ADR-010/ADR-009: a non-trivial answer with zero citations is an
        # ungroundedness signal — the model asserted something it isn't
        # attributing to any retrieved chunk. Flagged (not refused): honest
        # refusals ("the context doesn't contain this") also carry no
        # citations, and telling the two apart reliably needs an LLM judge,
        # which belongs in the eval service (ADR-009), not the serve path.
        if not response.citations and response.answer.strip():
            GUARDRAIL_FLAGS.labels(source="uncited_answer").inc()
            log.warning("guardrail_uncited_answer", answer_preview=response.answer[:200])
            guardrail_flagged = True

        # ADR-028: a citation naming a parent_id that was never in the
        # context shown to the model is a distinct failure from having no
        # citations at all — schema-valid, present, but pointing at content
        # the model never actually saw (fabricated or training-data-recalled
        # identifier). Pure set-membership check, no added latency.
        ungrounded = find_ungrounded_citations(response.citations, context_chunks)
        if ungrounded:
            GUARDRAIL_FLAGS.labels(source="ungrounded_citation").inc()
            log.warning(
                "guardrail_ungrounded_citation",
                ungrounded_parent_ids=[c.parent_id for c in ungrounded],
            )
            guardrail_flagged = True

        if guardrail_flagged:
            response = response.model_copy(update={"guardrail_flagged": True})
        return response
