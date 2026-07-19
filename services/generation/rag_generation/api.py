"""FastAPI app exposing `POST /generate` and `GET /healthz`.

Wires `configure_tracing`/`configure_logging` at startup and emits one span
per pipeline stage via `get_tracer(__name__)`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from opentelemetry.trace import Status, StatusCode
from rag_core.errors import install_error_handlers
from rag_core.logging import configure_logging
from rag_core.metrics import ANSWER_FEEDBACK, setup_metrics
from rag_core.rate_limit import FailOpenRateLimiter, build_route_limiter, get_redis_client
from rag_core.schemas import AnswerFeedback, GenerationResponse, QueryRequest
from rag_core.semantic_cache import SemanticAnswerCache
from rag_core.tracing import configure_tracing, get_tracer

from rag_generation.cache_embedder import CacheKeyEmbedder
from rag_generation.config import GenerationSettings
from rag_generation.generation import GenerationCallError, GroqGenerator
from rag_generation.guardrails import OutputValidationError
from rag_generation.pipeline import GenerationPipeline
from rag_generation.retrieval_client import RetrievalClient, RetrievalServiceError

tracer = get_tracer(__name__)


class _AppState:
    """Holds the process-lifetime pipeline instance built in the lifespan hook."""

    pipeline: GenerationPipeline | None = None
    generate_limiter: FailOpenRateLimiter | None = None


_state = _AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = GenerationSettings()
    configure_logging(settings.service_name, settings.log_level)
    configure_tracing(settings)

    retrieval_client = RetrievalClient(
        base_url=settings.retrieval_service_url,
        timeout_seconds=settings.retrieval_timeout_seconds,
    )
    generator = GroqGenerator(
        settings=settings,
        model=settings.generation_model,
        max_output_tokens=settings.max_output_tokens,
    )

    # ADR-026 semantic answer cache — both-or-neither with its key embedder.
    semantic_cache = None
    cache_embedder = None
    if settings.semantic_cache_enabled:
        semantic_cache = SemanticAnswerCache(
            settings.redis_url,
            similarity_threshold=settings.semantic_cache_similarity_threshold,
            ttl_seconds=settings.semantic_cache_ttl_seconds,
        )
        cache_embedder = CacheKeyEmbedder(settings.semantic_cache_embedding_model)

    _state.pipeline = GenerationPipeline(
        retrieval_client=retrieval_client,
        generator=generator,
        system_prompt=settings.system_prompt,
        compression_target_ratio=settings.compression_target_ratio,
        generation_model=settings.generation_model,
        semantic_cache=semantic_cache,
        cache_embedder=cache_embedder,
    )

    redis_client = get_redis_client(settings.redis_url)
    _state.generate_limiter = await build_route_limiter(
        redis_client, requests_per_minute=settings.rate_limit_per_minute, bucket_key="generate"
    )

    yield
    _state.pipeline = None
    _state.generate_limiter = None
    if semantic_cache is not None:
        await semantic_cache.close()
    await redis_client.aclose()


app = FastAPI(title="rag-generation", lifespan=lifespan)
setup_metrics(app)
install_error_handlers(app)  # ADR-033: normalize validation-error detail shape
logger = structlog.get_logger(__name__)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/feedback", status_code=202)
async def feedback(payload: AnswerFeedback) -> dict[str, str]:
    """Record user feedback on an answer (ADR-027).

    Increments the rating counter (the dashboard/alert signal) and structured-
    logs the full payload. A thumbs-down carrying the query + answer is a
    ready-made golden-set candidate — the eval service (ADR-009) or an offline
    job harvests these from the logs into regression cases, closing the
    production-failures-become-tests loop that 2026 practice treats as
    table stakes. Deliberately log-and-count rather than write to a new
    datastore: keeps this a thin, dependency-free signal capture; where those
    signals accumulate is the consumer's choice, not this endpoint's.
    """
    ANSWER_FEEDBACK.labels(rating=payload.rating).inc()
    logger.info(
        "answer_feedback",
        request_id=str(payload.request_id),
        rating=payload.rating,
        query=payload.query,
        answer_preview=(payload.answer or "")[:500],
        comment=payload.comment,
    )
    return {"status": "recorded"}


async def _apply_generate_rate_limit(request: Request) -> None:
    if _state.generate_limiter is None:
        return  # not yet initialized (e.g. during startup); fail open
    await _state.generate_limiter(request, Response())


@app.post(
    "/generate",
    response_model=GenerationResponse,
    dependencies=[Depends(_apply_generate_rate_limit)],
)
async def generate(request: QueryRequest) -> GenerationResponse:
    pipeline = _state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="generation pipeline not initialized")

    log = logger.bind(request_id=str(request.request_id))
    with tracer.start_as_current_span("generation.request") as span:
        span.set_attribute("request_id", str(request.request_id))
        span.set_attribute("query_preview", request.query[:200])
        try:
            result = await pipeline.run(request)
            span.set_status(Status(StatusCode.OK))
            return result
        except RetrievalServiceError as exc:
            log.error("retrieval_service_error", error=str(exc))
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise HTTPException(status_code=502, detail=f"retrieval service error: {exc}") from exc
        except GenerationCallError as exc:
            log.error("generation_call_error", error=str(exc))
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise HTTPException(status_code=502, detail=f"generation error: {exc}") from exc
        except OutputValidationError as exc:
            log.error("output_validation_error", error=str(exc))
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise HTTPException(
                status_code=502, detail=f"model output failed validation: {exc}"
            ) from exc
