"""FastAPI app exposing production sampling / observability endpoints.

`POST /score` scores a single production interaction (query, retrieved
context, answer) on the RAG Triad — intended for an external system that
samples a percentage of live traffic and posts it here for scoring.
`GET /healthz` is a liveness probe.

Each scoring call is wrapped in an OpenTelemetry span (`configure_tracing` /
`get_tracer`, from rag_core.tracing) and logged as a structured record via
structlog, so this could feed a dashboard/Langfuse-style system in a full
deployment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rag_core.errors import install_error_handlers
from rag_core.llm_clients import build_groq_client
from rag_core.logging import configure_logging
from rag_core.metrics import setup_metrics
from rag_core.rate_limit import build_route_limiter, get_redis_client
from rag_core.tracing import configure_tracing, get_tracer

from rag_eval.config import EvalSettings
from rag_eval.judges import (
    JudgeError,
    score_answer_relevance,
    score_context_precision,
    score_faithfulness,
)
from rag_eval.schemas import TriadResult

_settings = EvalSettings()
_logger = configure_logging(_settings.service_name, _settings.log_level)
_tracer = get_tracer(__name__)


class ScoreRequest(BaseModel):
    """A single production interaction to score, as sampled by an external system."""

    query: str = Field(min_length=1)
    retrieved_context: list[str]
    answer: str = Field(min_length=1)


class HealthResponse(BaseModel):
    status: str = "ok"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_tracing(_settings)
    app.state.groq_client = build_groq_client(_settings)

    redis_client = get_redis_client(_settings.redis_url)
    app.state.score_limiter = await build_route_limiter(
        redis_client, requests_per_minute=_settings.rate_limit_per_minute, bucket_key="score"
    )
    app.state.rate_limit_redis_client = redis_client

    try:
        yield
    finally:
        await app.state.groq_client.close()
        await redis_client.aclose()


app = FastAPI(title="rag-eval", lifespan=_lifespan)
setup_metrics(app)
install_error_handlers(app)  # ADR-033: normalize validation-error detail shape


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse()


async def _apply_score_rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "score_limiter", None)
    if limiter is None:
        return  # not yet initialized (e.g. during startup); fail open
    await limiter(request, Response())


@app.post(
    "/score",
    response_model=TriadResult,
    dependencies=[Depends(_apply_score_rate_limit)],
)
async def score(request: ScoreRequest) -> TriadResult:
    client: AsyncOpenAI = app.state.groq_client

    with _tracer.start_as_current_span("rag_eval.score") as span:
        span.set_attribute("rag_eval.query_length", len(request.query))
        span.set_attribute("rag_eval.answer_length", len(request.answer))
        span.set_attribute("rag_eval.num_retrieved_chunks", len(request.retrieved_context))

        try:
            faithfulness = await score_faithfulness(
                client,
                model=_settings.utility_model,
                answer=request.answer,
                context=request.retrieved_context,
                max_retries=_settings.judge_max_retries,
            )
            answer_relevance = await score_answer_relevance(
                client,
                model=_settings.utility_model,
                question=request.query,
                answer=request.answer,
                max_retries=_settings.judge_max_retries,
            )
            context_precision = await score_context_precision(
                client,
                model=_settings.utility_model,
                question=request.query,
                retrieved_chunks=request.retrieved_context,
                max_retries=_settings.judge_max_retries,
            )
        except JudgeError as exc:
            _logger.error(
                "rag_triad_scoring_failed",
                query=request.query,
                error=str(exc),
            )
            span.set_attribute("rag_eval.error", str(exc))
            raise HTTPException(status_code=502, detail=f"Judge scoring failed: {exc}") from exc

        result = TriadResult(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
        )

        _logger.info(
            "rag_triad_scored",
            query=request.query,
            faithfulness_score=faithfulness.score,
            answer_relevance_score=answer_relevance.score,
            context_precision_score=context_precision.score,
            num_retrieved_chunks=len(request.retrieved_context),
        )

        return result
