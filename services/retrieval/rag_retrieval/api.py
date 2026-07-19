"""FastAPI app for the retrieval service.

Exposes `POST /retrieve` (body: `rag_core.schemas.QueryRequest`, response:
`list[RetrievedChunk]`) and `GET /healthz`. All external clients (Qdrant via
VectorStore, OpenSearch, Neo4j driver, Groq, the embedder/reranker models)
are constructed once at startup via the FastAPI lifespan and torn down
cleanly at shutdown so nothing leaks connections across requests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
from rag_core.embedding_cache import EmbeddingCache
from rag_core.errors import install_error_handlers
from rag_core.llm_clients import build_groq_client
from rag_core.logging import configure_logging
from rag_core.metrics import setup_metrics
from rag_core.rate_limit import FailOpenRateLimiter, build_route_limiter, get_redis_client
from rag_core.schemas import QueryRequest, RetrievedChunk
from rag_core.sparse_search import SparseSearchClient
from rag_core.tracing import configure_tracing, get_tracer
from rag_core.vector_store import VectorStore

from rag_retrieval.config import RetrievalSettings
from rag_retrieval.embeddings import QueryEmbedder
from rag_retrieval.graph_query import GraphQueryClient
from rag_retrieval.pipeline import ALL_STRATEGIES, RetrievalPipeline
from rag_retrieval.query_expansion import QueryExpander
from rag_retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)

QueryStrategyLiteral = Literal["direct", "multi_query", "hyde", "decompose"]
_DEFAULT_QUERY_STRATEGY: QueryStrategyLiteral = Query(default="direct")


class _AppState:
    """Container for the request-lifetime singletons wired up at startup."""

    pipeline: RetrievalPipeline
    sparse_client: SparseSearchClient
    neo4j_driver: object | None
    embedding_cache: EmbeddingCache
    retrieve_limiter: FailOpenRateLimiter | None


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = RetrievalSettings()
    configure_logging(settings.service_name, settings.log_level)
    configure_tracing(settings)
    tracer = get_tracer(__name__)

    logger.info("retrieval_service.starting", service=settings.service_name)

    vector_store = VectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        embedding_dim=settings.embedding_dim,
    )
    sparse_client = SparseSearchClient(
        settings.opensearch_url,
        index_prefix=settings.opensearch_index_prefix,
        verify_certs=settings.opensearch_verify_certs,
    )
    embedding_cache = EmbeddingCache(settings.redis_url)
    embedder = QueryEmbedder(settings.embedding_model_name, embedding_cache=embedding_cache)
    reranker = Reranker(settings.reranker_model_name)

    rate_limit_redis_client = get_redis_client(settings.redis_url)
    retrieve_limiter = await build_route_limiter(
        rate_limit_redis_client,
        requests_per_minute=settings.rate_limit_per_minute,
        bucket_key="retrieve",
    )

    groq_client = build_groq_client(settings)
    query_expander = QueryExpander(
        groq_client, settings.utility_model, settings.utility_model_max_tokens
    )

    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    graph_client = GraphQueryClient(
        neo4j_driver, max_hops=settings.graph_max_hops, max_entities=settings.graph_max_entities
    )

    pipeline = RetrievalPipeline(
        vector_store=vector_store,
        sparse_client=sparse_client,
        embedder=embedder,
        reranker=reranker,
        query_expander=query_expander,
        graph_client=graph_client,
        rrf_k=settings.rrf_k,
        tracer=tracer,
    )

    _state.pipeline = pipeline
    _state.sparse_client = sparse_client
    _state.neo4j_driver = neo4j_driver
    _state.embedding_cache = embedding_cache
    _state.retrieve_limiter = retrieve_limiter

    try:
        yield
    finally:
        logger.info("retrieval_service.shutting_down")
        await sparse_client.close()
        await neo4j_driver.close()
        await embedding_cache.close()
        await rate_limit_redis_client.aclose()


app = FastAPI(title="rag-retrieval", lifespan=lifespan)
setup_metrics(app)
install_error_handlers(app)  # ADR-033: normalize validation-error detail shape


class HealthResponse(BaseModel):
    status: str


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


async def _apply_retrieve_rate_limit(request: Request) -> None:
    limiter = getattr(_state, "retrieve_limiter", None)
    if limiter is None:
        return  # not yet initialized (e.g. during startup); fail open
    await limiter(request, Response())


@app.post(
    "/retrieve",
    response_model=list[RetrievedChunk],
    dependencies=[Depends(_apply_retrieve_rate_limit)],
)
async def retrieve(
    request: QueryRequest,
    query_strategy: QueryStrategyLiteral = _DEFAULT_QUERY_STRATEGY,
) -> list[RetrievedChunk]:
    """Run the full hybrid+RRF+rerank(+graph) retrieval flow for `request`.

    `tenant_id` on `request` is a hard pre-filter (ADR-010) enforced all the
    way down in `VectorStore.search` and `SparseSearchClient.search` — this
    endpoint never bypasses it.
    """
    if query_strategy not in ALL_STRATEGIES:
        raise HTTPException(status_code=422, detail=f"invalid query_strategy: {query_strategy!r}")

    try:
        return await _state.pipeline.retrieve(request, query_strategy=query_strategy)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "retrieve.failed", request_id=str(request.request_id), error_type=type(exc).__name__
        )
        raise HTTPException(status_code=502, detail="retrieval pipeline failed") from exc
