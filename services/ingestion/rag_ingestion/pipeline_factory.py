"""Constructs a fully-wired `IngestionPipeline`.

Shared by two callers that each need their own instance of every
collaborator (`VectorStore`, embedder, Groq clients, ...) since they run in
separate processes: `api.py`'s FastAPI lifespan (constructs one at server
startup, used for the pre-ADR-015 `file_path` direct-ingest path and to
back `/ingest/{job_id}` status lookups) and `tasks.py`'s Celery worker
(constructs one per worker process at startup, ADR-015). Keeping this
factory in one place means the two processes can never drift apart on how
a pipeline gets built.
"""

from __future__ import annotations

from rag_core.embedding_cache import EmbeddingCache
from rag_core.llm_clients import build_groq_client
from rag_core.sparse_search import SparseIndexer
from rag_core.vector_store import VectorStore

from rag_ingestion.colpali_index import ColPaliEmbedder, ColPaliPageIndex
from rag_ingestion.config import IngestionSettings
from rag_ingestion.contextual import ContextualEnricher
from rag_ingestion.embeddings import SentenceTransformerEmbedder
from rag_ingestion.graph_extraction import GraphStore, TripleExtractor
from rag_ingestion.page_classifier import HeuristicPageClassifier
from rag_ingestion.pipeline import IngestionPipeline
from rag_ingestion.vision import GroqVisionDescriber


def build_pipeline(
    settings: IngestionSettings,
) -> tuple[IngestionPipeline, GraphStore, EmbeddingCache]:
    """Returns the pipeline plus the two collaborators the caller owns the
    lifecycle of (`GraphStore`/`EmbeddingCache` hold open connections that
    must be closed on shutdown; the caller decides when that is)."""
    embedder = SentenceTransformerEmbedder(settings.embedding_model_name)
    vector_store = VectorStore(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, embedding_dim=embedder.dimension
    )
    vision_describer = GroqVisionDescriber(settings, max_tokens=settings.vision_page_max_tokens)
    sparse_indexer = SparseIndexer(
        settings.opensearch_url,
        index_prefix=settings.opensearch_index_prefix,
        verify_certs=settings.opensearch_verify_certs,
    )
    triple_extractor = TripleExtractor(settings)
    graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    embedding_cache = EmbeddingCache(settings.redis_url)

    # ADR-023: contextual retrieval enrichment, on by default; a deployment
    # that can't spend one utility call per chunk turns it off and gets the
    # raw-text indexing behavior unchanged.
    contextual_enricher = (
        ContextualEnricher(
            build_groq_client(settings),
            model=settings.utility_model,
            max_concurrency=settings.contextual_enrichment_concurrency,
        )
        if settings.contextual_enrichment_enabled
        else None
    )

    # ADR-029: opt-in ColPali visual retrieval, an ADDITIONAL signal for
    # table/figure-dense pages alongside the vision-description text path
    # above — never a replacement. Constructing ColPaliEmbedder loads the
    # real multi-GB model, so this only happens when explicitly enabled.
    colpali_index = (
        ColPaliPageIndex(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            embedder=ColPaliEmbedder(settings.colpali_model_name),
        )
        if settings.colpali_enabled
        else None
    )

    pipeline = IngestionPipeline(
        settings=settings,
        page_classifier=HeuristicPageClassifier(),
        vision_describer=vision_describer,
        embedder=embedder,
        vector_store=vector_store,
        sparse_indexer=sparse_indexer,
        triple_extractor=triple_extractor,
        graph_store=graph_store,
        embedding_cache=embedding_cache,
        embedding_model_name=settings.embedding_model_name,
        contextual_enricher=contextual_enricher,
        colpali_index=colpali_index,
    )
    return pipeline, graph_store, embedding_cache
