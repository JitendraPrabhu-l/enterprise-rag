"""Retrieval-service settings, layered on the shared `BaseServiceSettings`."""

from __future__ import annotations

from rag_core.config import BaseServiceSettings


class RetrievalSettings(BaseServiceSettings):
    service_name: str = "rag-retrieval"

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    """Must match the ingestion service's embedding model exactly — query and
    document vectors are only comparable if produced by the same model."""
    embedding_dim: int = 384
    """Dimension of BAAI/bge-small-en-v1.5; update if `embedding_model_name` changes."""

    reranker_model_name: str = "BAAI/bge-reranker-base"
    """Self-hosted cross-encoder (ADR-005) — no Cohere API dependency, cost-conscious."""

    rrf_k: int = 60
    """RRF constant (ADR-004): score = sum over rankers of 1 / (k + rank)."""

    default_top_k: int = 40
    """Wide retrieval width per ranker before fusion/rerank (ADR-005)."""
    default_top_n: int = 5
    """Final result count after cross-encoder rerank (ADR-005)."""

    graph_max_hops: int = 2
    graph_max_entities: int = 10
    """Cap on entities extracted from a query for graph traversal fan-out."""

    query_expansion_variations: int = 3
    """Number of paraphrased variations to generate for multi_query strategy."""

    utility_model_max_tokens: int = 512
