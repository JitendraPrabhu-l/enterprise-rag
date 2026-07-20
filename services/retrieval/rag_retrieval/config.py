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

    # --- Corrective retrieval loop (ADR-038, CRAG-style) ---
    corrective_retrieval_enabled: bool = True
    """When on, retrieval grades its own top result and, if the best
    cross-encoder rerank score is below `corrective_confidence_floor`,
    escalates the query strategy and retries (ADR-038). Turns a single static
    retrieve→rerank pass into a bounded self-correcting loop: weak retrieval
    is caught and re-attempted at serve time instead of quietly feeding a
    thin context to the generator. Disable to force exactly one pass."""

    corrective_confidence_floor: float = 0.0
    """Cross-encoder score below which the top result is deemed insufficient
    and a corrective retry fires. bge-reranker-base emits logits centered near
    0 (positive = relevant, negative = not), so 0.0 is a sensible "the best
    thing we found isn't clearly relevant" boundary. Raise to correct more
    aggressively (more retries, higher cost); lower to correct only on the
    truly weak."""

    corrective_max_retries: int = 2
    """Max corrective re-retrievals after the initial pass (ADR-038). Capped —
    the CRAG/self-RAG literature converges on a small bound (≤5-6) so a
    genuinely unanswerable query can't loop forever; 2 escalations (e.g.
    direct → multi_query → decompose) is the practical sweet spot here."""
