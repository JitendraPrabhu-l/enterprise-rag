"""Ingestion-service settings, layered on the shared `BaseServiceSettings`."""

from __future__ import annotations

from rag_core.config import BaseServiceSettings


class IngestionSettings(BaseServiceSettings):
    service_name: str = "rag-ingestion"

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    """Dimension of BAAI/bge-small-en-v1.5; update if `embedding_model_name` changes."""

    parent_chunk_tokens: int = 1024
    child_chunk_tokens: int = 128
    chunk_overlap_ratio: float = 0.15

    semantic_split_similarity_threshold: float = 0.55
    """Cosine similarity below which consecutive sentence windows are treated as a
    semantic boundary and force a split, even before the token budget is hit."""

    graph_enabled: bool = False
    """Global default for the optional GraphRAG (ADR-006) stage; individual ingest
    requests may still override this per source_domain via the API payload."""

    contextual_enrichment_enabled: bool = True
    """ADR-023 contextual retrieval: prepend an LLM-generated situating sentence to
    each chunk before embedding/BM25 indexing (−49% retrieval failures in the
    published evaluation, −67% combined with reranking). Costs one utility-model
    call per chunk at ingest time; disable to index raw text byte-for-byte as
    before."""

    contextual_enrichment_concurrency: int = 8
    """Bounded parallelism for enrichment calls — high enough to keep ingest fast,
    low enough to stay inside Groq utility-tier rate limits."""

    vision_page_max_tokens: int = 1024
    upload_dir: str = "/tmp/rag-ingestion-uploads"

    colpali_enabled: bool = False
    """ADR-029: opt-in ColPali late-interaction visual retrieval — an
    ADDITIONAL retrieval signal for table/figure-dense pages, indexed
    alongside (never instead of) the existing vision-description text path.
    Off by default: colpali-engine's model backbone is a multi-GB download
    and CPU inference is slow enough that this is a deliberate deployment
    decision, not a default every ingest should pay for."""

    colpali_model_name: str = "vidore/colpali-v1.3"

    # --- MinIO / S3-compatible object storage (ADR-014) ---
    minio_endpoint_url: str = "http://minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin_change_me"
    minio_bucket: str = "rag-documents"

    # --- Celery async ingestion (ADR-015) ---
    ingest_job_result_ttl_seconds: float = 24 * 60 * 60
    """How long a completed job's status stays queryable via GET
    /ingest/{job_id} before Celery's Redis result backend expires it."""
