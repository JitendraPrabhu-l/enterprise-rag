"""Environment-driven settings, shared across services.

Each service subclasses `BaseServiceSettings` and adds its own fields; all
settings load from environment variables (12-factor) with `.env` as a local
dev convenience, never as the source of truth in deployed environments.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "rag-service"
    environment: str = "development"
    log_level: str = "INFO"

    otel_exporter_otlp_endpoint: str = "http://otel-collector:4318"
    otel_traces_sample_rate: float = 1.0
    """Success-path trace sampling (ADR: 100% errors always kept, this governs the rest)."""

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None

    # Shared by retrieval (BM25 search) and ingestion (sparse indexing) — the
    # prefix lives here rather than per-service because the two sides MUST
    # agree on index naming or sparse search silently returns nothing for
    # every domain (the exact failure mode ADR-020 fixed).
    opensearch_url: str = "http://opensearch:9200"
    opensearch_index_prefix: str = "rag"
    opensearch_verify_certs: bool = True

    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me-in-prod"

    # --- Redis: embedding cache + rate limiting (ADR-013) ---
    # Shared by every service for rate limiting; only rag_ingestion/rag_retrieval
    # also use it for embedding-cache lookups. Both use cases fail open — a
    # Redis outage falls back to always-recompute / always-allow rather than
    # failing the request, so this is a performance/safety dependency, not an
    # availability one.
    redis_url: str = "redis://redis:6379/0"
    rate_limit_per_minute: int = 60

    # --- LLM provider: Groq ---
    # Groq speaks the OpenAI-compatible chat completions API, so one client
    # class (openai.AsyncOpenAI with a custom base_url) serves every call
    # site — only base_url/api_key/model differ.
    #
    # Every LLM role (generation, vision, utility) runs on Groq. OpenRouter's
    # free-tier models were tried first (ADR-011) but proved too congested in
    # practice (sustained 429s across multiple distinct free models/backends);
    # Groq has its own separate, more reliable rate-limit pool and its
    # openai/gpt-oss-120b model covers both text and vision, so one provider
    # now serves the whole stack (ADR-012). generation_model/vision_model/
    # utility_model stay as independent settings — same model by default, but
    # each is still swappable without code changes if a task benefits from a
    # different one later.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    generation_model: str = "openai/gpt-oss-120b"
    vision_model: str = "openai/gpt-oss-120b"
    utility_model: str = "openai/gpt-oss-120b"
