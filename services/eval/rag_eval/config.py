"""Environment-driven settings for the eval service.

Extends `BaseServiceSettings` (rag_core.config) with the pipeline service URLs
this service calls over HTTP, and the RAG Triad pass/fail thresholds used by
the CI/CD gate (ADR-009).
"""

from __future__ import annotations

from rag_core.config import BaseServiceSettings


class EvalSettings(BaseServiceSettings):
    service_name: str = "rag-eval"

    retrieval_service_url: str = "http://retrieval:8000"
    generation_service_url: str = "http://generation:8000"

    faithfulness_threshold: float = 0.8
    answer_relevance_threshold: float = 0.75
    context_precision_threshold: float = 0.7

    judge_max_retries: int = 3
    """Tenacity retry attempts for judge calls to the Groq API and pipeline HTTP calls."""

    http_timeout_seconds: float = 60.0
    """Timeout for calls to the retrieval/generation services."""
