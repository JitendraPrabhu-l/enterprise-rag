"""Prometheus metrics, shared across services (ADR-021).

Every service calls `setup_metrics(app)` once at app construction. This
exposes `/metrics` and instruments all routes with the standard RED signals
(request rate, error rate, duration histograms) under the default
`http_request_*` metric names, labeled by handler/method/status — which is
what the alert rules in deploy/prometheus/alert-rules.yml key on.

`/healthz` and `/metrics` themselves are excluded from instrumentation: both
are scraped/polled on fixed timers, so including them drowns the real traffic
signal in monitoring noise.

Domain counters live here too (not in the service that increments them) so
metric names are defined exactly once, next to the instrumentation setup.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

# ADR-010 guardrail observability: incremented by the generation pipeline
# whenever the injection heuristic flags the query or a retrieved chunk.
# A sudden rate increase is either an attack or a false-positive regression —
# both warrant a human look, hence the alert rule on this counter.
GUARDRAIL_FLAGS = Counter(
    "rag_guardrail_flags_total",
    "Prompt-injection guardrail flags raised by the generation pipeline",
    labelnames=("source",),  # "query" | "chunk" | "uncited_answer" | "ungrounded_citation"
)

INGEST_JOBS = Counter(
    "rag_ingest_jobs_total",
    "Ingestion jobs accepted by the API, by outcome of submission",
    labelnames=("status",),  # "accepted"
)

# ADR-026 semantic answer cache: a hit skips retrieval + generation entirely,
# so the hit ratio is the direct proxy for LLM-cost avoided. An unexpectedly
# low ratio in production means the similarity threshold is too strict (or
# traffic is genuinely all-unique); a high ratio with rising complaint rate
# means it is too loose and serving near-miss answers.
SEMANTIC_CACHE = Counter(
    "rag_semantic_cache_total",
    "Semantic answer-cache lookups by outcome",
    labelnames=("outcome",),  # "hit" | "miss"
)

# ADR-027 answer feedback: the production signal that turns real failures into
# eval cases. Rate and up/down split are the health metric a dashboard shows.
ANSWER_FEEDBACK = Counter(
    "rag_answer_feedback_total",
    "User feedback on generated answers",
    labelnames=("rating",),  # "up" | "down"
)


def setup_metrics(app: FastAPI) -> None:
    """Instrument `app` and expose GET /metrics in Prometheus text format."""
    Instrumentator(
        excluded_handlers=["/metrics", "/healthz"],
    ).instrument(app).expose(app, include_in_schema=False)
