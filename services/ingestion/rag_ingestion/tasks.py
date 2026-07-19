"""Celery task definitions for async ingestion (ADR-015).

`run_ingestion_task` is a plain synchronous Celery task (Celery has no
native async task support) that bridges into the fully-async
`IngestionPipeline` via `asyncio.run` — the standard pattern for running
async code from inside sync-only calling code, same situation as
`sentence-transformers`/`boto3` calls elsewhere in this service, just at
the outermost layer instead of one method.

Each worker process constructs its own pipeline once, at Celery's
`worker_process_init` signal, and reuses it across every task that process
executes — the same "build once at process startup, not per-request/task"
pattern the FastAPI lifespan uses, just triggered by a different framework
hook.

Celery ships no py.typed marker and its `@app.task`/`@signal.connect`
decorators are untyped, so mypy strict sees the decorated functions as
untyped regardless of their own explicit annotations — the two
`# type: ignore[untyped-decorator]` below are a real, external gap in
Celery's own stub coverage, not a gap in this module.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Literal

import structlog
from celery import Celery
from celery.result import AsyncResult
from celery.signals import worker_process_init
from pydantic import BaseModel

from rag_ingestion.config import IngestionSettings
from rag_ingestion.pipeline import IngestionPipeline, IngestRequest
from rag_ingestion.pipeline_factory import build_pipeline

logger = structlog.get_logger(__name__)

_settings = IngestionSettings()

celery_app = Celery("rag_ingestion", broker=_settings.redis_url, backend=_settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Bounded so completed job state doesn't accumulate in Redis forever
    # (ADR-015's stated tradeoff versus Celery's unbounded default).
    result_expires=int(_settings.ingest_job_result_ttl_seconds),
)

_pipeline: IngestionPipeline | None = None


@worker_process_init.connect  # type: ignore[untyped-decorator]
def _init_worker_pipeline(**_kwargs: object) -> None:
    """Runs once per worker process at startup — not per task — so the
    embedding model, Qdrant/Neo4j clients, etc. are loaded a single time
    and reused across every task that process executes."""
    global _pipeline
    pipeline, _graph_store, _embedding_cache = build_pipeline(_settings)
    _pipeline = pipeline
    logger.info("ingestion_worker.pipeline_initialized")


def _get_worker_pipeline() -> IngestionPipeline:
    if _pipeline is None:
        # Only reachable if a task runs before worker_process_init fires,
        # or in a test/eager-mode context that never emits the signal —
        # build one on demand rather than fail the task outright.
        pipeline, _graph_store, _embedding_cache = build_pipeline(_settings)
        return pipeline
    return _pipeline


@celery_app.task(name="rag_ingestion.run_ingestion", bind=True)  # type: ignore[untyped-decorator]
def run_ingestion_task(self: Any, request_payload: dict[str, Any]) -> dict[str, Any]:
    request = IngestRequest(**request_payload)
    pipeline = _get_worker_pipeline()
    result = asyncio.run(pipeline.ingest(request))
    return asdict(result)


def submit_ingest_job(request: IngestRequest) -> str:
    """Enqueues an ingestion job, returning the Celery task/job ID."""
    async_result = run_ingestion_task.delay(asdict(request))
    return async_result.id  # type: ignore[no-any-return]


JobState = Literal["queued", "running", "succeeded", "failed"]

# Celery's Redis result backend reports "never submitted" and "submitted,
# still queued" identically as PENDING — there is no way to distinguish
# them from AsyncResult alone. A fabricated/typo'd job ID therefore reports
# "queued" forever rather than a 404 (ADR-015's documented tradeoff).
_CELERY_STATE_TO_JOB_STATE: dict[str, JobState] = {
    "PENDING": "queued",
    "RECEIVED": "queued",
    "STARTED": "running",
    "RETRY": "running",
    "SUCCESS": "succeeded",
    "FAILURE": "failed",
}


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    result: dict[str, Any] | None = None
    error: str | None = None


def get_job_status(job_id: str) -> IngestJobStatusResponse:
    async_result: AsyncResult = celery_app.AsyncResult(job_id)
    job_state = _CELERY_STATE_TO_JOB_STATE.get(async_result.state, "queued")

    if job_state == "succeeded":
        return IngestJobStatusResponse(job_id=job_id, status=job_state, result=async_result.result)
    if job_state == "failed":
        return IngestJobStatusResponse(
            job_id=job_id, status=job_state, error=str(async_result.result)
        )
    return IngestJobStatusResponse(job_id=job_id, status=job_state)
