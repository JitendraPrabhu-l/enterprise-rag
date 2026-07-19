"""FastAPI surface for the ingestion service.

`POST /ingest` accepts either an uploaded file or a server-local file path
(the latter for batch/offline ingestion jobs that already have files staged
on a shared volume), plus the ADR-010 tenancy/access fields that must be
attached to every chunk at ingest time — retrieval enforces them as a hard
pre-filter, so they cannot be backfilled later without a re-index.

`file_path` is constrained to `IngestionSettings.upload_dir`
(`_resolve_staged_file_path`) — it is NOT a general "read any path on this
filesystem" escape hatch; only files already staged in the directory the
API and worker containers share are eligible.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from rag_core.errors import install_error_handlers
from rag_core.logging import configure_logging
from rag_core.metrics import INGEST_JOBS, setup_metrics
from rag_core.rate_limit import build_route_limiter, get_redis_client
from rag_core.tracing import configure_tracing

from rag_ingestion.config import IngestionSettings
from rag_ingestion.object_storage import ObjectStore
from rag_ingestion.pipeline import IngestRequest
from rag_ingestion.tasks import IngestJobStatusResponse, get_job_status, submit_ingest_job

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = IngestionSettings()
    configure_tracing(settings)
    configure_logging(settings.service_name, settings.log_level)

    # ADR-015: the API process only enqueues Celery jobs and reports their
    # status — it never runs IngestionPipeline itself (a separate worker
    # does), so it never loads the embedding model, connects to Neo4j, etc.
    # Startup here is deliberately cheap; the worker is where a broken
    # deployment (bad Qdrant URL, missing Groq key, ...) surfaces.
    redis_client = get_redis_client(settings.redis_url)
    ingest_limiter = await build_route_limiter(
        redis_client, requests_per_minute=settings.rate_limit_per_minute, bucket_key="ingest"
    )

    object_store = ObjectStore(
        endpoint_url=settings.minio_endpoint_url,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
    )

    app.state.settings = settings
    app.state.ingest_limiter = ingest_limiter
    app.state.object_store = object_store

    logger.info("ingestion_service_started")
    try:
        yield
    finally:
        await redis_client.aclose()
        logger.info("ingestion_service_stopped")


app = FastAPI(title="rag-ingestion", lifespan=_lifespan)
setup_metrics(app)
install_error_handlers(app)  # ADR-033: normalize validation-error detail shape


def _get_settings(request: Request) -> IngestionSettings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Ingestion service not yet initialized")
    return settings  # type: ignore[no-any-return]


def _get_object_store(request: Request) -> ObjectStore:
    object_store = getattr(request.app.state, "object_store", None)
    if object_store is None:
        raise HTTPException(status_code=503, detail="Object store not yet initialized")
    return object_store  # type: ignore[no-any-return]


async def _apply_ingest_rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "ingest_limiter", None)
    if limiter is None:
        return  # not yet initialized (e.g. during startup); fail open
    await limiter(request, Response())


def _resolve_staged_file_path(file_path: str, settings: IngestionSettings) -> Path:
    """Resolve a caller-supplied `file_path` and require it to be a
    descendant of `settings.upload_dir` — the one directory this service is
    actually meant to read caller-named paths from (ADR-015's shared
    upload volume between the API and worker containers).

    Without this check, `file_path` was an unauthenticated arbitrary local
    file read: any network caller reaching this endpoint could pass
    `file_path=/etc/passwd` (or any other container-readable path,
    including mounted secrets) and have it parsed and indexed as if it
    were a legitimate document — worse than a simple info leak, since a
    successfully ingested "document" becomes searchable by anyone who can
    later query that domain. `Path.resolve()` normalizes `..` segments and
    symlinks before the containment check runs, so neither can be used to
    escape `upload_dir` after the check passes.
    """
    staging_root = Path(settings.upload_dir).resolve()
    resolved = Path(file_path).resolve()
    if not resolved.is_relative_to(staging_root):
        raise HTTPException(
            status_code=403,
            detail=(
                f"file_path must be inside the configured upload directory "
                f"({settings.upload_dir}); server-local batch ingestion is only "
                f"supported for files already staged there."
            ),
        )
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"file_path not found: {file_path}")
    return resolved


class IngestAcceptedResponse(BaseModel):
    """ADR-015: /ingest no longer runs the pipeline in-request — this is
    everything the caller gets back immediately; the actual IngestResult is
    only available later, via GET /ingest/{job_id}."""

    job_id: str
    status: str = "queued"


class HealthResponse(BaseModel):
    status: str = "ok"


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/ingest",
    response_model=IngestAcceptedResponse,
    status_code=202,
    dependencies=[Depends(_apply_ingest_rate_limit)],
)
async def ingest(
    source_domain: Annotated[str, Form()],
    tenant_id: Annotated[str, Form()] = "public",
    access_role: Annotated[str, Form()] = "public",
    title: Annotated[str | None, Form()] = None,
    graph_enabled: Annotated[bool | None, Form()] = None,
    document_id: Annotated[str | None, Form()] = None,
    file_path: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    *,
    settings: Annotated[IngestionSettings, Depends(_get_settings)],
    object_store: Annotated[ObjectStore, Depends(_get_object_store)],
) -> IngestAcceptedResponse:
    """Accepts either a multipart `file` upload or a `file_path` already
    reachable on the service's filesystem — exactly one must be provided.

    A multipart upload is durably stored in MinIO/S3 (ADR-014) — that
    `s3://` reference becomes the document's permanent `uri` — *and* also
    written to local disk under `upload_dir`, since `IngestionPipeline`
    itself (pypdf parsing, vision page rendering) reads from a real local
    path regardless of where the document is durably archived.

    ADR-015: the actual pipeline run happens asynchronously on a Celery
    worker — this returns immediately with a job ID; poll
    GET /ingest/{job_id} for the result.
    """
    if file is None and not file_path:
        raise HTTPException(status_code=422, detail="Provide either 'file' or 'file_path'")
    if file is not None and file_path:
        raise HTTPException(status_code=422, detail="Provide only one of 'file' or 'file_path'")

    resolved_document_id = document_id or str(uuid.uuid4())
    resolved_source_uri: str | None = None

    if file is not None:
        suffix = Path(file.filename or "").suffix or ".bin"
        contents = await file.read()

        object_key = f"{resolved_document_id}{suffix}"
        resolved_source_uri = await object_store.put(
            object_key, contents, content_type=file.content_type or "application/octet-stream"
        )

        # Shared with the ingestion-worker container via a common volume
        # (docker-compose.yml) — the worker reads this same path when it
        # picks the job up, since IngestionPipeline needs a real local file.
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = upload_dir / object_key
        resolved_path.write_bytes(contents)
        resolved_file_path = str(resolved_path)
    else:
        assert file_path is not None  # guarded above
        resolved_file_path = str(_resolve_staged_file_path(file_path, settings))

    request = IngestRequest(
        file_path=resolved_file_path,
        document_id=resolved_document_id,
        source_domain=source_domain,
        tenant_id=tenant_id,
        access_role=access_role,
        title=title,
        graph_enabled=graph_enabled,
        source_uri=resolved_source_uri,
    )

    job_id = submit_ingest_job(request)
    INGEST_JOBS.labels(status="accepted").inc()
    return IngestAcceptedResponse(job_id=job_id)


@app.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
async def ingest_job_status(job_id: str) -> IngestJobStatusResponse:
    return get_job_status(job_id)
