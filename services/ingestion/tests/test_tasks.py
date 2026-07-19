from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_ingestion.tasks import (
    _CELERY_STATE_TO_JOB_STATE,
    IngestJobStatusResponse,
    get_job_status,
)


class TestCeleryStateMapping:
    """Regression coverage for the Celery-state -> public-API-state
    translation (ADR-015) — GET /ingest/{job_id} must never leak Celery's
    own vocabulary (PENDING/RECEIVED/STARTED/RETRY/SUCCESS/FAILURE) to
    callers, only the documented queued/running/succeeded/failed set."""

    def test_every_mapped_value_is_one_of_the_four_public_states(self) -> None:
        assert set(_CELERY_STATE_TO_JOB_STATE.values()) <= {
            "queued",
            "running",
            "succeeded",
            "failed",
        }

    def test_pending_and_received_both_map_to_queued(self) -> None:
        assert _CELERY_STATE_TO_JOB_STATE["PENDING"] == "queued"
        assert _CELERY_STATE_TO_JOB_STATE["RECEIVED"] == "queued"

    def test_started_and_retry_both_map_to_running(self) -> None:
        assert _CELERY_STATE_TO_JOB_STATE["STARTED"] == "running"
        assert _CELERY_STATE_TO_JOB_STATE["RETRY"] == "running"

    def test_success_maps_to_succeeded(self) -> None:
        assert _CELERY_STATE_TO_JOB_STATE["SUCCESS"] == "succeeded"

    def test_failure_maps_to_failed(self) -> None:
        assert _CELERY_STATE_TO_JOB_STATE["FAILURE"] == "failed"


class TestGetJobStatus:
    def test_unknown_celery_state_defaults_to_queued_not_an_error(self) -> None:
        """A job ID Celery has never seen (e.g. a typo, or asked before the
        broker has recorded it) reports PENDING in real Celery — treating
        any unrecognized state as 'queued' rather than raising keeps this
        endpoint from 500ing on a state string this mapping doesn't yet
        know about."""
        mock_result = MagicMock(state="SOME_FUTURE_STATE_NOT_YET_MAPPED")
        with patch("rag_ingestion.tasks.celery_app.AsyncResult", return_value=mock_result):
            status = get_job_status("job-123")
        assert status.status == "queued"

    def test_succeeded_job_includes_the_result_payload(self) -> None:
        mock_result = MagicMock(state="SUCCESS", result={"document_id": "doc-1", "chunk_count": 5})
        with patch("rag_ingestion.tasks.celery_app.AsyncResult", return_value=mock_result):
            status = get_job_status("job-123")
        assert status.status == "succeeded"
        assert status.result == {"document_id": "doc-1", "chunk_count": 5}
        assert status.error is None

    def test_failed_job_includes_the_error_as_a_string_not_the_raw_exception(self) -> None:
        mock_result = MagicMock(state="FAILURE", result=ValueError("bad pdf"))
        with patch("rag_ingestion.tasks.celery_app.AsyncResult", return_value=mock_result):
            status = get_job_status("job-123")
        assert status.status == "failed"
        assert status.error == "bad pdf"
        assert status.result is None

    def test_queued_job_has_no_result_or_error(self) -> None:
        mock_result = MagicMock(state="PENDING")
        with patch("rag_ingestion.tasks.celery_app.AsyncResult", return_value=mock_result):
            status = get_job_status("job-123")
        assert status.status == "queued"
        assert status.result is None
        assert status.error is None

    def test_response_always_includes_the_requested_job_id(self) -> None:
        mock_result = MagicMock(state="STARTED")
        with patch("rag_ingestion.tasks.celery_app.AsyncResult", return_value=mock_result):
            status = get_job_status("the-specific-job-id")
        assert status.job_id == "the-specific-job-id"


class TestIngestJobStatusResponseSchema:
    def test_minimal_construction_defaults_result_and_error_to_none(self) -> None:
        response = IngestJobStatusResponse(job_id="job-1", status="queued")
        assert response.result is None
        assert response.error is None
