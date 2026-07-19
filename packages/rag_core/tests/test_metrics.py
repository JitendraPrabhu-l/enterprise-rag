"""ADR-021: every service exposes Prometheus metrics via `setup_metrics`."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_core.metrics import setup_metrics


def _build_app() -> FastAPI:
    app = FastAPI()
    setup_metrics(app)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/work")
    def work() -> dict[str, str]:
        return {"status": "done"}

    return app


# One app for the whole module: prometheus_client registers collectors in a
# process-global registry, so `setup_metrics` — like in the real services —
# must run exactly once per process. A fresh app per test would silently
# no-op on re-registration and fail the handler-label assertions.
_APP = _build_app()


def test_metrics_endpoint_exists_and_serves_prometheus_text() -> None:
    # Context manager runs the app's startup events, where the
    # instrumentator registers its collectors.
    with TestClient(_APP) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_request" in response.text


def test_requests_are_counted_per_handler() -> None:
    with TestClient(_APP) as client:
        client.get("/work")
        body = client.get("/metrics").text
        assert 'handler="/work"' in body


def test_healthz_is_excluded_from_instrumentation() -> None:
    """Healthchecks fire on fixed timers; counting them buries real traffic."""
    with TestClient(_APP) as client:
        client.get("/healthz")
        body = client.get("/metrics").text
        assert 'handler="/healthz"' not in body
