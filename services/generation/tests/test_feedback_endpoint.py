"""Tests for `POST /feedback` (ADR-027).

Drives the real ASGI app via `httpx.ASGITransport` — no lifespan triggered
(the endpoint has no dependency on `_state.pipeline`/Redis/model loads), so
this stays a fast, dependency-free test while exercising the real route,
request validation, and response shape.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from prometheus_client import REGISTRY

from rag_generation.api import app


def _sample(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
class TestFeedbackEndpoint:
    async def test_valid_up_feedback_returns_202(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/feedback", json={"request_id": str(uuid4()), "rating": "up"}
        )
        assert resp.status_code == 202
        assert resp.json() == {"status": "recorded"}

    async def test_valid_down_feedback_with_full_context_returns_202(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post(
            "/feedback",
            json={
                "request_id": str(uuid4()),
                "rating": "down",
                "query": "What was Q3 revenue?",
                "answer": "I don't know.",
                "comment": "This is in the 10-Q, page 12.",
            },
        )
        assert resp.status_code == 202

    async def test_invalid_rating_is_rejected_with_422(self, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/feedback", json={"request_id": str(uuid4()), "rating": "neutral"}
        )
        assert resp.status_code == 422

    async def test_missing_request_id_is_rejected_with_422(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/feedback", json={"rating": "up"})
        assert resp.status_code == 422

    async def test_up_feedback_increments_the_up_counter(
        self, client: httpx.AsyncClient
    ) -> None:
        before = _sample("rag_answer_feedback_total", {"rating": "up"})

        await client.post("/feedback", json={"request_id": str(uuid4()), "rating": "up"})

        after = _sample("rag_answer_feedback_total", {"rating": "up"})
        assert after == before + 1

    async def test_down_feedback_increments_the_down_counter_not_up(
        self, client: httpx.AsyncClient
    ) -> None:
        up_before = _sample("rag_answer_feedback_total", {"rating": "up"})
        down_before = _sample("rag_answer_feedback_total", {"rating": "down"})

        await client.post("/feedback", json={"request_id": str(uuid4()), "rating": "down"})

        assert _sample("rag_answer_feedback_total", {"rating": "down"}) == down_before + 1
        assert _sample("rag_answer_feedback_total", {"rating": "up"}) == up_before
