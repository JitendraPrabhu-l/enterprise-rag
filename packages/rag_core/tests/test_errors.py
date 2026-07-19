"""Tests for `install_error_handlers` (ADR-033): FastAPI's own request-
validation errors must render `detail` as a string, matching every
hand-raised `HTTPException(detail=str)` in this codebase, instead of
Pydantic's structured error-list shape.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from rag_core.errors import install_error_handlers


class _Payload(BaseModel):
    rating: str
    count: int


def _app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/echo")
    async def echo(payload: _Payload) -> dict:
        return {"rating": payload.rating, "count": payload.count}

    return app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
class TestValidationErrorShape:
    async def test_valid_request_passes_through_unaffected(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/echo", json={"rating": "up", "count": 1})
        assert resp.status_code == 200
        assert resp.json() == {"rating": "up", "count": 1}

    async def test_missing_field_returns_string_detail_not_a_list(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/echo", json={"rating": "up"})

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert isinstance(detail, str), f"detail must be a string, got {type(detail)}: {detail!r}"
        assert "count" in detail

    async def test_wrong_type_returns_string_detail_not_a_list(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/echo", json={"rating": "up", "count": "not-a-number"})

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert isinstance(detail, str)
        assert "count" in detail

    async def test_multiple_errors_are_all_present_in_one_string(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post("/echo", json={})

        detail = resp.json()["detail"]
        assert isinstance(detail, str)
        assert "rating" in detail
        assert "count" in detail

    async def test_malformed_json_body_still_returns_string_detail(
        self, client: httpx.AsyncClient
    ) -> None:
        resp = await client.post(
            "/echo", content=b"not valid json{{{", headers={"content-type": "application/json"}
        )

        assert resp.status_code == 422
        assert isinstance(resp.json()["detail"], str)
