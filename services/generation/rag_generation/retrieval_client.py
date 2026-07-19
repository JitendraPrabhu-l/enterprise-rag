"""Async HTTP client for the retrieval service's `POST /retrieve`.

This service never talks to Qdrant/OpenSearch directly — retrieval is always
brokered through the retrieval service's HTTP API (service isolation). Every
failure mode (timeout, connection error, non-2xx, malformed response body) is
surfaced as `RetrievalServiceError` — nothing is swallowed.
"""

from __future__ import annotations

import httpx
from pydantic import ValidationError
from rag_core.schemas import QueryRequest, RetrievedChunk
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RetrievalServiceError(Exception):
    """Raised for any failure calling the retrieval service — network error,
    timeout, non-2xx response, or a response body that fails to parse into
    `RetrievedChunk` objects."""


class RetrievalClient:
    """Thin async wrapper around `POST {base_url}/retrieve`."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        url = f"{self._base_url}/retrieve"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    url,
                    content=request.model_dump_json(),
                    headers={"content-type": "application/json"},
                )
        except httpx.TransportError as exc:
            # Let tenacity retry transport-level errors (connection refused,
            # DNS failure, timeout); re-raised as our typed error if retries
            # are exhausted (reraise=True propagates the original exception,
            # so wrap it here for callers outside the retry decorator too).
            raise RetrievalServiceError(
                f"failed to reach retrieval service at {url}: {exc}"
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise RetrievalServiceError(
                f"retrieval service returned HTTP {response.status_code} for {url}: "
                f"{response.text[:500]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RetrievalServiceError(
                f"retrieval service returned non-JSON response body: {exc}"
            ) from exc

        if not isinstance(payload, list):
            raise RetrievalServiceError(
                f"retrieval service response was not a JSON array (got {type(payload).__name__})"
            )

        try:
            return [RetrievedChunk.model_validate(item) for item in payload]
        except ValidationError as exc:
            raise RetrievalServiceError(
                f"retrieval service response did not match RetrievedChunk schema: {exc}"
            ) from exc
