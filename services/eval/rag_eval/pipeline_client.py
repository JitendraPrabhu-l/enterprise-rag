"""Async HTTP client wrapper for calling the retrieval and generation services.

Both services are assumed reachable at configurable base URLs (env vars via
`EvalSettings`) and speak the `rag_core.schemas` contract: `POST /retrieve`
takes a `QueryRequest` body and returns `list[RetrievedChunk]`; `POST /generate`
takes a `QueryRequest` body and returns a `GenerationResponse`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic import ValidationError
from rag_core.schemas import GenerationResponse, QueryRequest, RetrievedChunk
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_T = TypeVar("_T")


class PipelineClientError(Exception):
    """Base class for pipeline HTTP client failures."""


class PipelineConnectionError(PipelineClientError):
    """Raised when the retrieval/generation service is unreachable or times out."""


class PipelineResponseError(PipelineClientError):
    """Raised when the retrieval/generation service returns a non-2xx status or a
    response body that does not match the expected `rag_core.schemas` contract.
    """


def _retryable() -> Callable[[Callable[[], Awaitable[_T]]], Callable[[], Awaitable[_T]]]:
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )


class PipelineClient:
    """Thin async wrapper around the retrieval and generation service HTTP APIs."""

    def __init__(
        self,
        *,
        retrieval_base_url: str,
        generation_base_url: str,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._retrieval_base_url = retrieval_base_url.rstrip("/")
        self._generation_base_url = generation_base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self) -> PipelineClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def retrieve(self, request: QueryRequest) -> list[RetrievedChunk]:
        """POST /retrieve on the retrieval service; returns the ranked chunk list."""
        url = f"{self._retrieval_base_url}/retrieve"

        @_retryable()
        async def _do_call() -> httpx.Response:
            try:
                return await self._client.post(
                    url, content=request.model_dump_json(), headers=_json_headers()
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                raise PipelineConnectionError(
                    f"Could not reach retrieval service at {url}"
                ) from exc

        response = await _do_call()
        if response.status_code >= 400:
            raise PipelineResponseError(
                f"Retrieval service returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PipelineResponseError(
                f"Retrieval service response was not valid JSON: {response.text[:500]!r}"
            ) from exc
        try:
            return [RetrievedChunk.model_validate(item) for item in payload]
        except (ValidationError, TypeError) as exc:
            raise PipelineResponseError(
                f"Retrieval service response did not match list[RetrievedChunk]: {payload!r}"
            ) from exc

    async def generate(self, request: QueryRequest) -> GenerationResponse:
        """POST /generate on the generation service; returns the full GenerationResponse."""
        url = f"{self._generation_base_url}/generate"

        @_retryable()
        async def _do_call() -> httpx.Response:
            try:
                return await self._client.post(
                    url, content=request.model_dump_json(), headers=_json_headers()
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                raise PipelineConnectionError(
                    f"Could not reach generation service at {url}"
                ) from exc

        response = await _do_call()
        if response.status_code >= 400:
            raise PipelineResponseError(
                f"Generation service returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PipelineResponseError(
                f"Generation service response was not valid JSON: {response.text[:500]!r}"
            ) from exc
        try:
            return GenerationResponse.model_validate(payload)
        except ValidationError as exc:
            raise PipelineResponseError(
                f"Generation service response did not match GenerationResponse: {payload!r}"
            ) from exc


def _json_headers() -> dict[str, str]:
    return {"content-type": "application/json"}
