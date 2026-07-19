"""Shared error-response contract across all four services (ADR-033).

Every service raises plain `fastapi.HTTPException(status_code, detail=str)`
today, which FastAPI renders as `{"detail": "<the string>"}` — but FastAPI's
OWN request-validation errors (a malformed request body/query param) render
`detail` as a LIST of structured objects instead of a string, because that's
Pydantic's own validation-error shape passed through unchanged. A client
therefore cannot treat `detail` as "always a string" or "always a list"; it
has to branch on the response's shape to find out which endpoint failed and
why, which is exactly the kind of accidental inconsistency this module
closes without requiring every route in every service to be rewritten.

`install_error_handlers(app)` adds ONE exception handler, for
`RequestValidationError` specifically — the one case where FastAPI's
default rendering diverges from every hand-raised `HTTPException` in this
codebase. It flattens Pydantic's structured error list into the same
`detail: str` shape every other error already uses, rather than introducing
a new envelope shape none of the existing hand-raised exceptions use.
Hand-raised `HTTPException(detail=str)` calls are untouched — they already
match the contract this module standardizes on.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _format_validation_errors(exc: RequestValidationError) -> str:
    """Render Pydantic's structured error list as one human-readable
    string: `"body.rating: Value error, rating must be 'up' or 'down'"` per
    error, joined with `"; "` — enough detail to debug, in the same
    `detail: str` shape as every hand-raised HTTPException in this stack."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(segment) for segment in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) if parts else "Request validation failed."


def install_error_handlers(app: FastAPI) -> None:
    """Call once per service, right after `FastAPI(...)` construction —
    see any service's `api.py` for the call site. Idempotent to add to a
    service that has no other error-handling customization; does not
    touch hand-raised HTTPExceptions, which already return `detail: str`.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": _format_validation_errors(exc)},
        )
