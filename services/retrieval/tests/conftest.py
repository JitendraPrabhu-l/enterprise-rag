"""Shared pytest fixtures for the retrieval service test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Belt-and-suspenders: fail loudly if a test accidentally tries real I/O.

    All external clients (OpenSearch, Neo4j, Groq, Qdrant) must be mocked in
    unit tests per the brief — this fixture is a safety net, not a
    substitute for explicit mocking in each test.
    """
    yield
