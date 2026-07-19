"""Shared fixtures for ingestion service tests.

`FakeEmbedder` gives chunking tests a deterministic, dependency-free stand-in
for `SentenceTransformerEmbedder`: it hashes text into a small vector space so
identical/similar strings produce similar vectors without downloading any
real model, keeping unit tests fast and offline.
"""

from __future__ import annotations

import hashlib

import pytest
from rag_core.schemas import AccessRole, DocumentMetadata, SourceType


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder — no ML model required.

    Two sentences sharing more words produce vectors with higher cosine
    similarity, which is enough to exercise the semantic-boundary-splitting
    logic in `chunking.py` without a real embedding model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        words = text.lower().split()
        if not words:
            return vector
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = digest[0] % self._dimension
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        source_type=SourceType.PDF,
        source_domain="test-domain",
        tenant_id="tenant-a",
        access_role=AccessRole.INTERNAL,
        last_updated_epoch=1_700_000_000,
        page_count=1,
    )
