"""Tests that `SentenceTransformerEmbedder` actually threads `batch_size`
through to the underlying model's `.encode()` call — sentence-transformers'
own default (32) undershoots the 64-256 throughput sweet spot production
embedding-pipeline guidance converges on, so this is a real cost/latency
lever, not just a constructor parameter that gets ignored.

The real `SentenceTransformer` class is monkeypatched to a stand-in that
records its `.encode()` call kwargs — no real model is loaded, keeping this
fast and network-free.
"""

from __future__ import annotations

import numpy as np
import pytest


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_calls: list[dict] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        self.encode_calls.append(kwargs)
        return np.zeros((len(texts), 3))


@pytest.fixture
def patched_embedder(monkeypatch: pytest.MonkeyPatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    from rag_ingestion.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder


class TestBatchSizeIsThreadedToEncode:
    def test_default_batch_size_is_128_not_sentence_transformers_default_32(
        self, patched_embedder
    ) -> None:
        embedder = patched_embedder("fake-model")
        embedder.embed(["text one", "text two"])

        [call_kwargs] = embedder._model.encode_calls
        assert call_kwargs["batch_size"] == 128

    def test_explicit_batch_size_overrides_default(self, patched_embedder) -> None:
        embedder = patched_embedder("fake-model", batch_size=64)
        embedder.embed(["text"])

        [call_kwargs] = embedder._model.encode_calls
        assert call_kwargs["batch_size"] == 64

    def test_empty_texts_never_calls_encode(self, patched_embedder) -> None:
        embedder = patched_embedder("fake-model")
        result = embedder.embed([])

        assert result == []
        assert embedder._model.encode_calls == []
