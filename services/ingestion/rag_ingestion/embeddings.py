"""Local embedding model wrapper (cost-conscious ADR posture: self-hosted, no
external embedding API). Used both for child-chunk embeddings that go into
Qdrant and for the sentence-window vectors that drive semantic-boundary
chunking in `chunking.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Thin, synchronous wrapper — sentence-transformers has no native asyncio
    support, so callers needing async should offload via `asyncio.to_thread`.

    The import is deferred to `__init__` (rather than module level) so that
    modules which only need the `Embedder` Protocol — e.g. `chunking.py`'s
    type hints — don't pay the (heavy) sentence-transformers/torch import
    cost or fail to import in environments where only a fake/test embedder
    is ever constructed.
    """

    def __init__(
        self, model_name: str = "BAAI/bge-small-en-v1.5", *, batch_size: int = 128
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: SentenceTransformer = SentenceTransformer(model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"model {model_name!r} did not report a sentence embedding dimension")
        self._dimension = int(dimension)
        # sentence-transformers' own default (32) undershoots the throughput
        # sweet spot for CPU/GPU batch inference; 128 sits in the middle of
        # the 64-256 range production embedding-pipeline guidance converges
        # on, without pushing memory usage high enough to matter for a
        # ~384-dim model. `encode` batches internally regardless of how many
        # texts a single `embed()` call receives, so this is a real
        # throughput lever, not just documentation.
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self._batch_size,
        )
        return [vector.tolist() for vector in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Assumes both vectors are already L2-normalized (as `embed` produces),
    so this reduces to a plain dot product."""
    vec_a = np.asarray(a, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
