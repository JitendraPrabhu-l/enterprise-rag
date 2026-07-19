"""ColPali late-interaction visual retrieval (ADR-029) — an opt-in THIRD
retrieval leg for table/figure-dense pages, additive to the existing
dense+sparse hybrid search (ADR-004) and the vision-description text path
(ADR-001), never a replacement for either.

Why this is architecturally distinct from everything else in `rag_core`:

- Every other index in this stack stores ONE vector per unit (one dense
  embedding per child chunk, one BM25 document per chunk) — a single-vector
  space where cosine/BM25 similarity is a single number. ColPali embeds a
  whole page as a GRID of patch embeddings (one per ~32x32 image patch,
  commonly 1000+ patches per page) and scores a query against a page via
  MaxSim: for each query-token embedding, take its best-matching patch, sum
  those best-matches across all query tokens. This needs Qdrant's
  multi-vector storage (`MultiVectorConfig` with `MAX_SIM` comparator) —
  a fundamentally different collection shape than `VectorStore`'s
  single-vector collections, which is why this is its own class/collection
  family (`rag_visual_{domain}`) rather than a code path bolted onto
  `VectorStore`.
- It embeds the PAGE IMAGE directly — no OCR, no vision-model text
  description, no chunking. This is complementary to, not a replacement
  for, the existing `GroqVisionDescriber` text-description path: the text
  description feeds the same embedding+BM25+RRF machinery every other
  chunk uses, while ColPali gives a second, independent retrieval signal
  for exactly the pages where OCR/description is most likely to lose
  information (dense tables, charts with fine-grained values, unusual
  layouts) — real published results show ColPali-style retrieval wins
  specifically where text extraction distorts visual structure.
- Genuinely heavy: colpali-engine + its vision-language model backbone is
  a multi-GB download, and CPU inference is slow enough that this MUST
  stay opt-in (`COLPALI_ENABLED=false` by default) — enabling it is a
  deliberate deployment decision given real backing compute, not a
  default every ingest pays for.

Failure policy mirrors ADR-023's contextual enrichment exactly: a
per-page indexing failure logs and is skipped, never fails the ingest
job — this is an additional retrieval signal, not a correctness
dependency for the existing pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog
from qdrant_client import AsyncQdrantClient, models

logger = structlog.get_logger(__name__)

_PATCH_VECTOR_NAME = "colpali_patches"


def visual_collection_name(source_domain: str) -> str:
    """One collection per domain, in its own namespace (`rag_visual_*`) —
    never shares a collection with `VectorStore`'s single-vector
    collections (`rag_*`), since the two have incompatible vector configs
    (single dense vector vs. a named multi-vector patch grid)."""
    return f"rag_visual_{source_domain}"


class PageImageEmbedder(Protocol):
    """Abstraction over the ColPali model: embed a rendered page image into
    its patch-embedding grid. Kept as a Protocol (mirroring `VisionDescriber`
    in vision.py) so tests never need to load the real multi-GB model —
    only `ColPaliEmbedder` below does that, and only when actually
    constructed by a deployment that opted in."""

    async def embed_page(self, image_bytes: bytes) -> list[list[float]]: ...

    @property
    def patch_dimension(self) -> int: ...


class ColPaliEmbedder:
    """Wraps the real `colpali-engine` model. Constructing this loads
    multi-GB model weights — only instantiate when
    `IngestionSettings.colpali_enabled` is True (see pipeline_factory.py)."""

    def __init__(self, model_name: str = "vidore/colpali-v1.3") -> None:
        # Deferred import: colpali-engine (and its torch dependency) is an
        # optional, heavy extra — importing it at module load time would
        # force every ingestion-service process to pay that cost even when
        # colpali_enabled=False, and would break environments (like most
        # test runs) where the package isn't installed at all.
        from colpali_engine.models import ColPali, ColPaliProcessor

        self._model = ColPali.from_pretrained(model_name)
        self._processor = ColPaliProcessor.from_pretrained(model_name)
        # ColPali-v1.3's patch embedding dimension; update if model_name changes.
        self._patch_dimension = 128

    @property
    def patch_dimension(self) -> int:
        return self._patch_dimension

    async def embed_page(self, image_bytes: bytes) -> list[list[float]]:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = self._processor.process_images([image])
        embeddings = self._model(**batch)
        # embeddings shape: (1, num_patches, patch_dim) -> squeeze batch dim.
        # colpali-engine/torch are untyped third-party libraries (same gap
        # as sentence-transformers elsewhere in this codebase).
        return embeddings[0].tolist()  # type: ignore[no-any-return]


class ColPaliPageIndex:
    """Qdrant-backed multi-vector store for ColPali page embeddings.

    Deliberately separate from `rag_core.vector_store.VectorStore`: see the
    module docstring for why the storage shape (named multi-vector,
    MAX_SIM comparator) doesn't fit that class's single-dense-vector
    collections.
    """

    def __init__(self, url: str, api_key: str | None, embedder: PageImageEmbedder) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self._embedder = embedder

    async def _ensure_collection(self, source_domain: str) -> None:
        name = visual_collection_name(source_domain)
        if await self._client.collection_exists(name):
            return
        await self._client.create_collection(
            collection_name=name,
            vectors_config={
                _PATCH_VECTOR_NAME: models.VectorParams(
                    size=self._embedder.patch_dimension,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )

    async def index_page(
        self,
        *,
        source_domain: str,
        document_id: str,
        page_number: int,
        image_bytes: bytes,
    ) -> bool:
        """Embed and upsert one page's patch grid. Returns True on success,
        False on failure (logged, never raised) — matching ADR-023's
        per-item best-effort policy: one page's ColPali failure must not
        block the rest of the document's ordinary text/vision indexing."""
        try:
            patch_vectors = await self._embedder.embed_page(image_bytes)
            await self._ensure_collection(source_domain)
            point_id = f"{document_id}:p{page_number}"
            await self._client.upsert(
                collection_name=visual_collection_name(source_domain),
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={_PATCH_VECTOR_NAME: patch_vectors},
                        payload={
                            "document_id": document_id,
                            "page_number": page_number,
                        },
                    )
                ],
            )
            return True
        except Exception:
            logger.warning(
                "colpali_index_page_failed",
                document_id=document_id,
                page_number=page_number,
                exc_info=True,
            )
            return False

    async def search(
        self,
        *,
        query_vectors: list[list[float]],
        source_domain: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """MaxSim search: `query_vectors` is the query's OWN patch/token
        embedding grid (from the same ColPali processor, applied to text
        instead of an image) — Qdrant computes MaxSim natively for a
        MultiVectorConfig collection given a multi-vector query.

        Returns [] (not an error) if the domain has no visual collection
        yet — mirrors `VectorStore.search`'s handling of a
        never-ingested domain."""
        name = visual_collection_name(source_domain)
        if not await self._client.collection_exists(name):
            return []
        hits = await self._client.query_points(
            collection_name=name,
            query=query_vectors,
            using=_PATCH_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        results: list[dict[str, Any]] = []
        for hit in hits.points:
            # Qdrant types a point's payload as `dict | None` — every point
            # this class itself writes always carries one (index_page always
            # sets document_id/page_number), but a point written some other
            # way (or corrupted) could not. Skip rather than crash the whole
            # search on one malformed point.
            if hit.payload is None:
                logger.warning("colpali_search_skipped_point_with_no_payload", point_id=hit.id)
                continue
            results.append(
                {
                    "score": hit.score,
                    "document_id": hit.payload["document_id"],
                    "page_number": hit.payload["page_number"],
                }
            )
        return results
