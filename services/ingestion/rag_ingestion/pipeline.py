"""Ingestion pipeline orchestration: parse -> classify -> (vision describe) ->
chunk -> embed -> upsert -> (optional GraphRAG).

Owns wiring between the otherwise-independent modules; each module stays unit
-testable in isolation because the pipeline is the only place they're
composed together.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
from pypdf import PdfReader
from rag_core.embedding_cache import EmbeddingCache
from rag_core.schemas import (
    AccessRole,
    ChunkRecord,
    DocumentMetadata,
    ParentContext,
    SourceType,
)
from rag_core.sparse_search import SparseIndexer
from rag_core.vector_store import VectorStore

from rag_ingestion.chunking import chunk_document
from rag_ingestion.colpali_index import ColPaliPageIndex
from rag_ingestion.config import IngestionSettings
from rag_ingestion.contextual import ContextualEnricher
from rag_ingestion.embeddings import Embedder
from rag_ingestion.graph_extraction import GraphStore, TripleExtractor, extract_and_store_graph
from rag_ingestion.page_classifier import PageCategory, PageClassifier
from rag_ingestion.parsing import ParsedDocument, parse_document
from rag_ingestion.vision import VisionDescriber

logger = structlog.get_logger(__name__)

_SUFFIX_TO_SOURCE_TYPE = {
    ".pdf": SourceType.PDF,
    ".txt": SourceType.TEXT,
    ".md": SourceType.MARKDOWN,
}


@dataclass(frozen=True)
class IngestRequest:
    file_path: str
    """Local filesystem path the pipeline actually reads from (pypdf parsing,
    vision page rendering) — always a real local path, regardless of where
    the document is durably stored."""
    document_id: str
    source_domain: str
    tenant_id: str
    access_role: str | None = None
    title: str | None = None
    graph_enabled: bool | None = None
    """Per-request override; falls back to `IngestionSettings.graph_enabled` when None."""
    source_uri: str | None = None
    """Durable reference persisted as DocumentMetadata.uri (ADR-014: an
    `s3://bucket/key` MinIO/S3 reference for multipart uploads). Falls back
    to `file_path` when None — the pre-ADR-014 behavior for the `file_path`
    ingestion mode, where the local path *is* the durable reference."""


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    page_count: int
    parent_count: int
    chunk_count: int
    triples_written: int
    duration_seconds: float


def _render_page_to_png(pdf_path: str, page_number: int) -> bytes:
    """Rasterizes a single PDF page to PNG bytes for the vision call.

    pypdf itself has no rasterizer; `page.images` only yields embedded raster
    images, not a full-page render. We use that as our best-effort source of
    "the image on this page" — sufficient for figure/table-heavy pages, whose
    dominant content is typically one embedded image or scanned bitmap. If a
    page has no embedded image (e.g. a vector-drawn table), we fall back to
    encoding the extracted text as the "image" input is not applicable, so
    callers should treat a `LookupError` here as "nothing to render."
    """
    reader = PdfReader(pdf_path)
    page = reader.pages[page_number - 1]
    images = list(page.images)
    if not images:
        raise LookupError(f"No embedded raster image found on page {page_number}")
    # Prefer the largest embedded image as the representative page render.
    largest = max(images, key=lambda img: len(img.data))
    return bytes(largest.data)


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: IngestionSettings,
        page_classifier: PageClassifier,
        vision_describer: VisionDescriber,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_indexer: SparseIndexer,
        triple_extractor: TripleExtractor | None = None,
        graph_store: GraphStore | None = None,
        embedding_cache: EmbeddingCache | None = None,
        embedding_model_name: str = "",
        contextual_enricher: ContextualEnricher | None = None,
        colpali_index: ColPaliPageIndex | None = None,
    ) -> None:
        self._settings = settings
        self._page_classifier = page_classifier
        self._vision_describer = vision_describer
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_indexer = sparse_indexer
        self._triple_extractor = triple_extractor
        self._graph_store = graph_store
        self._embedding_cache = embedding_cache
        self._embedding_model_name = embedding_model_name
        self._contextual_enricher = contextual_enricher
        self._colpali_index = colpali_index

    async def ingest(self, request: IngestRequest) -> IngestResult:
        started = time.monotonic()
        log = logger.bind(document_id=request.document_id, source_domain=request.source_domain)

        parsed = await asyncio.to_thread(parse_document, request.file_path, request.document_id)
        log.info("document_parsed", page_count=parsed.page_count)

        metadata = self._build_metadata(request, parsed)
        page_texts, describe_calls = await self._resolve_page_texts(parsed, request, log)

        parents, chunks = chunk_document(
            document_id=request.document_id,
            page_texts=page_texts,
            metadata=metadata,
            embedder=self._embedder,
            parent_chunk_tokens=self._settings.parent_chunk_tokens,
            child_chunk_tokens=self._settings.child_chunk_tokens,
            chunk_overlap_ratio=self._settings.chunk_overlap_ratio,
            semantic_similarity_threshold=self._settings.semantic_split_similarity_threshold,
        )
        log.info("document_chunked", parent_count=len(parents), chunk_count=len(chunks))

        parents_by_id = {p.parent_id: p for p in parents}

        # ADR-023 contextual retrieval: situate each chunk within its parent
        # BEFORE embedding/BM25 indexing, so both hybrid legs search the
        # context-enriched form (searchable_text). Runs ahead of
        # _embed_chunks by necessity — the prefix is part of what gets
        # embedded and cached.
        if self._contextual_enricher is not None:
            chunks = await self._contextual_enricher.enrich(
                chunks, parents_by_id, document_title=metadata.title
            )
            log.info(
                "chunks_contextually_enriched",
                enriched=sum(1 for c in chunks if c.context_prefix),
            )

        chunks = await self._embed_chunks(chunks)
        await self._vector_store.upsert_chunks(chunks, parents=parents_by_id)
        log.info("chunks_upserted", chunk_count=len(chunks))

        await self._maybe_enable_quantization(request.source_domain, log)

        # ADR-020: the BM25 half of hybrid search (ADR-004). Deliberately NOT
        # wrapped in a try/except — a document indexed dense-only would make
        # hybrid retrieval silently inconsistent, so a sparse write failure
        # (after SparseIndexer's own retries) fails the whole job.
        sparse_count = await self._sparse_indexer.index_chunks(
            chunks, source_domain=request.source_domain
        )
        log.info("chunks_sparse_indexed", chunk_count=sparse_count)

        triples_written = await self._maybe_run_graph_extraction(request, parents, log)

        duration = time.monotonic() - started
        return IngestResult(
            document_id=request.document_id,
            page_count=parsed.page_count,
            parent_count=len(parents),
            chunk_count=len(chunks),
            triples_written=triples_written,
            duration_seconds=duration,
        )

    def _build_metadata(self, request: IngestRequest, parsed: ParsedDocument) -> DocumentMetadata:
        suffix = Path(request.file_path).suffix.lower()
        source_type = _SUFFIX_TO_SOURCE_TYPE.get(suffix, SourceType.HTML)
        access_role = AccessRole(request.access_role) if request.access_role else AccessRole.PUBLIC

        return DocumentMetadata(
            document_id=request.document_id,
            source_type=source_type,
            source_domain=request.source_domain,
            tenant_id=request.tenant_id,
            access_role=access_role,
            title=request.title,
            uri=request.source_uri or request.file_path,
            last_updated_epoch=int(time.time()),
            page_count=parsed.page_count,
        )

    async def _resolve_page_texts(
        self,
        parsed: ParsedDocument,
        request: IngestRequest,
        log: structlog.stdlib.BoundLogger,
    ) -> tuple[list[tuple[int, str]], int]:
        """Returns (page_number, text) pairs, replacing table/figure-dense pages'
        text with the vision model's description where a renderable image
        exists. Sparse pages with no renderable content are skipped entirely
        rather than contributing empty chunks."""
        file_path = request.file_path
        page_texts: list[tuple[int, str]] = []
        describe_calls = 0
        colpali_indexed = 0

        for page in parsed.pages:
            category = self._page_classifier.classify(page)

            if category == PageCategory.SPARSE:
                continue

            if category == PageCategory.TABLE_OR_FIGURE_DENSE:
                try:
                    image_bytes = await asyncio.to_thread(
                        _render_page_to_png, file_path, page.page_number
                    )
                except LookupError:
                    # No embedded image to describe; fall back to whatever raw
                    # text pypdf extracted rather than dropping the page.
                    if page.text.strip():
                        page_texts.append((page.page_number, page.text))
                    continue

                description = await self._vision_describer.describe_page(
                    image_bytes, page_number=page.page_number
                )
                describe_calls += 1
                if description.strip():
                    page_texts.append((page.page_number, description))
                elif page.text.strip():
                    page_texts.append((page.page_number, page.text))

                # ADR-029: ColPali is an ADDITIONAL retrieval signal for this
                # same table/figure-dense page — indexed alongside the text
                # description above, never instead of it. Best-effort: a
                # ColPali failure must not affect the text path already
                # completed above (index_page itself never raises, but the
                # guard here is cheap insurance against a future change to
                # that contract).
                if self._colpali_index is not None:
                    try:
                        indexed = await self._colpali_index.index_page(
                            source_domain=request.source_domain,
                            document_id=request.document_id,
                            page_number=page.page_number,
                            image_bytes=image_bytes,
                        )
                        colpali_indexed += 1 if indexed else 0
                    except Exception:
                        log.warning(
                            "colpali_index_call_failed", page_number=page.page_number, exc_info=True
                        )
                continue

            if page.text.strip():
                page_texts.append((page.page_number, page.text))

        log.info("vision_describe_calls", count=describe_calls, colpali_indexed=colpali_indexed)
        return page_texts, describe_calls

    async def _embed_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Embeds chunk text, consulting the Redis embedding cache (ADR-013)
        first when one is configured. Only cache misses hit the model — a
        corpus with repeated boilerplate (e.g. SEC filing standard-language
        pages) can skip the model entirely for a meaningful fraction of its
        chunks. Cache absence/failure is transparent: every chunk still gets
        embedded, just always via the model instead of some via cache."""
        if not chunks:
            return chunks

        # searchable_text, not .text (ADR-023): the embedding must represent
        # the same context-situated form BM25 indexes. Cache keys follow
        # automatically — enriched and raw forms are different content and
        # therefore different cache entries, never a stale collision.
        texts = [c.searchable_text for c in chunks]
        cached: list[list[float] | None] = [None] * len(texts)
        if self._embedding_cache is not None:
            cached = await self._embedding_cache.get_many(self._embedding_model_name, texts)

        miss_indices = [i for i, vector in enumerate(cached) if vector is None]
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            computed = await asyncio.to_thread(self._embedder.embed, miss_texts)
            for i, vector in zip(miss_indices, computed, strict=True):
                cached[i] = vector

            if self._embedding_cache is not None:
                await self._embedding_cache.set_many(
                    self._embedding_model_name,
                    list(zip(miss_texts, computed, strict=True)),
                )

        vectors: list[list[float]] = []
        for maybe_vector in cached:
            assert maybe_vector is not None  # every index was either a hit or filled above
            vectors.append(maybe_vector)

        return [
            c.model_copy(update={"embedding": vector})
            for c, vector in zip(chunks, vectors, strict=True)
        ]

    async def _maybe_enable_quantization(
        self, source_domain: str, log: structlog.stdlib.BoundLogger
    ) -> None:
        """ADR-003: turn on scalar INT8 quantization for a domain that has
        grown past QUANTIZATION_THRESHOLD_VECTORS (idempotent no-op below
        threshold or if already enabled — see VectorStore's own docstring
        for why this must be checked post-upsert rather than at collection
        creation). Best-effort by design: a quantization-decision failure
        must never fail an otherwise-successful ingest job, since this is a
        storage/RAM optimization, not a correctness dependency."""
        try:
            newly_quantized = await self._vector_store.enable_quantization_if_due(source_domain)
            if newly_quantized:
                log.info("vector_store_quantization_enabled", source_domain=source_domain)
        except Exception:
            log.warning("vector_store_quantization_check_failed", exc_info=True)

    async def _maybe_run_graph_extraction(
        self,
        request: IngestRequest,
        parents: list[ParentContext],
        log: structlog.stdlib.BoundLogger,
    ) -> int:
        graph_enabled = (
            request.graph_enabled
            if request.graph_enabled is not None
            else self._settings.graph_enabled
        )
        if not graph_enabled:
            return 0
        if self._triple_extractor is None or self._graph_store is None:
            log.warning("graph_enabled_but_not_configured")
            return 0

        triples_written = await extract_and_store_graph(
            parents, extractor=self._triple_extractor, graph_store=self._graph_store
        )
        log.info("graph_extraction_complete", triples_written=triples_written)
        return triples_written
