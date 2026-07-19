from __future__ import annotations

from unittest.mock import MagicMock

from rag_ingestion.config import IngestionSettings
from rag_ingestion.page_classifier import HeuristicPageClassifier
from rag_ingestion.parsing import ParsedDocument
from rag_ingestion.pipeline import IngestionPipeline, IngestRequest


def _pipeline() -> IngestionPipeline:
    """Every collaborator is a bare MagicMock — `_build_metadata` never
    touches any of them, so this is enough to construct the pipeline and
    call the one method under test in isolation."""
    return IngestionPipeline(
        settings=IngestionSettings(),
        page_classifier=HeuristicPageClassifier(),
        vision_describer=MagicMock(),
        embedder=MagicMock(),
        vector_store=MagicMock(),
        sparse_indexer=MagicMock(),
    )


def _parsed(document_id: str = "doc-1", page_count: int = 3) -> ParsedDocument:
    return ParsedDocument(document_id=document_id, source_path="/tmp/whatever.pdf", pages=[])


class TestBuildMetadataUri:
    """Regression coverage for ADR-014: a multipart upload's durable
    reference (an s3://bucket/key MinIO/S3 URI) must become the persisted
    DocumentMetadata.uri, not the transient local path the pipeline reads
    the file from — but the pre-ADR-014 file_path-only ingestion mode
    (source_uri unset) must keep using file_path as before."""

    def test_source_uri_is_used_as_the_persisted_uri_when_present(self) -> None:
        request = IngestRequest(
            file_path="/tmp/rag-ingestion-uploads/doc-1.pdf",
            document_id="doc-1",
            source_domain="sec-filings",
            tenant_id="public",
            source_uri="s3://rag-documents/doc-1.pdf",
        )
        metadata = _pipeline()._build_metadata(request, _parsed())
        assert metadata.uri == "s3://rag-documents/doc-1.pdf"

    def test_falls_back_to_file_path_when_source_uri_is_none(self) -> None:
        """The file_path ingestion mode (server-local, already-staged files)
        has no object-store upload at all — file_path is itself the durable
        reference, same as before ADR-014."""
        request = IngestRequest(
            file_path="/data/staged/doc-1.pdf",
            document_id="doc-1",
            source_domain="sec-filings",
            tenant_id="public",
            source_uri=None,
        )
        metadata = _pipeline()._build_metadata(request, _parsed())
        assert metadata.uri == "/data/staged/doc-1.pdf"

    def test_source_type_is_still_derived_from_file_path_not_source_uri(self) -> None:
        """The suffix used to pick SourceType must come from the real local
        file (file_path), not the S3 key — they happen to share a suffix in
        the current upload path, but that's not guaranteed in general and
        the two fields serve different purposes."""
        request = IngestRequest(
            file_path="/tmp/rag-ingestion-uploads/doc-1.pdf",
            document_id="doc-1",
            source_domain="sec-filings",
            tenant_id="public",
            source_uri="s3://rag-documents/doc-1.pdf",
        )
        metadata = _pipeline()._build_metadata(request, _parsed())
        assert metadata.source_type.value == "pdf"
