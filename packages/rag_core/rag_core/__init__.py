"""Shared contracts, config, tracing, and clients for the Production RAG stack.

Every service (ingestion, retrieval, generation, eval) depends on this package
instead of on each other, so the chunk record and trace span contracts stay
consistent across service boundaries. See docs/HLD.md.
"""

from rag_core.schemas import ChunkRecord, DocumentMetadata, ParentContext, RetrievedChunk

__all__ = [
    "ChunkRecord",
    "DocumentMetadata",
    "ParentContext",
    "RetrievedChunk",
]
