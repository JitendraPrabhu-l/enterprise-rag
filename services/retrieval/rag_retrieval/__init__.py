"""Retrieval service: hybrid dense+sparse search, RRF fusion, cross-encoder
reranking, query expansion (multi-query / HyDE), and opt-in GraphRAG context.

Implements ADR-004 (hybrid search + RRF), ADR-005 (two-stage retrieval +
rerank), ADR-006 (GraphRAG as opt-in secondary), and ADR-010 (hard tenancy
pre-filter). See the top-level engineering-stack reference for the query
transformation strategies.
"""

from __future__ import annotations

__all__: list[str] = []
