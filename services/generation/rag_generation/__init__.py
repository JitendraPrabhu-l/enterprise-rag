"""Generation service: retrieval-augmented answer synthesis with Claude.

Implements ADR-007 (prompt caching + model tiering), ADR-008 (entropy-based
context compression), and ADR-010 (input/output guardrails). This service
never talks to Qdrant/OpenSearch directly — it calls the retrieval service
over HTTP and hands the result to Claude for grounded, cited synthesis.
"""

from __future__ import annotations

__all__: list[str] = []
