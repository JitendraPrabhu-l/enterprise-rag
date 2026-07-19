"""Contextual retrieval enrichment (ADR-023 — Anthropic's technique).

Before a chunk is embedded/BM25-indexed, a utility-model call generates a
1-2 sentence summary situating the chunk within its parent document ("From
AcmeCorp's 2023 10-K, liquidity discussion: ..."). That prefix travels with
the chunk as `ChunkRecord.context_prefix` and both indexes see
`searchable_text` (prefix + raw text) — the published numbers are a 49%
reduction in retrieval failures (67% combined with reranking), and the
technique remains the consensus enterprise default in 2026.

Failure policy mirrors the embedding cache (ADR-013): enrichment is an
optimization, so any per-chunk LLM failure logs, counts, and falls back to
the raw chunk text — ingestion never fails because enrichment did. The
generator is unaffected either way: it always reads the raw parent passage.

Cost note: one utility-tier call per chunk, batched with bounded
concurrency. On Groq's utility model this is fast and cheap; corpora where
even that is too much set CONTEXTUAL_ENRICHMENT_ENABLED=false and get
pre-ADR-023 behavior byte-for-byte.
"""

from __future__ import annotations

import asyncio

import structlog
from openai import AsyncOpenAI
from rag_core.schemas import ChunkRecord, ParentContext

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You situate document excerpts for a search index. Given a document "
    "passage and one chunk from it, reply with ONE short sentence (max ~40 "
    "words) stating what document/section the chunk is from and what it "
    "discusses, so the chunk is findable out of context. Reply with the "
    "sentence only — no preamble, no quotes."
)

# The parent passage is already ~1024 tokens; cap what we send as situating
# context so enrichment stays a utility-tier call even for outlier parents.
_MAX_PARENT_CHARS = 6_000


class ContextualEnricher:
    """Adds `context_prefix` to chunks via bounded-concurrency utility calls."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        max_concurrency: int = 8,
        max_tokens: int = 120,
    ) -> None:
        self._client = client
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_tokens = max_tokens

    async def enrich(
        self,
        chunks: list[ChunkRecord],
        parents_by_id: dict[str, ParentContext],
        *,
        document_title: str | None,
    ) -> list[ChunkRecord]:
        """Returns new ChunkRecords with context_prefix set where enrichment
        succeeded; order and count are preserved. Chunks whose call failed
        come back unchanged (context_prefix=None) — searchable_text then
        degrades to the raw text for exactly those chunks."""
        if not chunks:
            return chunks
        enriched = await asyncio.gather(
            *(self._enrich_one(chunk, parents_by_id, document_title) for chunk in chunks)
        )
        succeeded = sum(1 for c in enriched if c.context_prefix)
        logger.info(
            "contextual.enriched",
            total=len(chunks),
            succeeded=succeeded,
            failed=len(chunks) - succeeded,
        )
        return list(enriched)

    async def _enrich_one(
        self,
        chunk: ChunkRecord,
        parents_by_id: dict[str, ParentContext],
        document_title: str | None,
    ) -> ChunkRecord:
        parent = parents_by_id.get(chunk.parent_id)
        parent_text = (parent.text if parent else "")[:_MAX_PARENT_CHARS]
        title_line = f"Document title: {document_title}\n" if document_title else ""
        user_prompt = (
            f"{title_line}<passage>\n{parent_text}\n</passage>\n\n"
            f"<chunk>\n{chunk.text}\n</chunk>"
        )
        try:
            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            prefix = (response.choices[0].message.content or "").strip()
        except Exception:
            logger.warning("contextual.enrich_failed", chunk_id=chunk.chunk_id, exc_info=True)
            return chunk
        if not prefix:
            return chunk
        return chunk.model_copy(update={"context_prefix": prefix})
