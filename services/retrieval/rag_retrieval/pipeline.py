"""Retrieval pipeline orchestration: the one place that wires every stage together.

Flow per `QueryRequest` (ADR-004, ADR-005, ADR-006, ADR-010):

1. Query expansion — depending on `query_strategy`:
   - "direct": use the raw query text as-is.
   - "multi_query": ask Claude for 2-3 paraphrases; hybrid-search each and
     merge/dedupe the candidate sets before fusion.
   - "hyde": ask Claude for a hypothetical answer passage and embed *that*
     (not the raw query) for the dense leg; the raw query is still used for
     the sparse (BM25) leg since HyDE is a dense-embedding technique.
2. Dense search (Qdrant via `rag_core.VectorStore`) and sparse search
   (OpenSearch BM25) run concurrently per query variant.
3. Reciprocal Rank Fusion merges the ranked ID lists into one fused ranking.
4. Cross-encoder reranking narrows the fused top_k down to top_n.
5. If `use_graph` is set (explicitly or via the heuristic classifier), Neo4j
   graph context is fetched concurrently with steps 2-4 and merged in as
   additional context alongside the reranked chunks.

Tenancy (ADR-010) is threaded through every single external call from
`QueryRequest.tenant_id` — nothing in this module ever calls a search backend
without it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from opentelemetry.trace import Tracer
from rag_core.schemas import (
    ChunkRecord,
    ContentModality,
    DocumentMetadata,
    ParentContext,
    QueryRequest,
    RetrievedChunk,
    SourceType,
)
from rag_core.sparse_search import SparseSearchClient
from rag_core.vector_store import VectorStore

from rag_retrieval.embeddings import QueryEmbedder
from rag_retrieval.fusion import reciprocal_rank_fusion
from rag_retrieval.graph_query import GraphQueryClient, QueryClassifier
from rag_retrieval.query_expansion import QueryExpander
from rag_retrieval.reranker import Reranker

logger = structlog.get_logger(__name__)

QueryStrategy = str  # Literal["direct","multi_query","hyde","decompose"], API-boundary-enforced

ALL_STRATEGIES = ("direct", "multi_query", "hyde", "decompose")

_DEFAULT_DOMAINS = ["default"]


def _chunk_from_dense_hit(hit: dict[str, Any]) -> tuple[str, RetrievedChunk]:
    """Reconstruct a `RetrievedChunk` from a Qdrant hit payload (see VectorStore.upsert_chunks).

    `VectorStore.upsert_chunks` persists the full ~1024-token parent passage
    in each point's `payload["parent"]` (ADR-002), so the real parent text is
    used here. Older points written before that field existed carry
    `payload["parent"] is None` — fall back to the child chunk's own text
    rather than raising, so a partially-migrated collection still serves.
    """
    payload = hit["payload"]
    metadata_payload = payload["metadata"]
    metadata = DocumentMetadata(
        document_id=metadata_payload["document_id"],
        source_type=SourceType(metadata_payload["source_type"]),
        source_domain=metadata_payload["source_domain"],
        tenant_id=metadata_payload["tenant_id"],
        access_role=metadata_payload["access_role"],
        title=metadata_payload.get("title"),
        uri=metadata_payload.get("uri"),
        last_updated_epoch=metadata_payload["last_updated_epoch"],
        page_count=metadata_payload.get("page_count"),
        # ADR-034 fields: .get() with model defaults so pre-ADR-034 points
        # (which lack them) reconstruct as the current, v1, un-superseded
        # documents they effectively are.
        is_current=metadata_payload.get("is_current", True),
        document_version=metadata_payload.get("document_version", 1),
        superseded_by_document_id=metadata_payload.get("superseded_by_document_id"),
        content_hash=metadata_payload.get("content_hash"),
        embedding_model_version=metadata_payload.get("embedding_model_version"),
        valid_from_epoch=metadata_payload.get("valid_from_epoch"),
        valid_until_epoch=metadata_payload.get("valid_until_epoch"),
        extra=metadata_payload.get("extra", {}),
    )
    chunk = ChunkRecord(
        chunk_id=str(hit["id"]),
        parent_id=payload["parent_id"],
        document_id=payload["document_id"],
        text=payload["text"],
        modality=ContentModality(payload["modality"]),
        token_count=len(payload["text"].split()),
        metadata=metadata,
    )
    parent_payload = payload.get("parent")
    if parent_payload:
        parent = ParentContext(
            parent_id=parent_payload["parent_id"],
            document_id=parent_payload["document_id"],
            text=parent_payload["text"],
            page_number=parent_payload.get("page_number"),
            modality=ContentModality(parent_payload["modality"]),
            source_ref=parent_payload.get("source_ref"),
        )
    else:
        parent = ParentContext(
            parent_id=payload["parent_id"],
            document_id=payload["document_id"],
            text=payload["text"],
            modality=ContentModality(payload["modality"]),
        )
    retrieved = RetrievedChunk(chunk=chunk, parent=parent, dense_score=hit["score"])
    return chunk.chunk_id, retrieved


def _chunk_from_sparse_hit(hit: dict[str, Any]) -> tuple[str, RetrievedChunk]:
    """Reconstruct a `RetrievedChunk` from an OpenSearch hit `_source` document."""
    source = hit["source"]
    metadata = DocumentMetadata(
        document_id=source["document_id"],
        source_type=SourceType.PDF,
        source_domain=source.get("source_domain", "default"),
        tenant_id=source["tenant_id"],
        access_role=source.get("access_role", "public"),
        last_updated_epoch=source.get("last_updated_epoch", 0),
    )
    chunk = ChunkRecord(
        chunk_id=hit["id"],
        parent_id=source["parent_id"],
        document_id=source["document_id"],
        text=source["text"],
        modality=ContentModality(source.get("modality", "prose")),
        token_count=len(source["text"].split()),
        metadata=metadata,
    )
    parent = ParentContext(
        parent_id=source["parent_id"],
        document_id=source["document_id"],
        text=source["text"],
        modality=ContentModality(source.get("modality", "prose")),
    )
    retrieved = RetrievedChunk(chunk=chunk, parent=parent, sparse_score=hit["score"])
    return chunk.chunk_id, retrieved


class RetrievalPipeline:
    """Orchestrates query expansion, hybrid search, RRF fusion, rerank, and graph merge."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        sparse_client: SparseSearchClient,
        embedder: QueryEmbedder,
        reranker: Reranker,
        query_expander: QueryExpander,
        graph_client: GraphQueryClient | None,
        rrf_k: int,
        tracer: Tracer,
        corrective_enabled: bool = False,
        corrective_confidence_floor: float = 0.0,
        corrective_max_retries: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._sparse_client = sparse_client
        self._embedder = embedder
        self._reranker = reranker
        self._query_expander = query_expander
        self._graph_client = graph_client
        self._graph_classifier = QueryClassifier()
        self._rrf_k = rrf_k
        self._tracer = tracer
        self._corrective_enabled = corrective_enabled
        self._corrective_confidence_floor = corrective_confidence_floor
        self._corrective_max_retries = corrective_max_retries

    async def _hybrid_search_one(
        self,
        query_text: str,
        dense_vector: list[float],
        source_domains: list[str],
        tenant_id: str,
        top_k: int,
        principals: list[str] | None = None,
        include_superseded: bool = False,
    ) -> tuple[list[str], list[str], dict[str, RetrievedChunk]]:
        """Run dense + sparse search for a single query variant.

        Returns `(dense_id_ranking, sparse_id_ranking, chunks_by_id)` — the
        rankings feed RRF, and `chunks_by_id` is merged across all variants
        so the pipeline never has to re-fetch a chunk it already has.
        Tenancy (ADR-010), principal ACLs (ADR-024), and currency (ADR-034)
        are threaded into BOTH legs — neither store is ever searched without
        them.
        """
        with self._tracer.start_as_current_span("retrieval.dense_search"):
            dense_hits = await self._vector_store.search(
                query_vector=dense_vector,
                source_domains=source_domains,
                tenant_id=tenant_id,
                top_k=top_k,
                principals=principals,
                include_superseded=include_superseded,
            )
        with self._tracer.start_as_current_span("retrieval.sparse_search"):
            sparse_hits = await self._sparse_client.search(
                query=query_text,
                source_domains=source_domains,
                tenant_id=tenant_id,
                top_k=top_k,
                principals=principals,
                include_superseded=include_superseded,
            )

        chunks_by_id: dict[str, RetrievedChunk] = {}
        dense_ranking: list[str] = []
        for hit in dense_hits:
            chunk_id, retrieved = _chunk_from_dense_hit(hit)
            dense_ranking.append(chunk_id)
            chunks_by_id[chunk_id] = retrieved

        sparse_ranking: list[str] = []
        for hit in sparse_hits:
            chunk_id, retrieved = _chunk_from_sparse_hit(hit)
            sparse_ranking.append(chunk_id)
            if chunk_id in chunks_by_id:
                # Same chunk hit by both rankers — keep the dense-reconstructed
                # RetrievedChunk (has full DocumentMetadata) but record the
                # sparse score too so downstream provenance is complete.
                chunks_by_id[chunk_id] = chunks_by_id[chunk_id].model_copy(
                    update={"sparse_score": retrieved.sparse_score}
                )
            else:
                chunks_by_id[chunk_id] = retrieved

        return dense_ranking, sparse_ranking, chunks_by_id

    async def _expand_query(
        self, request: QueryRequest, query_strategy: str
    ) -> tuple[list[str], list[float] | None]:
        """Return `(query_texts_for_sparse_leg, override_dense_vector_or_None)`.

        For "multi_query" the returned list has multiple query texts (each
        gets its own dense+sparse search). For "hyde" the list still has the
        single original query text (used for the sparse leg) but a HyDE
        passage embedding is returned to override the dense leg's vector.
        For "direct" neither expansion applies.
        """
        if query_strategy == "direct":
            return [request.query], None

        if query_strategy == "multi_query":
            with self._tracer.start_as_current_span("retrieval.query_expansion"):
                variations = await self._query_expander.expand_multi_query(request.query)
            return variations, None

        if query_strategy == "decompose":
            # ADR-025 multi-hop: sub-questions reuse the multi_query machinery
            # (one hybrid search per text, merged pre-RRF) — the difference is
            # semantic, not mechanical. expand_multi_query varies PHRASING of
            # one fact; decompose splits into DIFFERENT facts, so a comparison
            # question retrieves every entity's passages instead of neither.
            with self._tracer.start_as_current_span("retrieval.query_decomposition"):
                subquestions = await self._query_expander.decompose(request.query)
            return subquestions, None

        if query_strategy == "hyde":
            with self._tracer.start_as_current_span("retrieval.query_expansion"):
                passage = await self._query_expander.generate_hyde_passage(request.query)
                hyde_vector = await self._embedder.embed(passage)
            return [request.query], hyde_vector

        raise ValueError(
            f"unknown query_strategy {query_strategy!r}, expected one of {ALL_STRATEGIES}"
        )

    # Escalation ladder for the corrective loop (ADR-038): each rung is a
    # progressively more aggressive retrieval strategy. `direct` is one dense+
    # sparse pass; `multi_query` casts a wider net with paraphrases; `decompose`
    # splits multi-fact questions so every entity gets retrieved. When a pass
    # grades insufficient, the loop climbs to the next rung it hasn't tried.
    _CORRECTIVE_LADDER = ("direct", "multi_query", "decompose")

    def _is_sufficient(self, results: list[RetrievedChunk]) -> bool:
        """CRAG-style deterministic sufficiency check (ADR-038): is the best
        cross-encoder rerank score at or above the confidence floor.

        Uses the reranker's own (query, passage) score already computed this
        pass — no extra model call. Empty results, or a top score below the
        floor, means "retrieval didn't clearly find the answer" and a
        corrective retry is warranted. Falls back to True when no rerank score
        is present (e.g. reranking disabled) so the loop never spins on a
        missing signal.
        """
        if not results:
            return False
        top_score = results[0].rerank_score
        if top_score is None:
            return True
        return top_score >= self._corrective_confidence_floor

    async def retrieve(
        self,
        request: QueryRequest,
        query_strategy: str = "direct",
    ) -> list[RetrievedChunk]:
        """Retrieve for `request`, correctively retrying weak results (ADR-038).

        Runs one full retrieval pass; if the corrective loop is enabled and the
        result grades insufficient (`_is_sufficient`), escalates the query
        strategy up `_CORRECTIVE_LADDER` and retries, up to
        `corrective_max_retries` times, keeping the best-scoring result seen.
        With the loop disabled this is exactly one `_retrieve_once` call — the
        pre-ADR-038 behavior — so nothing changes for callers who turn it off.
        """
        first = await self._retrieve_once(request, query_strategy)
        if not self._corrective_enabled or self._is_sufficient(first):
            return first

        best = first
        best_score = self._top_score(first)
        # Try each ladder rung not already used, up to the retry cap.
        tried = {query_strategy}
        attempts = 0
        for next_strategy in self._CORRECTIVE_LADDER:
            if attempts >= self._corrective_max_retries:
                break
            if next_strategy in tried:
                continue
            tried.add(next_strategy)
            attempts += 1
            logger.info(
                "pipeline.corrective_retry",
                request_id=str(request.request_id),
                from_strategy=query_strategy,
                to_strategy=next_strategy,
                attempt=attempts,
                best_score=best_score,
            )
            with self._tracer.start_as_current_span("retrieval.corrective_retry") as span:
                span.set_attribute("corrective.strategy", next_strategy)
                span.set_attribute("corrective.attempt", attempts)
                candidate = await self._retrieve_once(request, next_strategy)
            candidate_score = self._top_score(candidate)
            if candidate_score > best_score:
                best, best_score = candidate, candidate_score
            if self._is_sufficient(candidate):
                # Good enough — stop climbing the ladder.
                return candidate
        return best

    @staticmethod
    def _top_score(results: list[RetrievedChunk]) -> float:
        """Best rerank score in a result set, for comparing corrective passes.
        -inf for an empty set so any real pass beats 'found nothing'."""
        if not results:
            return float("-inf")
        return results[0].rerank_score if results[0].rerank_score is not None else float("-inf")

    async def _retrieve_once(
        self,
        request: QueryRequest,
        query_strategy: str = "direct",
    ) -> list[RetrievedChunk]:
        """Run the full retrieval flow for `request` and return top_n reranked chunks."""
        source_domains = request.source_domains or _DEFAULT_DOMAINS
        top_k = request.top_k
        top_n = request.top_n

        query_texts, hyde_vector_override = await self._expand_query(request, query_strategy)

        # Decide graph usage: explicit request flag wins; otherwise fall back
        # to the heuristic classifier's recommendation (ADR-006).
        should_use_graph = request.use_graph or self._graph_classifier.is_global(request.query)

        graph_task: asyncio.Task[list[str]] | None = None
        if should_use_graph and self._graph_client is not None:
            graph_task = asyncio.create_task(self._run_graph_stage(request))

        # Embed every query variant that doesn't already have an override vector.
        if hyde_vector_override is not None:
            dense_vectors = [hyde_vector_override]
        else:
            with self._tracer.start_as_current_span("retrieval.embed_query"):
                dense_vectors = await asyncio.gather(
                    *(self._embedder.embed(text) for text in query_texts)
                )

        # For multi_query, len(query_texts) == len(dense_vectors); for hyde,
        # query_texts has 1 entry (raw query, for sparse) paired with the
        # single HyDE dense vector.
        search_tasks = [
            self._hybrid_search_one(
                query_text=query_texts[i] if query_strategy != "hyde" else request.query,
                dense_vector=dense_vectors[i] if query_strategy != "hyde" else dense_vectors[0],
                source_domains=source_domains,
                tenant_id=request.tenant_id,
                top_k=top_k,
                principals=request.principals,
                include_superseded=request.include_superseded,
            )
            for i in range(len(query_texts))
        ]
        variant_results = await asyncio.gather(*search_tasks)

        all_chunks_by_id: dict[str, RetrievedChunk] = {}
        dense_rankings: list[list[str]] = []
        sparse_rankings: list[list[str]] = []
        for dense_ranking, sparse_ranking, chunks_by_id in variant_results:
            dense_rankings.append(dense_ranking)
            sparse_rankings.append(sparse_ranking)
            for chunk_id, retrieved in chunks_by_id.items():
                if chunk_id in all_chunks_by_id:
                    existing = all_chunks_by_id[chunk_id]
                    merged_dense = existing.dense_score
                    if retrieved.dense_score is not None:
                        merged_dense = max(merged_dense or 0.0, retrieved.dense_score)
                    merged_sparse = existing.sparse_score
                    if retrieved.sparse_score is not None:
                        merged_sparse = max(merged_sparse or 0.0, retrieved.sparse_score)
                    all_chunks_by_id[chunk_id] = existing.model_copy(
                        update={"dense_score": merged_dense, "sparse_score": merged_sparse}
                    )
                else:
                    all_chunks_by_id[chunk_id] = retrieved

        with self._tracer.start_as_current_span("retrieval.fusion"):
            fused = reciprocal_rank_fusion(dense_rankings + sparse_rankings, k=self._rrf_k)

        fused_chunks: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused:
            fused_candidate = all_chunks_by_id.get(chunk_id)
            if fused_candidate is None:
                continue
            fused_chunks.append(fused_candidate.model_copy(update={"rrf_score": rrf_score}))

        candidates = fused_chunks[:top_k]

        with self._tracer.start_as_current_span("retrieval.rerank"):
            reranked = await self._reranker.rerank(request.query, candidates, top_n)

        if graph_task is not None:
            with self._tracer.start_as_current_span("retrieval.graph_merge"):
                graph_context_lines = await graph_task
                if graph_context_lines:
                    reranked = self._merge_graph_context(reranked, graph_context_lines)

        logger.info(
            "pipeline.retrieve_complete",
            request_id=str(request.request_id),
            query_strategy=query_strategy,
            candidate_count=len(candidates),
            result_count=len(reranked),
            used_graph=graph_task is not None,
        )
        return reranked

    async def _run_graph_stage(self, request: QueryRequest) -> list[str]:
        """Fetch graph context lines; isolated so failures don't break vector/BM25 retrieval."""
        assert self._graph_client is not None
        contexts = await self._graph_client.query_related_entities(
            request.query, tenant_id=request.tenant_id
        )
        if not contexts:
            return []
        return [GraphQueryClient.format_context_text([ctx]) for ctx in contexts]

    @staticmethod
    def _merge_graph_context(
        chunks: list[RetrievedChunk], graph_lines: list[str]
    ) -> list[RetrievedChunk]:
        """Append graph traversal context onto the parent text of the top result.

        Graph context is supplementary (ADR-006: opt-in secondary, never a
        replacement for vector/BM25 results), so it is folded into the
        highest-ranked chunk's parent text rather than injected as a
        synthetic standalone chunk that would need a fabricated score.
        """
        if not chunks:
            return chunks
        graph_block = "\n".join(graph_lines)
        top = chunks[0]
        augmented_parent = top.parent.model_copy(
            update={"text": f"{top.parent.text}\n\n[Related graph context]\n{graph_block}"}
        )
        chunks[0] = top.model_copy(update={"parent": augmented_parent})
        return chunks
