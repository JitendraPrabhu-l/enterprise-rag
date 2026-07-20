"""Run a golden retrieval set through the live retrieval service and score it
on deterministic IR metrics (ADR-037).

The retrieval-stage counterpart to `eval_runner.run_eval_gate` (which scores
the final answer on the LLM-judged RAG Triad). This one drives `POST /retrieve`
for each golden query, compares the ranked chunk_ids it returns against the
item's known-relevant set, and aggregates recall@k / MRR / nDCG — no LLM, so
it is cheap, repeatable, and localizes a regression to retrieval rather than
generation.
"""

from __future__ import annotations

from rag_core.schemas import QueryRequest

from rag_eval.pipeline_client import PipelineClient
from rag_eval.retrieval_metrics import (
    RetrievalGateResult,
    aggregate_retrieval_metrics,
    score_retrieval,
)
from rag_eval.schemas import GoldenRetrievalItem


async def run_retrieval_gate(
    golden_set: list[GoldenRetrievalItem],
    *,
    pipeline_client: PipelineClient,
    k: int,
    recall_threshold: float,
    mrr_threshold: float,
    ndcg_threshold: float,
    top_k: int = 40,
) -> RetrievalGateResult:
    """Retrieve each golden query and aggregate deterministic IR metrics.

    `top_k` is the retrieval depth requested from the service (how many
    candidates it returns); `k` is the cutoff the metrics score at (recall@k,
    nDCG@k) and is independent — you typically retrieve a wide candidate set
    but grade the top handful a user would actually see. A query that errors
    at the service raises, exactly like the Triad gate: a gate that could not
    exercise retrieval for an item has not verified it.
    """
    per_query = []
    for item in golden_set:
        request = QueryRequest(
            query=item.query,
            tenant_id=item.tenant_id,
            source_domains=item.source_domains,
            top_k=top_k,
        )
        retrieved = await pipeline_client.retrieve(request)
        # The service returns chunks already ranked (RRF + rerank order,
        # ADR-004/005); preserve that order — position is exactly what MRR and
        # nDCG measure.
        retrieved_ids = [chunk.chunk.chunk_id for chunk in retrieved]
        per_query.append(
            score_retrieval(retrieved_ids, set(item.relevant_chunk_ids), k=k)
        )

    return aggregate_retrieval_metrics(
        per_query,
        k=k,
        recall_threshold=recall_threshold,
        mrr_threshold=mrr_threshold,
        ndcg_threshold=ndcg_threshold,
    )
