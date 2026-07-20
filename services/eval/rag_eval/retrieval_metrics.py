"""Deterministic, LLM-free retrieval quality metrics (ADR-037).

The RAG Triad (ADR-009) and RAGAS (ADR-017) are LLM-as-judge scores on the
*final answer*. A strong generator can paper over weak retrieval — answer the
question correctly despite the right passage never being retrieved — so an
answer-level gate cannot tell you *where* a regression lives. These metrics
isolate the retrieval stage: given a query with a known-relevant set of
chunk/document ids, they measure whether retrieval surfaced the right
evidence, and how highly it ranked it, with no model in the loop.

Three standard IR metrics, all computed on the *ranked* id list retrieval
returned (order matters):

- **recall@k** — of the known-relevant items, what fraction appear in the
  top-k. The "did we even find it" question; the direct, formal version of
  the "~49% fewer retrieval misses" claim (ADR-023).
- **MRR (mean reciprocal rank)** — 1/rank of the first relevant hit,
  averaged over queries. Rewards putting *a* relevant item high; the metric
  reranking (ADR-005) most directly moves.
- **nDCG@k** — discounted cumulative gain normalized against the ideal
  ordering. Rewards putting *all* relevant items high, discounted by
  position; the most complete ranking-quality signal of the three.

These are cheap and repeatable (no API cost, no judge variance), so they run
per-commit in the same CI gate as the Triad (ADR-009) and pinpoint whether a
score drop is a retrieval regression or a generation one. The golden set they
score against is seeded from production thumbs-down feedback (ADR-027) —
"test sets from real misses," not only synthetic questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    """Per-query retrieval quality, all in [0, 1]."""

    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids present in the top-k retrieved ids.

    Empty relevant set → 1.0 (nothing to find, so nothing was missed) — the
    convention that keeps a query with no ground truth from dragging a mean
    to zero. `k` is clamped to the retrieved length; asking for recall@10 on
    5 results scores over the 5 that exist, not over a padded 10.
    """
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (1-based rank of the first relevant id), or 0.0 if none appear.

    The per-query term MRR averages. First relevant hit at position 1 → 1.0,
    at position 2 → 0.5, and so on; no relevant hit anywhere → 0.0.
    """
    if not relevant_ids:
        return 1.0
    for index, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-gain nDCG@k: DCG of the actual top-k over the ideal DCG.

    Gain is 1 for a relevant id, 0 otherwise (binary relevance — the golden
    set marks ids relevant/not, not graded), discounted by log2(rank+1). The
    ideal ordering places every relevant id first, capped at k, so the score
    is 1.0 exactly when the top-k is all-relevant-first. Empty relevant set →
    1.0, consistent with recall.
    """
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, rid in enumerate(top_k, start=1)
        if rid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 1.0
    return dcg / idcg


def score_retrieval(
    retrieved_ids: list[str], relevant_ids: set[str], *, k: int
) -> RetrievalMetrics:
    """All three metrics for one query's ranked retrieval output."""
    return RetrievalMetrics(
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k),
    )


@dataclass
class RetrievalGateResult:
    """Aggregated retrieval metrics across a golden set, with threshold checks."""

    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    k: int
    recall_threshold: float
    mrr_threshold: float
    ndcg_threshold: float
    query_count: int

    @property
    def passed(self) -> bool:
        """Every mean must meet its threshold — a retrieval-stage CI gate that
        fails independently of, and complementary to, the answer-level Triad
        gate (ADR-009). A drop here localizes the regression to retrieval."""
        return (
            self.mean_recall_at_k >= self.recall_threshold
            and self.mean_mrr >= self.mrr_threshold
            and self.mean_ndcg_at_k >= self.ndcg_threshold
        )


def aggregate_retrieval_metrics(
    per_query: list[RetrievalMetrics],
    *,
    k: int,
    recall_threshold: float,
    mrr_threshold: float,
    ndcg_threshold: float,
) -> RetrievalGateResult:
    """Mean each metric across queries and compare to thresholds.

    Raises on an empty list rather than reporting a vacuous pass — a gate that
    scored zero queries has not verified anything, the same stance the Triad
    gate takes on an all-errored dataset (ADR-009).
    """
    if not per_query:
        raise ValueError("cannot aggregate retrieval metrics over zero queries")
    n = len(per_query)
    return RetrievalGateResult(
        mean_recall_at_k=sum(m.recall_at_k for m in per_query) / n,
        mean_mrr=sum(m.reciprocal_rank for m in per_query) / n,
        mean_ndcg_at_k=sum(m.ndcg_at_k for m in per_query) / n,
        k=k,
        recall_threshold=recall_threshold,
        mrr_threshold=mrr_threshold,
        ndcg_threshold=ndcg_threshold,
        query_count=n,
    )
