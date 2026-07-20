"""Tests for the deterministic retrieval metrics (ADR-037).

These are pure functions over ranked id lists — no pipeline, no LLM — so the
expected values can be computed by hand and pinned exactly. That exactness is
the whole point of the deterministic gate: unlike the LLM-judged Triad, these
must be reproducible to the decimal.
"""

from __future__ import annotations

import math

import pytest

from rag_eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self) -> None:
        # 1 of 2 relevant ids present → 0.5
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_relevant_beyond_k_not_counted(self) -> None:
        # 'b' sits at rank 4, outside k=3 → only 'a' counts → 0.5
        assert recall_at_k(["a", "x", "y", "b"], {"a", "b"}, k=3) == 0.5

    def test_empty_relevant_set_is_perfect(self) -> None:
        """Nothing to find means nothing was missed — keeps a no-ground-truth
        query from dragging a mean to zero."""
        assert recall_at_k(["a", "b"], set(), k=3) == 1.0

    def test_no_hits_is_zero(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, k=3) == 0.0


class TestReciprocalRank:
    def test_first_position_is_one(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_second_position_is_half(self) -> None:
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_uses_first_relevant_hit(self) -> None:
        # first relevant is at rank 2 → 0.5, even though another relevant is deeper
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_no_hit_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestNdcgAtK:
    def test_perfect_ordering_is_one(self) -> None:
        """All relevant ids first → DCG == IDCG → 1.0."""
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_relevant_lower_reduces_score(self) -> None:
        # one relevant id at rank 2: DCG = 1/log2(3); IDCG (1 relevant) = 1/log2(2) = 1
        expected = (1.0 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a", "y"], {"a"}, k=3) == pytest.approx(expected)

    def test_empty_relevant_set_is_perfect(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=3) == 1.0

    def test_no_hits_is_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a"}, k=3) == 0.0


class TestScoreRetrievalAndAggregate:
    def test_score_retrieval_bundles_all_three(self) -> None:
        m = score_retrieval(["a", "b", "x"], {"a", "b"}, k=3)
        assert m.recall_at_k == 1.0
        assert m.reciprocal_rank == 1.0
        assert m.ndcg_at_k == pytest.approx(1.0)

    def test_aggregate_means_and_thresholds_pass(self) -> None:
        per_query = [
            score_retrieval(["a", "b"], {"a", "b"}, k=2),  # all perfect
            score_retrieval(["x", "a"], {"a"}, k=2),  # recall 1.0, rr 0.5
        ]
        result = aggregate_retrieval_metrics(
            per_query, k=2, recall_threshold=0.9, mrr_threshold=0.7, ndcg_threshold=0.5
        )
        assert result.mean_recall_at_k == pytest.approx(1.0)
        assert result.mean_mrr == pytest.approx(0.75)
        assert result.query_count == 2
        assert result.passed

    def test_aggregate_fails_when_below_threshold(self) -> None:
        per_query = [score_retrieval(["x", "y"], {"a"}, k=2)]  # complete miss
        result = aggregate_retrieval_metrics(
            per_query, k=2, recall_threshold=0.5, mrr_threshold=0.5, ndcg_threshold=0.5
        )
        assert not result.passed

    def test_aggregate_over_zero_queries_raises(self) -> None:
        """A gate that scored nothing has verified nothing — matches the Triad
        gate's stance on an all-errored dataset (ADR-009/ADR-037)."""
        with pytest.raises(ValueError, match="zero queries"):
            aggregate_retrieval_metrics(
                [], k=2, recall_threshold=0.5, mrr_threshold=0.5, ndcg_threshold=0.5
            )
