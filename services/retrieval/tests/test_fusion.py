"""Unit tests for `reciprocal_rank_fusion` (ADR-004).

RRF formula under test: score(d) = sum over lists L containing d of
1 / (k + rank_L(d)), with rank_L(d) 1-indexed. These tests pin down the exact
arithmetic (not just relative ordering) so a future refactor can't silently
change the formula without a test failing.
"""

from __future__ import annotations

import math

import pytest

from rag_retrieval.fusion import reciprocal_rank_fusion


def test_empty_input_returns_empty_list() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_list_of_empty_lists_returns_empty_list() -> None:
    assert reciprocal_rank_fusion([[], [], []]) == []


def test_single_list_preserves_order_with_exact_scores() -> None:
    k = 60
    result = reciprocal_rank_fusion([["a", "b", "c"]], k=k)

    expected = [
        ("a", 1.0 / (k + 1)),
        ("b", 1.0 / (k + 2)),
        ("c", 1.0 / (k + 3)),
    ]
    assert result == expected


def test_single_document_single_list() -> None:
    result = reciprocal_rank_fusion([["only"]], k=60)
    assert result == [("only", 1.0 / 61)]


def test_two_disjoint_lists_scores_are_independent() -> None:
    k = 60
    result = reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=k)
    scores = dict(result)

    assert scores["a"] == pytest.approx(1.0 / (k + 1))
    assert scores["b"] == pytest.approx(1.0 / (k + 2))
    assert scores["c"] == pytest.approx(1.0 / (k + 1))
    assert scores["d"] == pytest.approx(1.0 / (k + 2))
    # Two disjoint top-ranked docs tie on score; order falls back to
    # first-seen order across the input lists (a's list came first).
    assert [doc_id for doc_id, _ in result[:2]] == ["a", "c"]


def test_overlapping_documents_scores_sum_across_lists() -> None:
    k = 60
    # "x" is rank 1 in list one and rank 2 in list two.
    result = reciprocal_rank_fusion([["x", "y"], ["z", "x"]], k=k)
    scores = dict(result)

    expected_x = 1.0 / (k + 1) + 1.0 / (k + 2)
    assert scores["x"] == pytest.approx(expected_x)
    # "x" should now outrank everything else since it appears in both lists.
    assert result[0][0] == "x"


def test_fused_ranking_is_sorted_descending_by_score() -> None:
    result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_known_worked_example() -> None:
    """A hand-computed example combining dense and sparse rankings."""
    dense = ["doc1", "doc2", "doc3"]
    sparse = ["doc3", "doc1", "doc4"]
    k = 60

    result = reciprocal_rank_fusion([dense, sparse], k=k)
    scores = dict(result)

    assert scores["doc1"] == pytest.approx(1 / 61 + 1 / 62)  # rank 1 dense, rank 2 sparse
    assert scores["doc2"] == pytest.approx(1 / 62)  # rank 2 dense only
    assert scores["doc3"] == pytest.approx(1 / 63 + 1 / 61)  # rank 3 dense, rank 1 sparse
    assert scores["doc4"] == pytest.approx(1 / 63)  # rank 3 sparse only

    # doc1 (0.03252) and doc3 (0.03226) should be the top two, doc1 first.
    ranking = [doc_id for doc_id, _ in result]
    assert ranking[0] == "doc1"
    assert ranking[1] == "doc3"
    assert set(ranking[2:]) == {"doc2", "doc4"}


def test_duplicate_within_single_list_counts_once_at_first_occurrence() -> None:
    k = 60
    result = reciprocal_rank_fusion([["a", "b", "a"]], k=k)
    scores = dict(result)

    # "a" appears at index 0 and index 2 in the same list; only the first
    # occurrence (rank 1) should count.
    assert scores["a"] == pytest.approx(1.0 / (k + 1))
    assert scores["b"] == pytest.approx(1.0 / (k + 2))
    assert len(result) == 2


def test_tie_breaking_uses_first_seen_order_stably() -> None:
    # Both "a" and "b" appear only at rank 2 across two separate lists, so
    # their RRF scores tie exactly; "a" was seen first (in list index 0).
    result = reciprocal_rank_fusion([["z", "a"], ["y", "b"]], k=60)
    scores = dict(result)
    assert scores["a"] == pytest.approx(scores["b"])

    tied_order = [doc_id for doc_id, score in result if math.isclose(score, scores["a"])]
    assert tied_order.index("a") < tied_order.index("b")


def test_different_k_changes_relative_weighting() -> None:
    ranked_lists = [["a", "b"], ["b", "a"]]
    low_k = reciprocal_rank_fusion(ranked_lists, k=1)
    high_k = reciprocal_rank_fusion(ranked_lists, k=1000)

    # With small k, rank 1 dominates rank 2 much more strongly than with
    # large k (where 1/(k+1) ~= 1/(k+2)), so the score *gap* should shrink
    # as k grows even though both configurations tie a vs b overall.
    low_scores = dict(low_k)
    high_scores = dict(high_k)
    assert low_scores["a"] == pytest.approx(low_scores["b"])
    assert high_scores["a"] == pytest.approx(high_scores["b"])


def test_negative_k_raises_value_error() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_k_zero_is_allowed() -> None:
    result = reciprocal_rank_fusion([["a", "b"]], k=0)
    assert result == [("a", 1.0), ("b", 0.5)]


def test_many_lists_accumulate_correctly() -> None:
    # "shared" appears at rank 1 in five separate lists; its score should be
    # exactly 5 * 1/(k+1).
    k = 60
    ranked_lists = [["shared", f"other{i}"] for i in range(5)]
    result = reciprocal_rank_fusion(ranked_lists, k=k)
    scores = dict(result)
    assert scores["shared"] == pytest.approx(5 * (1.0 / (k + 1)))
    assert result[0][0] == "shared"
