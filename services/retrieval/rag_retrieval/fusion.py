"""Reciprocal Rank Fusion (ADR-004): fuse multiple ranked lists into one.

Pure, side-effect-free, and independently unit-testable — no I/O, no external
clients. This is the one place the RRF formula is implemented; every caller
(dense+sparse fusion, multi-query result merging) goes through this function.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists of document IDs using Reciprocal Rank Fusion.

    RRF formula: for a document `d`, its fused score is

        score(d) = sum over each ranked list L containing d of  1 / (k + rank_L(d))

    where `rank_L(d)` is the 1-indexed position of `d` in list `L` (i.e. the
    top result has rank 1, not 0). Documents absent from a given list simply
    contribute 0 from that list. `k` is a smoothing constant (60 is the
    standard value from the original RRF paper) that dampens the influence of
    very high ranks and keeps low-ranked-but-present documents from being
    drowned out.

    Args:
        ranked_lists: One list per ranker (e.g. dense search, BM25 search, or
            one list per paraphrased query variant). Each inner list is a
            sequence of document IDs already sorted best-first. IDs may repeat
            across lists (that's the whole point of fusion) but must not
            repeat *within* a single list — only the first occurrence in a
            list is used if they do.
        k: RRF constant. Defaults to 60, the standard value.

    Returns:
        A list of `(document_id, fused_score)` tuples sorted by fused_score
        descending (ties broken by first-seen order, which is stable given
        Python's sort stability). Empty input yields an empty list.
    """
    if k < 0:
        raise ValueError(f"RRF constant k must be non-negative, got {k}")

    scores: dict[str, float] = {}
    first_seen_order: dict[str, int] = {}
    sequence = 0

    for ranked_list in ranked_lists:
        seen_in_this_list: set[str] = set()
        for rank_zero_indexed, doc_id in enumerate(ranked_list):
            if doc_id in seen_in_this_list:
                # Only the first occurrence within a single list counts.
                continue
            seen_in_this_list.add(doc_id)

            rank = rank_zero_indexed + 1  # 1-indexed rank per the RRF formula
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            if doc_id not in first_seen_order:
                first_seen_order[doc_id] = sequence
                sequence += 1

    fused = sorted(
        scores.items(),
        key=lambda item: (-item[1], first_seen_order[item[0]]),
    )
    return fused
