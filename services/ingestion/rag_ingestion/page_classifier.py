"""Heuristic page classifier (ADR-001): decides whether a page is dense enough
in tables/figures to warrant a vision-model description pass.

This is a deliberately cheap stand-in for a real layout model. It looks at
signals pypdf's `extract_text()` already gives us for free: how much text
came out at all (figure-heavy pages extract little text), how uniform the
line lengths are (tables produce many short, similarly-sized lines), and the
ratio of non-alphabetic characters (tables are dense with digits/punctuation).
Swap this whole module for an Unstructured.io or vision-based layout
classifier later — `PageClassifier` is the seam.
"""

from __future__ import annotations

import statistics
from enum import Enum
from typing import Protocol

from rag_ingestion.parsing import ParsedPage


class PageCategory(str, Enum):
    TEXT = "text"
    """Ordinary prose page — pypdf extraction is trustworthy as-is."""
    TABLE_OR_FIGURE_DENSE = "table_or_figure_dense"
    """Low text yield or high structural irregularity — route to VisionDescriber."""
    SPARSE = "sparse"
    """Almost no extractable content (e.g. a near-blank or pure-image page)."""


class PageClassifier(Protocol):
    def classify(self, page: ParsedPage) -> PageCategory: ...


class HeuristicPageClassifier:
    """Threshold-based classifier over `PageLayout` stats.

    Parameters are intentionally conservative (favor false positives into
    TABLE_OR_FIGURE_DENSE) since a spurious vision call is cheap relative to
    silently losing a table's content.
    """

    def __init__(
        self,
        *,
        min_chars_for_text_page: int = 200,
        non_alpha_ratio_threshold: float = 0.35,
        line_length_stdev_threshold: float = 6.0,
        min_lines_for_stdev_check: int = 4,
    ) -> None:
        self._min_chars_for_text_page = min_chars_for_text_page
        self._non_alpha_ratio_threshold = non_alpha_ratio_threshold
        self._line_length_stdev_threshold = line_length_stdev_threshold
        self._min_lines_for_stdev_check = min_lines_for_stdev_check

    def classify(self, page: ParsedPage) -> PageCategory:
        layout = page.layout

        if layout.char_count < self._min_chars_for_text_page:
            # Very little text extracted: either a scanned image, a mostly-blank
            # page, or a figure with a short caption. Either way pypdf couldn't
            # get useful text out of it, so treat it as needing vision — unless
            # there's truly nothing there at all.
            if layout.char_count == 0:
                return PageCategory.SPARSE
            return PageCategory.TABLE_OR_FIGURE_DENSE

        if layout.non_alpha_ratio >= self._non_alpha_ratio_threshold:
            return PageCategory.TABLE_OR_FIGURE_DENSE

        line_lengths = self._line_lengths(page)
        if len(line_lengths) >= self._min_lines_for_stdev_check:
            # Tables tend to produce many lines of similar length (columns padded
            # by whitespace) which is actually LOW stdev relative to prose, but
            # prose paragraphs wrapped by pypdf per-line extraction have HIGH
            # variance from short trailing lines vs. full-width lines. We flag
            # the unusual case: many lines, but stdev is near-zero (rigid grid),
            # which is uncommon for natural prose.
            stdev = statistics.pstdev(line_lengths)
            if stdev <= self._line_length_stdev_threshold:
                return PageCategory.TABLE_OR_FIGURE_DENSE

        return PageCategory.TEXT

    @staticmethod
    def _line_lengths(page: ParsedPage) -> list[int]:
        return [len(line) for line in page.text.splitlines() if line.strip()]
