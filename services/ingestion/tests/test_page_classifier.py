from __future__ import annotations

from rag_ingestion.page_classifier import HeuristicPageClassifier, PageCategory
from rag_ingestion.parsing import ParsedPage, _compute_layout


def _page(text: str, page_number: int = 1) -> ParsedPage:
    return ParsedPage(page_number=page_number, text=text, layout=_compute_layout(text))


class TestHeuristicPageClassifier:
    def test_empty_page_is_sparse(self) -> None:
        classifier = HeuristicPageClassifier()
        assert classifier.classify(_page("")) == PageCategory.SPARSE

    def test_near_empty_page_is_table_or_figure_dense(self) -> None:
        classifier = HeuristicPageClassifier()
        # Non-zero but below the min-chars-for-text-page threshold: likely a
        # figure with a short caption, not truly blank.
        assert classifier.classify(_page("Figure 3.")) == PageCategory.TABLE_OR_FIGURE_DENSE

    def test_normal_prose_page_is_text(self) -> None:
        classifier = HeuristicPageClassifier()
        prose = (
            "This is an ordinary paragraph of prose extracted from a document. "
            "It contains multiple sentences of varying length, describing some "
            "topic in natural language without any tabular structure at all. "
            "The sentence lengths vary quite a bit from short to long ones, "
            "which is typical of natural written English text found in reports "
            "and articles across many different domains and subject areas today."
        )
        assert classifier.classify(_page(prose)) == PageCategory.TEXT

    def test_high_non_alpha_ratio_page_is_table_or_figure_dense(self) -> None:
        classifier = HeuristicPageClassifier()
        # Simulates a numeric table: lots of digits/punctuation, few letters.
        table_like = "\n".join(f"{i},{i * 2},{i * 3.14:.2f},${i * 100}" for i in range(1, 30)) * 2
        assert classifier.classify(_page(table_like)) == PageCategory.TABLE_OR_FIGURE_DENSE

    def test_uniform_short_lines_are_table_or_figure_dense(self) -> None:
        classifier = HeuristicPageClassifier()
        # Many lines of near-identical length (a rigid grid) with enough total
        # characters to clear the min_chars_for_text_page threshold, and kept
        # below the non-alpha-ratio threshold so this exercises the stdev path
        # specifically.
        row = "abcd efgh ijkl"  # 14 chars, letters only
        grid = "\n".join(row for _ in range(20))
        assert classifier.classify(_page(grid)) == PageCategory.TABLE_OR_FIGURE_DENSE

    def test_custom_thresholds_are_respected(self) -> None:
        lenient = HeuristicPageClassifier(non_alpha_ratio_threshold=0.99)
        strict = HeuristicPageClassifier(non_alpha_ratio_threshold=0.01)

        prose_with_some_numbers = (
            "Revenue grew 12 percent in Q3 2024 compared to the prior year period, "
            "driven primarily by strong demand across all three business segments "
            "and continued expansion into new geographic markets during the quarter."
        )
        page = _page(prose_with_some_numbers)

        assert lenient.classify(page) == PageCategory.TEXT
        assert strict.classify(page) == PageCategory.TABLE_OR_FIGURE_DENSE

    def test_min_chars_threshold_is_configurable(self) -> None:
        classifier = HeuristicPageClassifier(min_chars_for_text_page=5)
        assert classifier.classify(_page("Short.")) == PageCategory.TEXT


class TestComputeLayout:
    def test_empty_text_has_zero_stats(self) -> None:
        layout = _compute_layout("")
        assert layout.char_count == 0
        assert layout.line_count == 0
        assert layout.avg_line_length == 0.0
        assert layout.non_alpha_ratio == 0.0

    def test_counts_nonblank_lines_only(self) -> None:
        layout = _compute_layout("line one\n\n   \nline two\n")
        assert layout.line_count == 2

    def test_non_alpha_ratio_reflects_punctuation_density(self) -> None:
        all_letters = _compute_layout("abcdefghij")
        all_digits = _compute_layout("1234567890")
        assert all_letters.non_alpha_ratio == 0.0
        assert all_digits.non_alpha_ratio == 1.0
