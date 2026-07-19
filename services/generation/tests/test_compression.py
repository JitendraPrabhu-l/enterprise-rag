from __future__ import annotations

from rag_generation.compression import compress_context, compress_text
from tests.conftest import make_retrieved_chunk


class TestTableProtection:
    def test_markdown_table_line_survives_untouched(self) -> None:
        table_line = "| Revenue | Q1 | Q2 | Q3 | Q4 |"
        filler = " ".join(
            f"This is some filler sentence number {i} about nothing important at all."
            for i in range(20)
        )
        text = f"{filler}\n{table_line}\n{filler}"

        compressed = compress_text(text, target_ratio=0.1)

        assert table_line in compressed

    def test_whitespace_aligned_columns_survive(self) -> None:
        columnar_line = "Revenue    120000    Expenses    98000    Profit    22000"
        filler = " ".join(
            f"Filler sentence {i} describing generic background context with no numbers."
            for i in range(20)
        )
        text = f"{filler}\n{columnar_line}\n{filler}"

        compressed = compress_text(text, target_ratio=0.1)

        assert columnar_line in compressed

    def test_multiple_table_rows_all_survive(self) -> None:
        rows = [
            "| Name | Score |",
            "| --- | --- |",
            "| Alice | 92 |",
            "| Bob | 87 |",
        ]
        filler = " ".join(f"Generic filler content number {i} with no data." for i in range(15))
        text = filler + "\n" + "\n".join(rows) + "\n" + filler

        compressed = compress_text(text, target_ratio=0.1)

        for row in rows:
            assert row in compressed


class TestNumericPreservation:
    def test_numeric_heavy_sentence_is_preserved(self) -> None:
        numeric_sentence = "Revenue grew 42.7% to $1,284,392 in fiscal year 2024 from 2023."
        filler = " ".join(
            f"This is filler prose sentence {i} that talks about nothing quantitative."
            for i in range(25)
        )
        text = f"{filler} {numeric_sentence} {filler}"

        compressed = compress_text(text, target_ratio=0.05)

        assert numeric_sentence in compressed

    def test_low_numeric_density_sentence_is_not_automatically_protected(self) -> None:
        # A sentence with only one digit-bearing token among many is below the
        # numeric-density protection threshold, so it is eligible for removal
        # like any other prose sentence (not asserting it IS removed, just
        # that the protection mechanism is density-based, not "any digit").
        from rag_generation.compression import _is_numeric_heavy, _tokenize

        low_density = "There were about 2 people in the very large crowded busy noisy room today."
        tokens = _tokenize(low_density)
        assert not _is_numeric_heavy(tokens)


class TestOverallCompression:
    def test_filler_heavy_text_is_reduced(self) -> None:
        rare_sentence = "The zygomorphic bioluminescent phenomenon was first cataloged in 1987."
        filler_sentences = [
            f"The weather was fine and the day continued as expected without incident {i}."
            for i in range(30)
        ]
        text = rare_sentence + " " + " ".join(filler_sentences)

        compressed = compress_text(text, target_ratio=0.3)

        original_tokens = len(text.split())
        compressed_tokens = len(compressed.split())
        assert compressed_tokens < original_tokens
        # Roughly respects the target ratio (allow slack since protected
        # content and greedy selection can overshoot slightly).
        assert compressed_tokens <= original_tokens * 0.6

    def test_target_ratio_one_keeps_everything(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        compressed = compress_text(text, target_ratio=1.0)
        original_words = set(text.split())
        compressed_words = set(compressed.split())
        assert original_words <= compressed_words

    def test_empty_text_returns_empty(self) -> None:
        assert compress_text("", target_ratio=0.5) == ""
        assert compress_text("   ", target_ratio=0.5).strip() == ""


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        text = (
            "Alpha sentence about zebras and quokkas. Beta sentence about ordinary things. "
            "Gamma sentence with 42 numeric values and 3.14 percent growth. "
            "Delta sentence repeating ordinary things again. Epsilon sentence about quokkas."
        )
        first = compress_text(text, target_ratio=0.5)
        second = compress_text(text, target_ratio=0.5)
        assert first == second

    def test_repeated_calls_are_stable_across_many_runs(self) -> None:
        text = " ".join(
            f"Sentence number {i} discusses topic {i % 5} in some detail." for i in range(40)
        )
        results = {compress_text(text, target_ratio=0.4) for _ in range(5)}
        assert len(results) == 1


class TestCompressContext:
    def test_compresses_parent_text_leaves_chunk_text_and_ids_untouched(self) -> None:
        long_text = " ".join(
            f"Filler background sentence {i} with no special meaning whatsoever here."
            for i in range(30)
        )
        rc = make_retrieved_chunk(long_text, parent_id="p-42", document_id="doc-42")

        [compressed_rc] = compress_context([rc], target_ratio=0.3)

        assert compressed_rc.parent.parent_id == "p-42"
        assert compressed_rc.parent.document_id == "doc-42"
        assert len(compressed_rc.parent.text.split()) < len(long_text.split())
        # Child chunk text (the small retrieval unit) is untouched by ADR-008
        # compression — only the larger parent passage is compressed.
        assert compressed_rc.chunk.text == rc.chunk.text

    def test_empty_chunk_list_returns_empty_list(self) -> None:
        assert compress_context([], target_ratio=0.5) == []
