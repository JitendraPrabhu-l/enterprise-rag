"""Baseline PDF/text extraction (ADR-001 fallback parser: pypdf, no Unstructured.io).

Produces page-level text plus enough layout signal (line count, average line
length, whitespace ratio) for `page_classifier.PageClassifier` to decide
whether a page needs a vision-model pass. Kept behind a narrow interface so a
real layout-aware parser (Unstructured.io, etc.) can be swapped in later
without touching `pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class PageLayout:
    """Cheap layout statistics used by the heuristic page classifier."""

    line_count: int
    avg_line_length: float
    non_alpha_ratio: float
    """Fraction of non-alphabetic, non-whitespace characters — high values (lots of
    pipes, digits, dashes) are a weak proxy for tables; low text density with wide
    line-length variance is a weak proxy for figures with captions."""
    char_count: int


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    """1-indexed page number, matching how humans and citations refer to pages."""
    text: str
    layout: PageLayout


@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    source_path: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def _compute_layout(text: str) -> PageLayout:
    lines = [line for line in text.splitlines() if line.strip()]
    line_count = len(lines)
    avg_line_length = sum(len(line) for line in lines) / line_count if line_count else 0.0

    stripped = text.strip()
    if stripped:
        non_alpha = sum(1 for ch in stripped if not ch.isalpha() and not ch.isspace())
        non_alpha_ratio = non_alpha / len(stripped)
    else:
        non_alpha_ratio = 0.0

    return PageLayout(
        line_count=line_count,
        avg_line_length=avg_line_length,
        non_alpha_ratio=non_alpha_ratio,
        char_count=len(stripped),
    )


def parse_pdf(path: str | Path, document_id: str) -> ParsedDocument:
    """Extract per-page text and layout stats from a PDF using pypdf.

    Raises `PdfReadError` (from pypdf) on a corrupt/unreadable file — callers
    should treat that as an ingestion failure for this document, not retry
    blindly, since the bytes themselves are the problem.
    """
    resolved = Path(path)
    reader = PdfReader(str(resolved))

    pages: list[ParsedPage] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except PdfReadError:
            text = ""
        pages.append(ParsedPage(page_number=index + 1, text=text, layout=_compute_layout(text)))

    return ParsedDocument(document_id=document_id, source_path=str(resolved), pages=pages)


def parse_text_file(path: str | Path, document_id: str) -> ParsedDocument:
    """Treat a plain-text file as a single page — used for .txt/.md sources."""
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8", errors="replace")
    page = ParsedPage(page_number=1, text=text, layout=_compute_layout(text))
    return ParsedDocument(document_id=document_id, source_path=str(resolved), pages=[page])


def parse_document(path: str | Path, document_id: str) -> ParsedDocument:
    """Dispatch on file extension; the one entry point `pipeline.py` calls."""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(resolved, document_id)
    if suffix in (".txt", ".md"):
        return parse_text_file(resolved, document_id)
    raise ValueError(f"Unsupported file extension for ingestion: {suffix!r}")
