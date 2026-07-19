"""ADR-010: defense-in-depth guardrails.

1. `scan_for_injection` — a real regex/keyword heuristic classifier run on
   BOTH the user's query and every retrieved chunk's text before either is
   allowed into the prompt. Flagged text must never reach the prompt
   silently; callers are responsible for excluding it (see
   `rag_generation.pipeline`) and setting `guardrail_flagged=True`.
2. `validate_output` — parses the raw JSON the model returned into
   `GenerationResponse`. This is a real error path: if parsing fails the
   caller should retry once with a stricter instruction (see
   `build_retry_instruction`) and finally raise `OutputValidationError` if it
   still fails. No silent swallowing.
3. `find_ungrounded_citations` (ADR-028) is re-exported here from
   `rag_core.citation_verification` for backward-compatible import
   convenience — it lives in rag_core because the eval service needs the
   identical check for its own eval-time metric and has no dependency on
   this package.
"""

from __future__ import annotations

import json
import re
from uuid import UUID

from pydantic import ValidationError
from rag_core.citation_verification import find_ungrounded_citations
from rag_core.schemas import Citation, GenerationResponse

__all__ = [
    "OutputValidationError",
    "build_retry_instruction",
    "find_ungrounded_citations",
    "scan_for_injection",
    "validate_output",
]

# Prompt-injection heuristics. Patterns are intentionally broad but anchored
# on well-known injection phrasing so we don't flag ordinary questions that
# merely mention "instructions" or "system" in a benign sense (e.g. "what are
# the system requirements for X?").
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions?\b",
        re.I,
    ),
    re.compile(r"\bdisregard\s+(the\s+)?(system\s+prompt|previous|prior|above)\b", re.I),
    re.compile(r"\boverride\s+(the\s+)?(system\s+prompt|your\s+instructions?)\b", re.I),
    re.compile(
        r"\bforget\s+(all\s+|everything\s+)?(the\s+)?(previous|prior|above)\s+"
        r"(instructions?|context)\b",
        re.I,
    ),
    re.compile(r"\byou\s+are\s+now\s+(a|an|in)\b.{0,40}\bmode\b", re.I),
    re.compile(
        r"\bact\s+as\s+(if\s+you\s+(are|were)|a)\b.{0,40}\b"
        r"(unfiltered|jailbreak|dan|no\s+restrictions?)\b",
        re.I,
    ),
    re.compile(
        r"\bpretend\s+(you\s+are|to\s+be)\b.{0,40}\b(unfiltered|no\s+rules?|no\s+restrictions?)\b",
        re.I,
    ),
    re.compile(r"\breveal\s+(your\s+|the\s+)?(system\s+prompt|hidden\s+instructions?)\b", re.I),
    re.compile(r"\bprint\s+(your\s+|the\s+)?(system\s+prompt|initial\s+instructions?)\b", re.I),
    re.compile(r"\bnew\s+instructions?\s*:\s*", re.I),
    re.compile(r"\[\s*system\s*\]", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\bassistant\s*:\s*.{0,20}\bsure[, ]", re.I),
    re.compile(r"\bdo\s+not\s+(follow|obey)\s+(the\s+)?(system|original|previous)\b", re.I),
    re.compile(r"###\s*(instruction|system|admin)\b", re.I),
    re.compile(r"\bhidden\s+instructions?\b", re.I),
    re.compile(r"\bthis\s+is\s+(a\s+|the\s+)?(new\s+)?(system|admin|developer)\s+message\b", re.I),
    re.compile(r"\bsudo\s+mode\b", re.I),
    re.compile(r"\byou\s+must\s+(now\s+)?comply\b", re.I),
    re.compile(
        r"\bdisable\s+(all\s+)?(your\s+)?(safety|content)\s+(filters?|guardrails?|policies)\b",
        re.I,
    ),
)


def scan_for_injection(text: str) -> bool:
    """Heuristic prompt-injection classifier.

    Returns True if `text` matches a known injection pattern. Deterministic
    and dependency-free — pure regex over well-documented injection markers
    (instruction-override phrasing, fake role-play framing, hidden system/
    admin message markers). False positives are minimized by anchoring each
    pattern on multi-word phrasing rather than single trigger words.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


class OutputValidationError(Exception):
    """Raised when the model's structured output cannot be parsed into
    `GenerationResponse` even after one stricter-instruction retry."""


_RETRY_INSTRUCTION = (
    "Your previous response could not be parsed as valid JSON matching the "
    "required schema. Respond with ONLY a single, syntactically valid JSON "
    "object with exactly two top-level fields: `answer` (a string) and "
    "`citations` (an array of objects, each with `parent_id`, `document_id`, "
    "and optional `page_number`). Do not include any markdown code fences, "
    "explanation, or text before or after the JSON object."
)


def build_retry_instruction() -> str:
    """User-turn text to append when retrying a failed structured-output call."""
    return _RETRY_INSTRUCTION


def validate_output(
    raw_json: str, *, request_id: UUID, model: str, used_graph: bool
) -> GenerationResponse:
    """Parse and validate the model's raw JSON text into a `GenerationResponse`.

    Raises `OutputValidationError` (never swallows) if `raw_json` is not
    valid JSON, or does not conform to the expected `answer`/`citations`
    shape. Callers should catch this once, retry with
    `build_retry_instruction()` appended to the prompt, and let a second
    failure propagate.
    """
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"model output was not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise OutputValidationError("model output JSON was not an object")

    answer = payload.get("answer")
    raw_citations = payload.get("citations", [])
    if not isinstance(answer, str):
        raise OutputValidationError("model output JSON missing string 'answer' field")
    if not isinstance(raw_citations, list):
        raise OutputValidationError("model output JSON 'citations' field was not an array")

    try:
        citations = [Citation.model_validate(c) for c in raw_citations]
        return GenerationResponse(
            request_id=request_id,
            answer=answer,
            citations=citations,
            model=model,
            used_graph=used_graph,
            guardrail_flagged=False,
        )
    except ValidationError as exc:
        raise OutputValidationError(f"model output failed schema validation: {exc}") from exc
