"""Builds the OpenAI-compatible chat `messages` array for the final-answer
generation call (final-answer synthesis via Groq, ADR-007/ADR-012).

The standard OpenAI-compatible chat completions API has no client-side
prompt-caching markup (no `cache_control`, no "breakpoints") — some providers
cache repeated prefixes transparently server-side, with no API parameter
required at all. We therefore don't attempt to signal caching explicitly.
That said, the original ordering principle is still worth preserving even
without an explicit knob for it: STABLE, LARGE content (the system prompt)
is rendered first and VOLATILE, per-query content (compressed retrieved
context + the user's question) is rendered last, in its own message. If the
provider does any transparent prefix caching, an unchanging system message
followed by a varying user message remains the layout most likely to benefit
from it.

Structured outputs: the OpenAI-compatible `response_format` parameter varies
in how strictly it's supported across models (see `generation.py` for how
`{"type": "json_object"}` is requested). As a belt-and-suspenders measure,
this module also embeds the exact expected JSON shape directly in the prompt
text, so the model has explicit written instructions to follow even if the
provider/model doesn't enforce a schema server-side.
"""

from __future__ import annotations

from typing import TypedDict

from rag_core.schemas import RetrievedChunk

JSON_OUTPUT_INSTRUCTIONS = """\
Respond with ONLY a single JSON object (no markdown code fences, no text \
before or after it) matching exactly this shape:

{
  "answer": "<string: the grounded answer to the user's question>",
  "citations": [
    {
      "parent_id": "<string>",
      "document_id": "<string>",
      "page_number": <integer or null>
    }
  ]
}

`citations` must contain one entry per distinct context passage actually used \
to support the answer, using the `parent_id`, `document_id`, and \
`page_number` exactly as given in the context metadata for that passage. If \
the context does not support an answer, return an empty `citations` array.\
"""


class ChatMessage(TypedDict):
    role: str
    content: str


def _format_chunk(chunk: RetrievedChunk, index: int) -> str:
    parent = chunk.parent
    page = f", page {parent.page_number}" if parent.page_number is not None else ""
    header = (
        f"[Passage {index + 1}] parent_id={parent.parent_id} "
        f"document_id={parent.document_id}{page} modality={parent.modality.value}"
    )
    return f"{header}\n{parent.text}"


def build_context_block(compressed_chunks: list[RetrievedChunk]) -> str:
    """Render the compressed, guardrail-filtered chunks into a single
    volatile text block, one labeled passage per chunk, in ranking order."""
    if not compressed_chunks:
        return "No relevant context passages were retrieved for this query."
    sections = [_format_chunk(c, i) for i, c in enumerate(compressed_chunks)]
    return "## Context\n\n" + "\n\n".join(sections)


def build_prompt(
    query: str,
    compressed_chunks: list[RetrievedChunk],
    system_prompt: str,
) -> list[ChatMessage]:
    """Build the `messages` array for the chat completions call.

    Returns a two-message list: a `system` message carrying the stable
    system prompt (rendered first, since it's the same across requests) plus
    the JSON-output instructions, followed by a `user` message carrying the
    volatile per-query content (compressed context + the question). Ordering
    stable-then-volatile is retained purely as a hedge for any transparent
    provider-side prefix caching — see module docstring.
    """
    system_content = f"{system_prompt}\n\n{JSON_OUTPUT_INSTRUCTIONS}"

    context_block = build_context_block(compressed_chunks)
    user_content = f"{context_block}\n\n## Question\n\n{query}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
