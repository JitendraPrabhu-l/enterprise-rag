"""Synthetic question/context/answer triplet generation for repeatable eval sets.

For each source passage, ask the utility-tier model (served via Groq, an
OpenAI-compatible chat completions API) to generate a realistic user question
that passage would answer, plus a reference answer grounded in that passage.
This builds a repeatable `SyntheticEvalItem` dataset used to drive the CI/CD
gate (ADR-009) without depending on hand-curated eval data.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_eval.schemas import SyntheticEvalItem

_T = TypeVar("_T")

_GENERATION_MAX_TOKENS = 1024

_JSON_SHAPE_INSTRUCTION = """\
Respond with ONLY a JSON object of the form \
{"question": "<question text>", "reference_answer": "<answer text>"} \
-- no markdown, no code fences, no extra keys, no other text before or after the JSON."""

# NOTE: the JSON-shape instruction is appended AFTER `.format()` is called on
# this template, not concatenated into the template string itself — it
# contains literal `{` `}` braces that would otherwise be mis-parsed as
# str.format() replacement fields.
SYNTHETIC_QA_PROMPT_TEMPLATE = """\
You are building an evaluation dataset for a retrieval-augmented generation (RAG) \
system. Given a source passage, generate one realistic user question that this \
passage — and only this passage — would answer well, plus a reference answer.

Requirements:
- The question must be answerable using only the information in the passage.
- The question should read like something a real user would type, not a reading- \
comprehension quiz item (avoid phrasing like "According to the passage...").
- The reference answer must be fully grounded in the passage — no outside knowledge, \
no fabricated details.
- The reference answer should be concise (1-4 sentences) and directly answer the \
question.

PASSAGE:
{passage}

Respond with the question and reference answer as structured JSON."""


class SyntheticDataError(Exception):
    """Base class for synthetic dataset generation failures."""


class SyntheticDataResponseError(SyntheticDataError):
    """Raised when the model's response cannot be parsed into question + reference answer."""


def _judge_retry(
    max_attempts: int,
) -> Callable[[Callable[[], Awaitable[_T]]], Callable[[], Awaitable[_T]]]:
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, openai.APIError)),
    )


def _extract_message_text(response: ChatCompletion) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise SyntheticDataResponseError("Generation response had no choices.")
    message = choices[0].message
    text = getattr(message, "content", None)
    if isinstance(text, str) and text:
        return text
    raise SyntheticDataResponseError("Generation response contained no message content.")


async def _generate_one(
    client: AsyncOpenAI,
    *,
    model: str,
    passage: str,
    document_id: str,
    max_retries: int,
) -> SyntheticEvalItem:
    prompt = SYNTHETIC_QA_PROMPT_TEMPLATE.format(passage=passage) + "\n\n" + _JSON_SHAPE_INSTRUCTION

    @_judge_retry(max_retries)
    async def _do_call() -> ChatCompletion:
        return await client.chat.completions.create(
            model=model,
            max_tokens=_GENERATION_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

    response = await _do_call()
    raw_text = _extract_message_text(response)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SyntheticDataResponseError(
            f"Synthetic-data response was not valid JSON: {raw_text!r}"
        ) from exc

    try:
        question = str(payload["question"])
        reference_answer = str(payload["reference_answer"])
    except (KeyError, TypeError) as exc:
        raise SyntheticDataResponseError(
            f"Synthetic-data response JSON missing required fields: {payload!r}"
        ) from exc

    try:
        return SyntheticEvalItem(
            question=question,
            reference_context=passage,
            reference_answer=reference_answer,
            source_document_id=document_id,
        )
    except ValidationError as exc:
        raise SyntheticDataResponseError(
            f"Generated fields failed SyntheticEvalItem validation: {payload!r}"
        ) from exc


async def generate_synthetic_eval_set(
    client: AsyncOpenAI,
    *,
    model: str,
    source_documents: list[tuple[str, str]],
    max_retries: int = 3,
) -> list[SyntheticEvalItem]:
    """Generate one `SyntheticEvalItem` per source document.

    Args:
        client: An `AsyncOpenAI` client (routed to Groq's utility tier).
        model: The utility model to use (`settings.utility_model`).
        source_documents: A list of `(document_id, passage_text)` pairs. One
            synthetic item is generated per passage.
        max_retries: Tenacity retry attempts per passage for transient
            network/timeout/API errors.

    Returns:
        A list of `SyntheticEvalItem`, one per input document, in the same
        order as `source_documents`.

    Raises:
        SyntheticDataResponseError: if any passage's generated response cannot
            be parsed into a valid item. Not swallowed — a bad synthetic item
            would silently corrupt the eval set used to gate CI/CD.
    """
    items: list[SyntheticEvalItem] = []
    for document_id, passage in source_documents:
        item = await _generate_one(
            client,
            model=model,
            passage=passage,
            document_id=document_id,
            max_retries=max_retries,
        )
        items.append(item)
    return items
