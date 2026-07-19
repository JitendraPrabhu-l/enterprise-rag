"""LLM-as-judge scoring functions implementing the RAG Triad (ADR-009).

Each function asks the utility-tier model (served via Groq, an
OpenAI-compatible chat completions API — per ADR cost posture) to score one
axis of a generated answer against retrieved context. Basic JSON mode
(`response_format={"type": "json_object"}`) is combined with an explicit
instruction in the prompt describing the exact expected JSON shape, since
strict JSON-schema enforcement is not reliably supported across all
Groq-hosted models. The raw JSON is then parsed into a `TriadScore` pydantic
model.

These are real, working judges — no stubs. A malformed or missing judge
response raises `JudgeResponseError` rather than silently defaulting to 0.0,
because a silent default would corrupt the CI/CD gate's pass/fail decision.
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

from rag_eval.schemas import TriadScore

_T = TypeVar("_T")

_JUDGE_MAX_TOKENS = 512

_JSON_SHAPE_INSTRUCTION = """\
Respond with ONLY a JSON object of the form \
{"score": <float between 0.0 and 1.0>, "justification": "<one-sentence justification>"} \
-- no markdown, no code fences, no extra keys, no other text before or after the JSON."""


class JudgeError(Exception):
    """Base class for judge-call failures."""


class JudgeResponseError(JudgeError):
    """Raised when the judge model's response cannot be parsed into a TriadScore.

    Deliberately NOT swallowed into a default score — a judge failure must be
    visible to the caller (and, in the CI/CD gate, treated as a hard failure)
    rather than silently reported as 0.0, which would be indistinguishable
    from a genuinely bad answer.
    """


_FAITHFULNESS_BASE_TEMPLATE = """\
You are an exacting fact-checker evaluating whether a generated answer is fully \
grounded in its supporting context.

Your task: identify every factual claim made in the ANSWER, and determine whether \
each claim is directly supported by the CONTEXT. An answer is "faithful" if it makes \
no claims that go beyond what the context supports — no fabricated facts, no invented \
specifics (numbers, names, dates), no unsupported extrapolation, and no contradiction \
of the context.

Score on a continuous scale from 0.0 to 1.0:
- 1.0 = every claim in the answer is directly supported by the context. Fully grounded.
- 0.5 = roughly half of the substantive claims are supported; the rest are unsupported \
or unverifiable from the context.
- 0.0 = the answer is substantially fabricated or contradicts the context.

Judge ONLY whether claims are grounded in the given context — do not judge whether the \
answer is a good or complete answer to the question; that is scored separately.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with a score and a one-line justification identifying the least-grounded claim \
(or confirming full grounding if score is 1.0)."""

_ANSWER_RELEVANCE_BASE_TEMPLATE = """\
You are evaluating whether a generated answer actually addresses the user's specific \
question.

Your task: judge how directly and completely the ANSWER addresses the QUESTION, \
independent of whether the answer's claims are factually correct or grounded (that is \
scored separately). Penalize answers that are evasive, off-topic, address a different \
question than the one asked, omit parts of a multi-part question, or are needlessly \
padded with irrelevant information instead of answering directly.

Score on a continuous scale from 0.0 to 1.0:
- 1.0 = the answer directly and completely addresses every part of the question.
- 0.5 = the answer partially addresses the question, or addresses it but with \
significant irrelevant padding or an incomplete response to a multi-part question.
- 0.0 = the answer does not address the question at all (off-topic, non-answer, or \
answers a different question).

QUESTION:
{question}

ANSWER:
{answer}

Respond with a score and a one-line justification."""

_CONTEXT_PRECISION_BASE_TEMPLATE = """\
You are evaluating retrieval quality: of the chunks retrieved for a question, what \
fraction are actually relevant to answering it.

Your task: for each numbered chunk in RETRIEVED_CHUNKS, decide whether it contains \
information relevant to answering the QUESTION (directly useful for constructing an \
answer, not merely on the same general topic). Then compute the fraction of chunks \
that are relevant.

Score = (number of relevant chunks) / (total number of chunks), so the score is always \
one of a discrete set of fractions from 0.0 (no relevant chunks) to 1.0 (every chunk \
relevant). This measures signal vs. noise in retrieval, not answer quality.

QUESTION:
{question}

RETRIEVED_CHUNKS:
{numbered_chunks}

Respond with the computed score and a one-line justification naming which chunks (by \
number) were irrelevant, if any."""

# NOTE: the JSON-shape instruction is appended AFTER `.format()` is called on
# the base templates above, not concatenated into the template string itself
# — it contains literal `{` `}` braces that would otherwise be mis-parsed as
# str.format() replacement fields.
FAITHFULNESS_PROMPT_TEMPLATE = _FAITHFULNESS_BASE_TEMPLATE
ANSWER_RELEVANCE_PROMPT_TEMPLATE = _ANSWER_RELEVANCE_BASE_TEMPLATE
CONTEXT_PRECISION_PROMPT_TEMPLATE = _CONTEXT_PRECISION_BASE_TEMPLATE


def _format_numbered_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "(no chunks were retrieved)"
    return "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))


def _parse_score_response(raw_text: str) -> TriadScore:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(
            f"Judge response was not valid JSON despite JSON-mode + shape instructions: "
            f"{raw_text!r}"
        ) from exc

    try:
        return TriadScore.model_validate(payload)
    except ValidationError as exc:
        raise JudgeResponseError(
            f"Judge response JSON did not match the TriadScore schema: {payload!r}"
        ) from exc


def _extract_message_text(response: ChatCompletion) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise JudgeResponseError("Judge response had no choices.")
    message = choices[0].message
    text = getattr(message, "content", None)
    if isinstance(text, str) and text:
        return text
    raise JudgeResponseError("Judge response contained no message content.")


def _judge_retry(
    max_attempts: int,
) -> Callable[[Callable[[], Awaitable[_T]]], Callable[[], Awaitable[_T]]]:
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError, openai.APIError)),
    )


async def _call_judge(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> TriadScore:
    """Call the judge model with JSON mode and parse the result into a TriadScore.

    Network/timeout/API errors are retried via tenacity; a malformed or
    missing response raises `JudgeResponseError` immediately (not retried,
    since a well-formed request that comes back malformed is not a transient
    condition retrying would fix).
    """

    @_judge_retry(max_retries)
    async def _do_call() -> ChatCompletion:
        return await client.chat.completions.create(
            model=model,
            max_tokens=_JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

    response = await _do_call()
    raw_text = _extract_message_text(response)
    return _parse_score_response(raw_text)


async def score_faithfulness(
    client: AsyncOpenAI,
    *,
    model: str,
    answer: str,
    context: list[str],
    max_retries: int = 3,
) -> TriadScore:
    """Score whether `answer` contains claims not supported by `context` (ADR-009 axis 1).

    1.0 = fully grounded in the retrieved context; 0.0 = substantially fabricated.
    """
    prompt = (
        FAITHFULNESS_PROMPT_TEMPLATE.format(
            context=_format_numbered_chunks(context),
            answer=answer,
        )
        + "\n\n"
        + _JSON_SHAPE_INSTRUCTION
    )
    return await _call_judge(client, model=model, prompt=prompt, max_retries=max_retries)


async def score_answer_relevance(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    answer: str,
    max_retries: int = 3,
) -> TriadScore:
    """Score whether `answer` actually addresses `question` (ADR-009 axis 2).

    1.0 = fully addresses the question; 0.0 = off-topic or non-answer.
    """
    prompt = (
        ANSWER_RELEVANCE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        + "\n\n"
        + _JSON_SHAPE_INSTRUCTION
    )
    return await _call_judge(client, model=model, prompt=prompt, max_retries=max_retries)


async def score_context_precision(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    retrieved_chunks: list[str],
    max_retries: int = 3,
) -> TriadScore:
    """Score what fraction of `retrieved_chunks` are relevant to `question` (ADR-009 axis 3).

    1.0 = every retrieved chunk is relevant (no noise); 0.0 = none are.
    """
    prompt = (
        CONTEXT_PRECISION_PROMPT_TEMPLATE.format(
            question=question,
            numbered_chunks=_format_numbered_chunks(retrieved_chunks),
        )
        + "\n\n"
        + _JSON_SHAPE_INSTRUCTION
    )
    return await _call_judge(client, model=model, prompt=prompt, max_retries=max_retries)
