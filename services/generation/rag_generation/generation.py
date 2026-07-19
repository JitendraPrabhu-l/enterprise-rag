"""ADR-007/ADR-012: the final-answer generation call itself, via an
OpenAI-compatible client routed through Groq (`rag_core.llm_clients.build_groq_client`).

Calls `client.chat.completions.create(...)` with:

- `model=settings.generation_model` — never hardcoded, always the configured
  model id (a Groq-hosted model, not a Claude model).
- `messages=[...]` — the system + user messages from `prompt_builder`.
- `response_format={"type": "json_object"}` — the widely-supported basic JSON
  mode. We deliberately do NOT request the stricter `json_schema` variant:
  Groq is not guaranteed to honor strict schema enforcement identically
  across every model it hosts. Basic JSON mode plus the JSON shape spelled
  out in the prompt text (see `prompt_builder.JSON_OUTPUT_INSTRUCTIONS`) is
  the more robust combination.
- `temperature=0.1` — a low, grounded-generation temperature, suiting RAG
  synthesis where we want the model to stick closely to retrieved context
  rather than to explore creatively. The OpenAI-compatible API accepts
  arbitrary sampling parameters normally (some prior providers rejected
  non-default sampling parameters entirely; that restriction doesn't apply
  here).

There is no "effort"/reasoning-effort-style parameter equivalent to what some
other structured-output APIs offer; none is passed here.

`generate_structured` returns the raw text of the model's response; parsing
that text into JSON and validating it against `GenerationResponse` is the
caller's job (`rag_generation.guardrails.validate_output`), so retry-on-parse-
failure can be orchestrated by the pipeline without this module knowing about
retries.
"""

from __future__ import annotations

from typing import cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from rag_core.config import BaseServiceSettings
from rag_core.llm_clients import build_groq_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_generation.prompt_builder import ChatMessage


class GenerationCallError(Exception):
    """Raised when the generation API call itself fails (network/auth/rate-
    limit exhaustion) or returns a response with no usable text content."""


class GroqGenerator:
    def __init__(self, settings: BaseServiceSettings, model: str, max_output_tokens: int) -> None:
        self._client: AsyncOpenAI = build_groq_client(settings)
        self._model = model
        self._max_output_tokens = max_output_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=20),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)
        ),
        reraise=True,
    )
    async def generate_structured(self, messages: list[ChatMessage]) -> str:
        """Call the generation model and return the raw response text.

        Raises `GenerationCallError` if the API call fails outright or the
        response contains no message content to parse — never returns an
        empty string silently. Retryable SDK errors (rate limit, 5xx,
        connection failure) are retried by tenacity; non-retryable errors
        (bad request, auth, not-found) propagate immediately as
        `GenerationCallError`.
        """
        # `ChatMessage` (rag_generation.prompt_builder) is deliberately a
        # plain TypedDict so the prompt builder stays SDK-independent; cast
        # to the SDK's own param type only at this one call boundary, where
        # we actually hand the messages to the client.
        sdk_messages = cast(list[ChatCompletionMessageParam], messages)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_output_tokens,
                temperature=0.1,
                messages=sdk_messages,
                response_format={"type": "json_object"},
            )
        except openai.APIError as exc:
            raise GenerationCallError(f"generation API call failed: {exc}") from exc

        if not response.choices:
            raise GenerationCallError("generation response contained no choices")

        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise GenerationCallError(
                "generation response contained no message content "
                f"(finish_reason={choice.finish_reason!r})"
            )
        return str(content)
