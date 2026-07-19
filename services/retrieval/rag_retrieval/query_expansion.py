"""Query transformation strategies: multi-query expansion and HyDE.

Both strategies call the Groq utility model (`settings.utility_model`) via an
OpenAI-compatible chat completions client to transform the user's raw query
before it hits the hybrid search stage:

- multi_query: generate 2-3 paraphrased variations of the query, so lexical
  mismatches between the user's phrasing and the corpus's phrasing don't sink
  recall. Each variation is later run through hybrid search independently and
  the result lists are merged/deduped before RRF fusion.
- hyde (Hypothetical Document Embeddings): ask the model to write a
  hypothetical *answer* passage, then embed that passage (not the raw query)
  with the same local embedding model used at ingestion. The intuition is
  that an answer-shaped passage is closer, in embedding space, to real answer
  passages in the corpus than a short interrogative query is.

Both are real, callable, independently testable functions — no stubs.
"""

from __future__ import annotations

import structlog
from openai import APIError, AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_MULTI_QUERY_SYSTEM_PROMPT = (
    "You rewrite search queries for a retrieval system. Given a user's query, "
    "produce alternative phrasings that preserve the original meaning and intent "
    "but vary vocabulary, phrasing, and specificity so that lexical and semantic "
    "search can find relevant passages the original wording might miss. "
    "Output exactly one rewritten query per line, with no numbering, no "
    "bullets, no quotation marks, and no commentary — just the raw query text."
)

_HYDE_SYSTEM_PROMPT = (
    "You write hypothetical answer passages for a retrieval system (HyDE). "
    "Given a user's query, write a short, factual, self-contained passage (3-6 "
    "sentences) that would plausibly appear in a document and directly answer "
    "the query. Write it as if it were an excerpt from a real reference "
    "document — do not mention the question, do not hedge, do not say you are "
    "unsure. Output only the passage text, with no preamble or commentary."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You decompose complex questions for a retrieval system. A question is "
    "multi-hop when answering it requires combining evidence about DIFFERENT "
    "entities or facts (comparisons, cause chains, aggregations across "
    "documents). Given such a question, output the 2-4 simpler single-fact "
    "sub-questions whose answers together answer it, one per line, no "
    "numbering, no commentary. If the question is already a single-fact "
    "lookup, output the single word NONE."
)


class QueryExpansionError(RuntimeError):
    """Raised when the model fails to produce usable expansion output."""


class QueryExpander:
    """Generates multi-query paraphrases and HyDE hypothetical passages via Groq."""

    def __init__(self, client: AsyncOpenAI, model: str, max_tokens: int = 512) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(APIError),
    )
    async def _complete(self, system: str, user_content: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise QueryExpansionError(f"model returned no text content for model={self._model!r}")
        return text

    async def expand_multi_query(self, query: str, num_variations: int = 3) -> list[str]:
        """Generate `num_variations` paraphrased variants of `query` via Groq.

        Returns the original query plus up to `num_variations` paraphrases,
        deduplicated case-insensitively while preserving order. Never returns
        an empty list — on any parsing shortfall the original query alone is
        still included.
        """
        if num_variations < 1:
            raise ValueError(f"num_variations must be >= 1, got {num_variations}")

        user_content = f"Generate {num_variations} alternative phrasings of this query:\n\n{query}"
        raw = await self._complete(_MULTI_QUERY_SYSTEM_PROMPT, user_content)

        variations = [line.strip() for line in raw.splitlines() if line.strip()]

        seen: set[str] = {query.strip().lower()}
        deduped: list[str] = [query]
        for variation in variations:
            key = variation.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(variation)
            if len(deduped) - 1 >= num_variations:
                break

        logger.info(
            "query_expansion.multi_query",
            original=query,
            variation_count=len(deduped) - 1,
        )
        return deduped

    async def generate_hyde_passage(self, query: str) -> str:
        """Generate a hypothetical answer passage for `query` via Groq (HyDE).

        The caller is responsible for embedding the returned passage with the
        same embedding model used at ingestion — this function only produces
        the text.
        """
        passage = await self._complete(_HYDE_SYSTEM_PROMPT, query)
        logger.info("query_expansion.hyde", original=query, passage_chars=len(passage))
        return passage

    async def decompose(self, query: str, max_subquestions: int = 4) -> list[str]:
        """Decompose a multi-hop `query` into single-fact sub-questions (ADR-025).

        Research finding this implements: for multi-hop QA, running a
        retrieve-per-sub-question loop captures most of the gain over a single
        retrieval, and query decomposition is a statistically significant
        contributor on its own. Each returned sub-question is hybrid-searched
        independently and the candidate sets are merged before RRF — so a
        comparison question ("how does X's revenue compare to Y's") retrieves
        BOTH entities' passages instead of neither.

        Returns the original query plus its sub-questions (deduped, order
        preserved). A single-fact query — the model replies NONE — returns
        just `[query]`, making decomposition a safe no-op there. The original
        is always kept: sub-questions supplement it, never replace it, so a
        bad decomposition can only add candidates, not remove the direct hit.
        """
        raw = await self._complete(_DECOMPOSE_SYSTEM_PROMPT, query)
        if raw.strip().upper() == "NONE":
            logger.info("query_expansion.decompose", original=query, subquestion_count=0)
            return [query]

        seen: set[str] = {query.strip().lower()}
        result: list[str] = [query]
        for line in raw.splitlines():
            sub = line.strip().lstrip("0123456789.-) ").strip()
            key = sub.lower()
            if not sub or key in seen:
                continue
            seen.add(key)
            result.append(sub)
            if len(result) - 1 >= max_subquestions:
                break

        logger.info(
            "query_expansion.decompose", original=query, subquestion_count=len(result) - 1
        )
        return result
