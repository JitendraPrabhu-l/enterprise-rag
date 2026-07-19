"""GraphRAG secondary index (ADR-006, opt-in per source_domain).

Extracts (subject, relation, object) triples from parent-context passages
using the utility model, then upserts them into Neo4j as an idempotent
MERGE keyed on (entity name, document_id) for nodes and on the relation
type for edges. This is a separate pipeline stage — `pipeline.py` only
invokes it when `graph_enabled` is true, and a failure here must never fail
the vector-index write it runs alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
from openai import APIError
from rag_core.config import BaseServiceSettings
from rag_core.llm_clients import build_groq_client
from rag_core.schemas import ParentContext
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_EXTRACTION_PROMPT = """Extract factual (subject, relation, object) triples from the \
passage below, suitable for a knowledge graph. Rules:
- Only extract triples explicitly supported by the text; do not infer or guess.
- Use short canonical entity names (e.g. "Acme Corp", not "the company").
- Relations should be short verb phrases in snake_case (e.g. "acquired", "ceo_of").
- Respond with ONLY a JSON array of objects with keys "subject", "relation", "object". \
If there are no clear triples, respond with [].

Passage:
{passage}
"""


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str


class TripleExtractor:
    """Wraps a single utility-model call (routed through Groq) to pull triples
    out of a passage."""

    def __init__(self, settings: BaseServiceSettings, *, max_tokens: int = 1024) -> None:
        self._client = build_groq_client(settings)
        self._model = settings.utility_model
        self._max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type(APIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
    )
    async def extract(self, passage: str) -> list[Triple]:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(passage=passage)}],
        )

        raw = (response.choices[0].message.content or "").strip()
        return _parse_triples(raw)


def _parse_triples(raw: str) -> list[Triple]:
    # The model sometimes wraps JSON in a markdown fence despite instructions;
    # strip it defensively rather than failing the whole extraction on it.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
        cleaned = cleaned.strip()

    if not cleaned:
        return []

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("triple_extraction_parse_failed", raw=raw[:200])
        return []

    if not isinstance(parsed, list):
        return []

    triples: list[Triple] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        relation = item.get("relation")
        obj = item.get("object")
        if (
            isinstance(subject, str)
            and isinstance(relation, str)
            and isinstance(obj, str)
            and subject.strip()
            and relation.strip()
            and obj.strip()
        ):
            triples.append(
                Triple(subject=subject.strip(), relation=relation.strip(), object=obj.strip())
            )
    return triples


_MERGE_TRIPLE_CYPHER = """
MERGE (s:Entity {name: $subject, document_id: $document_id})
MERGE (o:Entity {name: $object, document_id: $document_id})
MERGE (s)-[r:RELATES {type: $relation}]->(o)
SET r.document_id = $document_id, r.parent_id = $parent_id
"""


class GraphStore:
    """Async Neo4j driver wrapper: owns connection lifecycle and does the
    idempotent MERGE upsert. One driver instance should be shared across a
    process's requests (driver pooling), not opened per-call.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self._driver.close()

    @retry(
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
    )
    async def upsert_triples(self, document_id: str, parent_id: str, triples: list[Triple]) -> None:
        if not triples:
            return
        async with self._driver.session() as session:
            for triple in triples:
                await session.run(
                    _MERGE_TRIPLE_CYPHER,
                    subject=triple.subject,
                    object=triple.object,
                    relation=triple.relation,
                    document_id=document_id,
                    parent_id=parent_id,
                )


async def extract_and_store_graph(
    parents: list[ParentContext],
    *,
    extractor: TripleExtractor,
    graph_store: GraphStore,
) -> int:
    """Runs triple extraction (native async LLM call) for each parent passage
    and upserts results into Neo4j. Returns the total triple count written,
    for logging/metrics.

    A single parent's extraction failure is logged and skipped rather than
    aborting the whole document — GraphRAG is a best-effort secondary index,
    per ADR-006, and must not block the primary vector-index write.
    """
    total = 0
    for parent in parents:
        try:
            triples = await extractor.extract(parent.text)
        except APIError:
            logger.exception("triple_extraction_failed", parent_id=parent.parent_id)
            continue

        if not triples:
            continue

        try:
            await graph_store.upsert_triples(parent.document_id, parent.parent_id, triples)
        except (Neo4jError, ServiceUnavailable, SessionExpired):
            logger.exception("graph_upsert_failed", parent_id=parent.parent_id)
            continue

        total += len(triples)

    return total
