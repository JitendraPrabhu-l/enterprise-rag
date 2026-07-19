"""GraphRAG as opt-in secondary retrieval (ADR-006).

Two pieces:
- `QueryClassifier`: a cheap heuristic (with an optional Claude fallback for
  ambiguous cases) that decides whether a query is a "local" factual lookup
  (vector/BM25 search is sufficient) or a "global"/thematic multi-hop query
  (graph traversal adds value). This only *recommends* graph use — the
  pipeline still respects `QueryRequest.use_graph` as an explicit override.
- `GraphQueryClient`: an async Neo4j driver wrapper that extracts candidate
  entity names from the query (simple capitalized-phrase heuristic — no
  heavyweight NER dependency), fuzzy-matches them against `:Entity` nodes,
  and traverses 1-2 hops to pull back related entities/relationships as
  extra textual context.

Driver/session lifecycle: a single `AsyncDriver` is created once (owned by
the pipeline/app lifespan) and reused; every query opens its session via
`async with self._driver.session(...)` so sessions are always closed
promptly and no connections leak across requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog
from neo4j import AsyncDriver
from neo4j.exceptions import Neo4jError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_RETRYABLE = (Neo4jError, ConnectionError, TimeoutError)

# Heuristic global/thematic signal words: comparisons, summaries, trends,
# and multi-hop relationship language tend to benefit from graph traversal
# far more than lookups of a single fact.
_GLOBAL_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"compare|comparison|relationship|relate[sd]?|connect(ed|ion)?|"
    r"overview|summari[sz]e|trend|across|between|impact|influence|"
    r"how (does|do|did)|why (does|do|did)|all (of )?the|"
    r"network|ecosystem|landscape"
    r")\b",
    re.IGNORECASE,
)

# Capitalized-phrase heuristic for candidate entity names: sequences of one
# or more Title-Case words (allows internal "of"/"and" for multi-word names
# like "Bank of America"), used as fuzzy-match seeds for the graph traversal.
_ENTITY_CANDIDATE_PATTERN = re.compile(
    r"\b[A-Z][\w&.-]*(?:\s+(?:[A-Z][\w&.-]*|of|and|the|for)){0,4}\b"
)
_STOPWORD_ENTITIES = {"the", "what", "who", "when", "where", "why", "how"}


@dataclass(frozen=True)
class GraphContext:
    """Textual context assembled from a graph traversal, ready to merge into the pipeline."""

    entity_name: str
    related_entity: str
    relationship_type: str
    hops: int


class QueryClassifier:
    """Heuristic local-vs-global query classifier (ADR-006)."""

    def is_global(self, query: str) -> bool:
        """Return True if `query` looks like a global/thematic multi-hop question.

        Purely heuristic and synchronous — no network call — so it's cheap
        enough to run on every request as a recommendation signal, distinct
        from the explicit `QueryRequest.use_graph` override.
        """
        if _GLOBAL_SIGNAL_PATTERN.search(query):
            return True
        # Multiple distinct capitalized entities in one query is itself a
        # weak multi-hop signal (e.g. "How did X affect Y?").
        candidates = extract_entity_candidates(query)
        return len(candidates) >= 2


def extract_entity_candidates(query: str) -> list[str]:
    """Extract candidate entity names from a query via a capitalized-phrase heuristic.

    Deliberately lightweight (no spaCy/NER model dependency) — this is a
    fuzzy seed list for graph traversal, not a precision entity linker. The
    first word of a sentence being capitalized is a known source of false
    positives; callers treat entity matching in Neo4j (case-insensitive
    CONTAINS) as the real precision filter.
    """
    matches = _ENTITY_CANDIDATE_PATTERN.findall(query)
    candidates: list[str] = []
    seen: set[str] = set()
    for match in matches:
        cleaned = match.strip()
        key = cleaned.lower()
        if not cleaned or key in _STOPWORD_ENTITIES or key in seen:
            continue
        seen.add(key)
        candidates.append(cleaned)
    return candidates


_TRAVERSAL_QUERY = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($entity_name)
MATCH path = (e)-[r*1..%d]-(related:Entity)
WHERE e.tenant_id = $tenant_id AND related.tenant_id = $tenant_id
RETURN e.name AS entity_name,
       related.name AS related_entity,
       [rel IN relationships(path) | type(rel)] AS rel_types,
       length(path) AS hops
LIMIT $limit
"""


class GraphQueryClient:
    """Async Neo4j wrapper for opt-in GraphRAG traversal (ADR-006)."""

    def __init__(self, driver: AsyncDriver, max_hops: int = 2, max_entities: int = 10) -> None:
        self._driver = driver
        self._max_hops = max_hops
        self._max_entities = max_entities

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    async def query_related_entities(
        self,
        query_text: str,
        tenant_id: str,
        limit_per_entity: int = 5,
    ) -> list[GraphContext]:
        """Traverse 1-2 hops from entities named in `query_text`, scoped to `tenant_id`.

        Opens and closes its own session via an async context manager so no
        connection is held across requests. Tenancy is enforced in the
        Cypher `WHERE` clause on both endpoints of the path, matching the
        hard pre-filter posture used elsewhere (ADR-010).
        """
        entity_candidates = extract_entity_candidates(query_text)[: self._max_entities]
        if not entity_candidates:
            return []

        cypher = _TRAVERSAL_QUERY % self._max_hops
        contexts: list[GraphContext] = []

        async with self._driver.session() as session:
            for entity_name in entity_candidates:
                result = await session.run(
                    cypher,
                    entity_name=entity_name,
                    tenant_id=tenant_id,
                    limit=limit_per_entity,
                )
                records = [record async for record in result]
                for record in records:
                    rel_types = record["rel_types"] or ["RELATED_TO"]
                    contexts.append(
                        GraphContext(
                            entity_name=record["entity_name"],
                            related_entity=record["related_entity"],
                            relationship_type=rel_types[0],
                            hops=record["hops"],
                        )
                    )

        logger.info(
            "graph_query.completed",
            entity_candidates=entity_candidates,
            context_count=len(contexts),
        )
        return contexts

    @staticmethod
    def format_context_text(contexts: list[GraphContext]) -> str:
        """Render graph traversal results as a flat text block for prompt/context merging."""
        if not contexts:
            return ""
        lines = [
            f"{ctx.entity_name} --[{ctx.relationship_type}]--> {ctx.related_entity} "
            f"({ctx.hops} hop{'s' if ctx.hops != 1 else ''})"
            for ctx in contexts
        ]
        return "\n".join(lines)
