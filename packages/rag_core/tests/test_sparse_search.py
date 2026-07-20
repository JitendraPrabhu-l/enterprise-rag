"""Tests for the sparse-search module (ADR-004 hybrid half #2, ADR-020).

`SparseSearchClient` tests are ADR-010 security-critical: the OpenSearch
client itself is mocked and the assertions are on the *query body* passed to
`client.search`, specifically that tenant_id is present as a hard
`bool.filter` clause (not merely a `should` clause, which would let results
leak across tenants since `should` doesn't require a match).

`SparseIndexer` tests are ADR-020 regression coverage: the ingestion write
path must produce documents matching `INDEX_MAPPING` exactly, keyed by
chunk_id (idempotent re-ingest), and must fail loudly — never silently — when
the bulk write fails, because a dense-only document makes hybrid retrieval
inconsistent.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_core.schemas import AccessRole, ChunkRecord, DocumentMetadata, SourceType
from rag_core.sparse_search import (
    INDEX_MAPPING,
    SparseIndexer,
    SparseSearchClient,
    chunk_to_sparse_doc,
    index_name,
)


def _make_client() -> SparseSearchClient:
    client = SparseSearchClient("http://opensearch.invalid:9200", index_prefix="rag")
    return client


def _hit(doc_id: str, score: float, source: dict[str, Any]) -> dict[str, Any]:
    return {"_id": doc_id, "_score": score, "_source": source}


def _find_acl_clause(search_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pick the ACL bool-clause (references allowed_principals) out of the
    filter array. Since ADR-034 the filter also contains a currency bool
    clause, so selecting by content — not position — keeps the ACL tests
    robust to the currency clause's presence."""
    filter_clauses = search_kwargs["body"]["query"]["bool"]["filter"]
    for clause in filter_clauses:
        if "bool" in clause and any(
            "allowed_principals" in str(s) for s in clause["bool"].get("should", [])
        ):
            return clause
    raise AssertionError("no ACL clause found in filter")


@pytest.mark.asyncio
async def test_search_includes_tenant_id_as_hard_filter_clause() -> None:
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    client._client = mock_os_client  # type: ignore[attr-defined]

    await client.search(
        query="quarterly revenue",
        source_domains=["sec-filings"],
        tenant_id="tenant-a",
        top_k=10,
    )

    assert mock_os_client.search.await_count == 1
    _, kwargs = mock_os_client.search.await_args
    body = kwargs["body"]

    filter_clauses = body["query"]["bool"]["filter"]
    assert {"term": {"tenant_id": "tenant-a"}} in filter_clauses


@pytest.mark.asyncio
async def test_tenant_filter_is_in_filter_context_not_should() -> None:
    """The tenant scoping must be mandatory (filter), not optional (should).

    A `should` clause without `minimum_should_match` does not require a
    match, which would mean a query could return other tenants' documents
    whenever they happen to score well lexically. This test fails if a
    future refactor "optimizes" tenant scoping into a should-clause.
    """
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    client._client = mock_os_client  # type: ignore[attr-defined]

    await client.search(
        query="anything",
        source_domains=["domain-x"],
        tenant_id="tenant-b",
        top_k=5,
    )

    _, kwargs = mock_os_client.search.await_args
    body = kwargs["body"]
    bool_query = body["query"]["bool"]

    assert "filter" in bool_query, "tenant_id must be enforced via bool.filter"
    tenant_terms_in_filter = [
        clause for clause in bool_query["filter"] if clause == {"term": {"tenant_id": "tenant-b"}}
    ]
    assert len(tenant_terms_in_filter) == 1

    should_clauses = bool_query.get("should", [])
    assert not any("tenant_id" in str(clause) for clause in should_clauses), (
        "tenant_id must not appear in a should-clause — that would make tenancy optional"
    )


@pytest.mark.asyncio
class TestSparseSearchAclFilter:
    """ADR-024: principal ACLs must be enforced identically to tenancy —
    a hard bool.filter clause, never a should. See test_acl_filter.py for
    the equivalent dense-search (Qdrant) coverage; both stores must agree."""

    async def test_principals_appear_as_hard_filter_clause(self) -> None:
        client = _make_client()
        mock_os_client = AsyncMock()
        mock_os_client.search.return_value = {"hits": {"hits": []}}
        client._client = mock_os_client  # type: ignore[attr-defined]

        await client.search(
            query="revenue",
            source_domains=["sec-filings"],
            tenant_id="tenant-a",
            top_k=10,
            principals=["user:alice", "group:eng"],
        )

        _, kwargs = mock_os_client.search.await_args
        filter_clauses = kwargs["body"]["query"]["bool"]["filter"]
        # Two bool clauses now: ADR-024 ACL and ADR-034 currency. Select the
        # ACL one specifically (it references allowed_principals) rather than
        # asserting a single bool clause, so the currency clause's presence
        # doesn't make this test brittle.
        acl_clauses = [
            c
            for c in filter_clauses
            if "bool" in c
            and any("allowed_principals" in str(s) for s in c["bool"].get("should", []))
        ]
        assert len(acl_clauses) == 1
        should = acl_clauses[0]["bool"]["should"]
        assert {"terms": {"allowed_principals": ["user:alice", "group:eng"]}} in should

    async def test_acl_should_also_admits_documents_with_no_acl_field(self) -> None:
        """Documents indexed before ADR-024 have no allowed_principals field
        — they must stay visible tenant-wide, via a must_not-exists should
        branch alongside the principal-terms match."""
        client = _make_client()
        mock_os_client = AsyncMock()
        mock_os_client.search.return_value = {"hits": {"hits": []}}
        client._client = mock_os_client  # type: ignore[attr-defined]

        await client.search(
            query="revenue", source_domains=["x"], tenant_id="t", top_k=5,
            principals=["user:alice"],
        )

        _, kwargs = mock_os_client.search.await_args
        acl_clause = [
            c
            for c in kwargs["body"]["query"]["bool"]["filter"]
            if "bool" in c
            and any("allowed_principals" in str(s) for s in c["bool"].get("should", []))
        ][0]
        assert {"bool": {"must_not": [{"exists": {"field": "allowed_principals"}}]}} in (
            acl_clause["bool"]["should"]
        )

    async def test_empty_principals_degrades_to_public_sentinel(self) -> None:
        """Fail-closed: no caller principals must never mean 'see everything'
        — it must degrade to exactly the 'public' sentinel, same as the
        dense-search side (build_acl_filter)."""
        client = _make_client()
        mock_os_client = AsyncMock()
        mock_os_client.search.return_value = {"hits": {"hits": []}}
        client._client = mock_os_client  # type: ignore[attr-defined]

        await client.search(
            query="revenue", source_domains=["x"], tenant_id="t", top_k=5, principals=[]
        )

        _, kwargs = mock_os_client.search.await_args
        acl_clause = _find_acl_clause(kwargs)
        assert {"terms": {"allowed_principals": ["public"]}} in acl_clause["bool"]["should"]

    async def test_omitted_principals_defaults_to_public_sentinel(self) -> None:
        """Backward compatibility: an existing caller not passing `principals`
        at all (pre-ADR-024 call site) gets exactly pre-ACL public-only
        behavior, not an error and not unrestricted access."""
        client = _make_client()
        mock_os_client = AsyncMock()
        mock_os_client.search.return_value = {"hits": {"hits": []}}
        client._client = mock_os_client  # type: ignore[attr-defined]

        await client.search(query="revenue", source_domains=["x"], tenant_id="t", top_k=5)

        _, kwargs = mock_os_client.search.await_args
        acl_clause = _find_acl_clause(kwargs)
        assert {"terms": {"allowed_principals": ["public"]}} in acl_clause["bool"]["should"]


@pytest.mark.asyncio
class TestSparseCurrencyFilter:
    """ADR-034: currency is a hard bool.filter clause by default, opt-out only —
    mirroring the dense side (build_acl_filter)."""

    @staticmethod
    def _find_currency_clause(search_kwargs: dict[str, Any]) -> dict[str, Any] | None:
        for clause in search_kwargs["body"]["query"]["bool"]["filter"]:
            if "bool" in clause and any(
                "is_current" in str(s) for s in clause["bool"].get("should", [])
            ):
                return clause
        return None

    async def test_currency_clause_present_by_default(self) -> None:
        client = _make_client()
        mock_os_client = AsyncMock()
        mock_os_client.search.return_value = {"hits": {"hits": []}}
        client._client = mock_os_client  # type: ignore[attr-defined]

        await client.search(query="q", source_domains=["x"], tenant_id="t", top_k=5)

        _, kwargs = mock_os_client.search.await_args
        currency = self._find_currency_clause(kwargs)
        assert currency is not None
        should = currency["bool"]["should"]
        assert {"term": {"is_current": True}} in should
        assert {"bool": {"must_not": [{"exists": {"field": "is_current"}}]}} in should

    async def test_include_superseded_removes_currency_clause(self) -> None:
        client = _make_client()
        mock_os_client = AsyncMock()
        mock_os_client.search.return_value = {"hits": {"hits": []}}
        client._client = mock_os_client  # type: ignore[attr-defined]

        await client.search(
            query="q", source_domains=["x"], tenant_id="t", top_k=5, include_superseded=True
        )

        _, kwargs = mock_os_client.search.await_args
        assert self._find_currency_clause(kwargs) is None


@pytest.mark.asyncio
async def test_different_tenants_produce_different_filter_values() -> None:
    """A query for tenant A must carry tenant A's filter, never tenant B's."""
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    client._client = mock_os_client  # type: ignore[attr-defined]

    await client.search(query="q", source_domains=["d"], tenant_id="tenant-a", top_k=5)
    _, kwargs_a = mock_os_client.search.await_args
    filter_a = kwargs_a["body"]["query"]["bool"]["filter"]

    await client.search(query="q", source_domains=["d"], tenant_id="tenant-b", top_k=5)
    _, kwargs_b = mock_os_client.search.await_args
    filter_b = kwargs_b["body"]["query"]["bool"]["filter"]

    assert {"term": {"tenant_id": "tenant-a"}} in filter_a
    assert {"term": {"tenant_id": "tenant-b"}} not in filter_a

    assert {"term": {"tenant_id": "tenant-b"}} in filter_b
    assert {"term": {"tenant_id": "tenant-a"}} not in filter_b


@pytest.mark.asyncio
async def test_search_queries_each_source_domain_index() -> None:
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.search.return_value = {"hits": {"hits": []}}
    client._client = mock_os_client  # type: ignore[attr-defined]

    await client.search(
        query="q",
        source_domains=["domain-a", "domain-b"],
        tenant_id="tenant-a",
        top_k=5,
    )

    called_indices = [call.kwargs["index"] for call in mock_os_client.search.await_args_list]
    assert called_indices == [index_name("rag", "domain-a"), index_name("rag", "domain-b")]


@pytest.mark.asyncio
async def test_search_returns_ranked_results_truncated_to_top_k() -> None:
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.search.return_value = {
        "hits": {
            "hits": [
                _hit("c1", 5.0, {"text": "one"}),
                _hit("c2", 9.0, {"text": "two"}),
                _hit("c3", 1.0, {"text": "three"}),
            ]
        }
    }
    client._client = mock_os_client  # type: ignore[attr-defined]

    results = await client.search(
        query="q", source_domains=["domain-a"], tenant_id="tenant-a", top_k=2
    )

    assert [r["id"] for r in results] == ["c2", "c1"]
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_skips_missing_index_without_error() -> None:
    from opensearchpy.exceptions import NotFoundError

    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})
    client._client = mock_os_client  # type: ignore[attr-defined]

    results = await client.search(
        query="q", source_domains=["never-ingested"], tenant_id="tenant-a", top_k=5
    )

    assert results == []


@pytest.mark.asyncio
async def test_ensure_index_creates_when_absent() -> None:
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.indices.exists.return_value = False
    client._client = mock_os_client  # type: ignore[attr-defined]

    await client.ensure_index("new-domain")

    mock_os_client.indices.create.assert_awaited_once()
    _, kwargs = mock_os_client.indices.create.await_args
    assert kwargs["index"] == index_name("rag", "new-domain")
    assert "mappings" in kwargs["body"]


@pytest.mark.asyncio
async def test_ensure_index_noop_when_present() -> None:
    client = _make_client()
    mock_os_client = AsyncMock()
    mock_os_client.indices.exists.return_value = True
    client._client = mock_os_client  # type: ignore[attr-defined]

    await client.ensure_index("existing-domain")

    mock_os_client.indices.create.assert_not_awaited()


# --------------------------------------------------------------------------
# SparseIndexer (ADR-020: the ingestion write path)
# --------------------------------------------------------------------------


def _chunk(chunk_id: str = "c-1", text: str = "quarterly revenue grew") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        parent_id="doc-1:p0",
        document_id="doc-1",
        text=text,
        token_count=4,
        metadata=DocumentMetadata(
            document_id="doc-1",
            source_type=SourceType.PDF,
            source_domain="sec-filings",
            tenant_id="tenant-a",
            access_role=AccessRole.INTERNAL,
            last_updated_epoch=1_700_000_000,
        ),
    )


def _patch_opensearch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AsyncMock, AsyncMock, MagicMock]:
    """Replace the module's AsyncOpenSearch constructor and async_bulk helper;
    returns (mock_client, mock_bulk, mock_client_cls)."""
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = True
    mock_client_cls = MagicMock(return_value=mock_client)
    mock_bulk = AsyncMock(return_value=(2, []))
    monkeypatch.setattr("rag_core.sparse_search.AsyncOpenSearch", mock_client_cls)
    monkeypatch.setattr("rag_core.sparse_search.async_bulk", mock_bulk)
    return mock_client, mock_bulk, mock_client_cls


def test_chunk_to_sparse_doc_projects_exactly_the_mapped_fields() -> None:
    """The doc body must match INDEX_MAPPING field-for-field: a field the
    mapping doesn't know is dead weight; a mapped field the doc omits makes
    BM25/tenancy silently blind to it."""
    doc = chunk_to_sparse_doc(_chunk())

    assert set(doc) == set(INDEX_MAPPING["mappings"]["properties"])
    assert doc["chunk_id"] == "c-1"
    assert doc["parent_id"] == "doc-1:p0"
    assert doc["document_id"] == "doc-1"
    assert doc["text"] == "quarterly revenue grew"
    assert doc["tenant_id"] == "tenant-a"
    assert doc["access_role"] == "internal"
    assert doc["source_domain"] == "sec-filings"


@pytest.mark.asyncio
async def test_index_chunks_bulk_writes_with_chunk_id_as_doc_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chunk_id as the OpenSearch _id is what makes re-ingestion idempotent
    (overwrite, not duplicate) — mirroring VectorStore.upsert_chunks."""
    mock_client, mock_bulk, _ = _patch_opensearch(monkeypatch)
    indexer = SparseIndexer("http://opensearch.invalid:9200", index_prefix="rag")

    count = await indexer.index_chunks([_chunk("c-1"), _chunk("c-2")], source_domain="sec-filings")

    assert count == 2
    (_, actions), _kwargs = mock_bulk.await_args
    assert [a["_id"] for a in actions] == ["c-1", "c-2"]
    assert all(a["_index"] == index_name("rag", "sec-filings") for a in actions)
    assert all(set(a["_source"]) == set(INDEX_MAPPING["mappings"]["properties"]) for a in actions)
    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_index_chunks_creates_missing_index_before_bulk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client, _, _ = _patch_opensearch(monkeypatch)
    mock_client.indices.exists.return_value = False
    indexer = SparseIndexer("http://opensearch.invalid:9200", index_prefix="rag")

    await indexer.index_chunks([_chunk()], source_domain="new-domain")

    mock_client.indices.create.assert_awaited_once()
    _, kwargs = mock_client.indices.create.await_args
    assert kwargs["index"] == index_name("rag", "new-domain")


@pytest.mark.asyncio
async def test_index_chunks_empty_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _, mock_bulk, mock_client_cls = _patch_opensearch(monkeypatch)
    indexer = SparseIndexer("http://opensearch.invalid:9200")

    count = await indexer.index_chunks([], source_domain="sec-filings")

    assert count == 0
    mock_client_cls.assert_not_called()
    mock_bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_chunks_propagates_bulk_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed sparse write must fail the ingestion job — silent dense-only
    ingestion is the exact bug ADR-020 exists to prevent."""
    mock_client, mock_bulk, _ = _patch_opensearch(monkeypatch)
    mock_bulk.side_effect = RuntimeError("bulk index failed")
    indexer = SparseIndexer("http://opensearch.invalid:9200")

    with pytest.raises(RuntimeError, match="bulk index failed"):
        await indexer.index_chunks([_chunk()], source_domain="sec-filings")

    mock_client.close.assert_awaited_once()
