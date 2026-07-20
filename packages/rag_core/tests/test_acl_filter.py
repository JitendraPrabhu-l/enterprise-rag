"""Tests for `build_acl_filter` (ADR-024 document-level ACL pre-filter) —
the dense-search half; the sparse (OpenSearch) half is covered in
test_sparse_search.py's ACL tests. Both must implement the identical
semantics: tenant AND (holds-an-allowed-principal OR no-ACL-field), with an
empty/missing principals list degrading to the "public" sentinel.
"""

from __future__ import annotations

from qdrant_client import models

from rag_core.vector_store import build_acl_filter


class TestBuildAclFilterStructure:
    def test_has_three_must_clauses_by_default(self) -> None:
        """Tenant scoping, ACL scoping, and currency (ADR-034) are all hard
        `must` (AND) conditions — a `should` for any would let out-of-scope
        or superseded documents rank in at lower score instead of being
        excluded outright (ADR-010/ADR-024/ADR-034). The currency clause is
        present by default because retrieval defaults to current content."""
        f = build_acl_filter(tenant_id="tenant-a", principals=["user:alice"])
        assert len(f.must) == 3

    def test_first_clause_is_tenant_scoping(self) -> None:
        f = build_acl_filter(tenant_id="tenant-a", principals=["user:alice"])
        tenant_condition = f.must[0]
        assert isinstance(tenant_condition, models.FieldCondition)
        assert tenant_condition.key == "metadata.tenant_id"
        assert tenant_condition.match.value == "tenant-a"

    def test_second_clause_matches_any_caller_principal(self) -> None:
        f = build_acl_filter(tenant_id="tenant-a", principals=["user:alice", "group:eng"])
        acl_group = f.must[1]
        principal_condition = acl_group.should[0]
        assert principal_condition.key == "metadata.allowed_principals"
        assert set(principal_condition.match.any) == {"user:alice", "group:eng"}

    def test_second_clause_also_admits_documents_with_no_acl_field(self) -> None:
        """Points indexed before ADR-024 carry no allowed_principals field at
        all — they must stay visible tenant-wide (their pre-ACL behavior),
        via an IsEmptyCondition OR'd alongside the principal match."""
        f = build_acl_filter(tenant_id="tenant-a", principals=["user:alice"])
        acl_group = f.must[1]
        assert isinstance(acl_group.should[1], models.IsEmptyCondition)
        assert acl_group.should[1].is_empty.key == "metadata.allowed_principals"


class TestBuildAclFilterFailsClosed:
    """The security-critical property: caller-supplied-nothing must never
    become caller-sees-everything."""

    def test_empty_principals_list_degrades_to_public_sentinel(self) -> None:
        f = build_acl_filter(tenant_id="tenant-a", principals=[])
        acl_group = f.must[1]
        assert acl_group.should[0].match.any == ["public"]

    def test_none_principals_degrades_to_public_sentinel(self) -> None:
        f = build_acl_filter(tenant_id="tenant-a", principals=None)
        acl_group = f.must[1]
        assert acl_group.should[0].match.any == ["public"]

    def test_whitespace_only_principal_is_dropped_not_matched_literally(self) -> None:
        """A blank string sneaking into the principals list must not become
        a literal match value (which could coincidentally match a
        mis-seeded allowed_principals entry) — it's filtered out, and if
        that empties the list, the public sentinel takes over."""
        f = build_acl_filter(tenant_id="tenant-a", principals=["  ", ""])
        acl_group = f.must[1]
        assert acl_group.should[0].match.any == ["public"]

    def test_mixed_blank_and_real_principals_keeps_only_real_ones(self) -> None:
        f = build_acl_filter(tenant_id="tenant-a", principals=["user:alice", "  ", ""])
        acl_group = f.must[1]
        assert acl_group.should[0].match.any == ["user:alice"]


class TestBuildAclFilterCurrency:
    """ADR-034: currency is a hard pre-filter by default, opt-out only."""

    def test_currency_clause_admits_current_or_field_absent(self) -> None:
        """The third must-clause keeps a doc that is explicitly is_current=true
        OR predates the field (IsEmpty) — mirroring the ACL backward-compat
        branch, so only an actively-marked-stale doc is ever filtered out."""
        f = build_acl_filter(tenant_id="tenant-a", principals=["user:alice"])
        currency_group = f.must[2]
        current_condition = currency_group.should[0]
        assert current_condition.key == "metadata.is_current"
        assert current_condition.match.value is True
        assert isinstance(currency_group.should[1], models.IsEmptyCondition)
        assert currency_group.should[1].is_empty.key == "metadata.is_current"

    def test_include_superseded_drops_the_currency_clause(self) -> None:
        """A caller who opts into historical content gets only the tenant +
        ACL clauses back — the currency filter is gone, so superseded
        versions become retrievable again."""
        f = build_acl_filter(
            tenant_id="tenant-a", principals=["user:alice"], include_superseded=True
        )
        assert len(f.must) == 2
        # Neither remaining clause references is_current.
        assert all(
            getattr(c, "key", None) != "metadata.is_current"
            for c in f.must
            if isinstance(c, models.FieldCondition)
        )
