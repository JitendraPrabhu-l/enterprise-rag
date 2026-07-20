"""The chunk record contract shared by ingestion, retrieval, generation, and eval.

Implements the parent-child split from ADR-002: `ChunkRecord` is the small,
embedded, vector-searchable unit; `ParentContext` is the larger passage sent
to the generator once a child chunk wins retrieval.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    IMAGE = "image"
    TEXT = "text"
    MARKDOWN = "markdown"


class ContentModality(str, Enum):
    """What kind of content produced this chunk's text.

    TABLE and FIGURE chunks carry a `source_ref` pointing at the original
    extracted asset in blob storage (ADR-001) — the text here is always the
    searchable *description*, never a re-encoded image.
    """

    PROSE = "prose"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"


class AccessRole(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    ADMIN = "admin"


class DocumentMetadata(BaseModel):
    """Programmatically injected header fields (ADR-001 metadata strategy).

    These are the hard pre-filter fields (ADR-010): tenancy/access scoping
    must be enforced as a database constraint on this metadata *before*
    vector distance is computed, never as a post-hoc filter.
    """

    document_id: str
    source_type: SourceType
    source_domain: str = Field(
        description="Logical collection/shard key, e.g. 'sec-filings', 'arxiv-cs'. "
        "Drives Qdrant collection routing per ADR-003."
    )
    tenant_id: str = "public"
    access_role: AccessRole = AccessRole.PUBLIC
    # Document-level ACL (ADR-024): principal identifiers (user/group ids from
    # the upstream IdP) allowed to retrieve this document's chunks, enforced
    # as a hard pre-filter in BOTH stores alongside tenant_id — the 2026
    # enterprise-RAG baseline ("permissions are a first-class data model,
    # enforced at the database retrieval layer, never post-hoc"). The
    # "public" sentinel keeps every pre-ACL document and caller working
    # unchanged.
    allowed_principals: list[str] = Field(default_factory=lambda: ["public"])
    title: str | None = None
    uri: str | None = None
    last_updated_epoch: int
    page_count: int | None = None
    # Freshness / versioning / supersession (ADR-034). The single most
    # common production-RAG failure is "similarity is not authority": a
    # stale-but-similar document out-ranks current guidance and answers
    # confidently from a superseded policy. These fields make currency a
    # first-class, filterable property so retrieval can default to current
    # content, and give every answer an auditable "which version produced
    # this" trail (the EU AI Act documentation requirement for high-risk
    # systems). All optional/defaulted so pre-ADR-034 points still index and
    # retrieve unchanged.
    is_current: bool = Field(
        default=True,
        description="Whether this document is the current, non-superseded version. "
        "Retrieval defaults to is_current=true (ADR-034) unless the caller opts "
        "into historical content via QueryRequest.include_superseded.",
    )
    document_version: int = Field(
        default=1, ge=1, description="Monotonic version of this logical document."
    )
    superseded_by_document_id: str | None = Field(
        default=None,
        description="If this version was replaced, the document_id that replaced it — "
        "the supersession trail retrieval and audit both read.",
    )
    content_hash: str | None = Field(
        default=None,
        description="Stable hash of the source content (ADR-036 dedup key + answer "
        "provenance stamp). None for pre-ADR-034 points.",
    )
    embedding_model_version: str | None = Field(
        default=None,
        description="Identifier of the embedding model that produced this chunk's "
        "vector — the re-embedding trigger (ADR-036) and half of the answer "
        "provenance stamp. None for pre-ADR-034 points.",
    )
    valid_from_epoch: int | None = Field(
        default=None, description="Unix epoch the document became effective, if known."
    )
    valid_until_epoch: int | None = Field(
        default=None,
        description="Unix epoch the document stops being effective, if known — "
        "an expiry a time-aware retrieval filter can enforce.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class ParentContext(BaseModel):
    """The ~1024-token passage handed to the generator (ADR-002)."""

    parent_id: str
    document_id: str
    text: str
    page_number: int | None = None
    modality: ContentModality = ContentModality.PROSE
    source_ref: str | None = Field(
        default=None,
        description="Blob storage pointer to the original table/figure asset, if any.",
    )


class ChunkRecord(BaseModel):
    """The small, embedded, vector-searchable child chunk (ADR-002)."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_id: str
    document_id: str
    text: str
    # Contextual retrieval (ADR-023, Anthropic technique): a 1-2 sentence
    # LLM-generated situating summary ("this chunk is from X's Q3 filing,
    # section on ..."), prepended to the text for BOTH embedding and BM25
    # indexing via searchable_text below. None = enrichment disabled or
    # failed for this chunk — indexing then uses the raw text, exactly the
    # pre-ADR-023 behavior. The generator always receives the raw parent
    # passage; the prefix exists only to make the chunk findable.
    context_prefix: str | None = None
    modality: ContentModality = ContentModality.PROSE
    token_count: int
    embedding: list[float] | None = None
    metadata: DocumentMetadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def searchable_text(self) -> str:
        """What the indexes see (ADR-023): context-situated when enrichment
        ran, raw otherwise. Every index write path must use this, never
        .text directly — one property, so the two indexes can't drift."""
        if self.context_prefix:
            return f"{self.context_prefix}\n{self.text}"
        return self.text

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text must not be empty")
        return v


class RetrievedChunk(BaseModel):
    """A chunk as it flows out of the retrieval pipeline, with score provenance."""

    chunk: ChunkRecord
    parent: ParentContext
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score is not None:
            return self.rrf_score
        return self.dense_score or 0.0


class QueryRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    query: str
    tenant_id: str = "public"
    # Caller's principal identifiers (ADR-024) — user id + group ids as
    # asserted by the API gateway/IdP upstream. A chunk is retrievable only
    # if its allowed_principals intersects this list. Defaults to the
    # "public" sentinel so unauthenticated/demo callers see exactly the
    # documents ingested without an ACL, and nothing else.
    principals: list[str] = Field(default_factory=lambda: ["public"])
    source_domains: list[str] | None = Field(
        default=None, description="Restrict to specific collections; None = search all."
    )
    top_k: int = Field(default=40, ge=1, le=200)
    top_n: int = Field(default=5, ge=1, le=20)
    use_graph: bool = False
    # ADR-034: retrieval defaults to current documents only. A caller who
    # genuinely wants historical/superseded content (audit, "what did the
    # policy say last year") opts in explicitly — the safe default is that a
    # superseded document never silently out-ranks the version that replaced
    # it.
    include_superseded: bool = False


class Citation(BaseModel):
    parent_id: str
    document_id: str
    title: str | None = None
    uri: str | None = None
    page_number: int | None = None


class AnswerProvenance(BaseModel):
    """The "which index version produced this answer" audit stamp (ADR-034).

    Populated by the generation pipeline from the chunks an answer actually
    cited — the embedding-model versions and content hashes behind those
    chunks. This is the EU AI Act documentation trail for high-risk systems:
    given a past answer, you can prove exactly which document versions and
    which embedding model stood behind it, and detect after the fact if the
    corpus or model has since moved on.
    """

    embedding_model_versions: list[str] = Field(default_factory=list)
    document_versions: dict[str, int] = Field(
        default_factory=dict, description="document_id -> document_version of cited docs."
    )
    content_hashes: list[str] = Field(default_factory=list)


class GenerationResponse(BaseModel):
    request_id: UUID
    answer: str
    citations: list[Citation]
    model: str
    used_graph: bool = False
    guardrail_flagged: bool = False
    # ADR-034 provenance stamp. Optional so cached/older responses and tests
    # that construct a response directly stay valid; the pipeline fills it in.
    provenance: AnswerProvenance | None = None


class AnswerFeedback(BaseModel):
    """User feedback on a generated answer (ADR-027).

    The production signal that turns real failures into eval cases: a
    thumbs-down with the original query and answer is exactly a candidate for
    the golden set. `rating` is the minimum; `query`/`answer`/`comment` are
    optional context a UI can attach for triage.
    """

    request_id: UUID
    rating: str = Field(description="'up' or 'down'")
    query: str | None = None
    answer: str | None = None
    comment: str | None = None

    @field_validator("rating")
    @classmethod
    def _valid_rating(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        return v
