from __future__ import annotations

import pytest

from rag_core.schemas import AccessRole, DocumentMetadata, SourceType


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        source_type=SourceType.PDF,
        source_domain="demo-corpus",
        tenant_id="tenant-a",
        access_role=AccessRole.PUBLIC,
        last_updated_epoch=1_700_000_000,
    )
