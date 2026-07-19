from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from rag_ingestion.object_storage import ObjectStore


def _store() -> ObjectStore:
    with patch("rag_ingestion.object_storage.boto3.client") as mock_client_factory:
        store = ObjectStore(
            endpoint_url="http://minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin_change_me",
            bucket="rag-documents",
        )
        store._client = mock_client_factory.return_value
    return store


@pytest.mark.asyncio
class TestPut:
    async def test_uploads_with_the_given_key_and_bucket(self) -> None:
        store = _store()
        uri = await store.put("doc-1.pdf", b"pdf bytes", content_type="application/pdf")

        store._client.put_object.assert_called_once_with(
            Bucket="rag-documents",
            Key="doc-1.pdf",
            Body=b"pdf bytes",
            ContentType="application/pdf",
        )
        assert uri == "s3://rag-documents/doc-1.pdf"

    async def test_defaults_content_type_when_not_given(self) -> None:
        store = _store()
        await store.put("doc-1.bin", b"data")

        _, kwargs = store._client.put_object.call_args
        assert kwargs["ContentType"] == "application/octet-stream"


@pytest.mark.asyncio
class TestGet:
    async def test_returns_the_object_bytes(self) -> None:
        store = _store()
        body = MagicMock()
        body.read.return_value = b"pdf bytes"
        store._client.get_object.return_value = {"Body": body}

        result = await store.get("doc-1.pdf")

        store._client.get_object.assert_called_once_with(Bucket="rag-documents", Key="doc-1.pdf")
        assert result == b"pdf bytes"

    async def test_reads_from_a_real_streaming_body_shape(self) -> None:
        """Sanity check against botocore's actual StreamingBody interface
        (a file-like object with .read()), not just a MagicMock stand-in."""
        store = _store()
        store._client.get_object.return_value = {"Body": BytesIO(b"real bytes")}

        result = await store.get("doc-1.pdf")

        assert result == b"real bytes"
