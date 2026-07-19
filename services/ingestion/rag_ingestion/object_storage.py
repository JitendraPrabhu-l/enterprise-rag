"""S3-compatible object storage for uploaded documents (ADR-014).

`boto3` has no native asyncio support (same situation as `sentence-transformers`
in `embeddings.py`), so every call here is offloaded via `asyncio.to_thread` —
this class's own methods are the async surface; nothing above it needs to know
`boto3` itself is synchronous.

Points at MinIO in every environment covered by this repo's docker-compose.yml,
but the only MinIO-specific things are the endpoint URL and credentials — the
same code talks to real AWS S3 unchanged if `minio_endpoint_url` becomes
`None` and real AWS credentials are supplied instead.
"""

from __future__ import annotations

import asyncio

import boto3
import structlog

logger = structlog.get_logger(__name__)


class ObjectStore:
    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
    ) -> None:
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    async def put(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Uploads `data` under `key`, returning the `s3://bucket/key` reference
        to persist as the document's durable `uri`."""
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        uri = f"s3://{self._bucket}/{key}"
        logger.info("object_storage.uploaded", key=key, size_bytes=len(data))
        return uri

    async def get(self, key: str) -> bytes:
        """Downloads the object at `key`. Used for batch/offline re-ingestion
        against a document already in the store, mirroring the `file_path`
        ingestion mode's role for local files."""
        response = await asyncio.to_thread(self._client.get_object, Bucket=self._bucket, Key=key)
        return response["Body"].read()  # type: ignore[no-any-return]
