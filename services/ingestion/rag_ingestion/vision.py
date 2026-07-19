"""Vision-model table/figure description (ADR-001) via an OpenAI-compatible
client routed through Groq (ADR-012).

Pages the heuristic classifier flags as table/figure-dense are rendered to an
image upstream (see `pipeline.py`, which owns the PDF->image rendering) and
handed here as raw bytes. We ask the vision model for a dense,
retrieval-friendly text description — this description, not the image, is
what gets embedded and indexed as a `ContentModality.TABLE`/`FIGURE` chunk.
"""

from __future__ import annotations

import base64
from typing import Protocol

import structlog
from openai import APIError
from rag_core.config import BaseServiceSettings
from rag_core.llm_clients import build_groq_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_SUPPORTED_MEDIA_TYPES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}

_DESCRIBE_PROMPT = (
    "You are assisting a retrieval-augmented generation pipeline. This image is a "
    "page from a document that likely contains a table or figure. Produce a dense, "
    "self-contained text description suitable for semantic search: transcribe "
    "table contents as structured text (row by row), describe figures/charts "
    "including axis labels, legends, and key values, and preserve any captions. "
    "Do not add commentary about the image itself — output only the description."
)


class VisionDescriber(Protocol):
    async def describe_page(self, image_bytes: bytes, *, page_number: int) -> str: ...


def _sniff_media_type(image_bytes: bytes) -> str:
    for magic, media_type in _SUPPORTED_MEDIA_TYPES.items():
        if image_bytes.startswith(magic):
            return media_type
    # WEBP: RIFF....WEBP
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Unrecognized image format; expected PNG, JPEG, GIF, or WEBP bytes")


class GroqVisionDescriber:
    """Wraps an OpenAI-compatible `chat.completions.create` call (routed through
    Groq) with an image_url content part."""

    def __init__(self, settings: BaseServiceSettings, *, max_tokens: int = 1024) -> None:
        self._client = build_groq_client(settings)
        self._model = settings.vision_model
        self._max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type(APIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
    )
    async def describe_page(self, image_bytes: bytes, *, page_number: int) -> str:
        media_type = _sniff_media_type(image_bytes)
        encoded = base64.standard_b64encode(image_bytes).decode("ascii")

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _DESCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )

        description = response.choices[0].message.content or ""
        if not description.strip():
            logger.warning("vision_describe_empty_response", page_number=page_number)
            return ""
        return description.strip()
