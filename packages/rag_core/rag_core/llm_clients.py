"""Shared OpenAI-compatible client construction for Groq.

Groq speaks the OpenAI chat completions API, so a single client class
(`openai.AsyncOpenAI`) with a custom `base_url` serves every LLM call site in
the stack — generation, vision, and utility-tier calls all route through
Groq (ADR-012). Centralizing construction here means every service builds
this client identically instead of each re-deriving the base_url/header
setup.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from rag_core.config import BaseServiceSettings


def build_groq_client(settings: BaseServiceSettings) -> AsyncOpenAI:
    """Client for every LLM call site: generation (final-answer synthesis),
    vision (table/figure description), and utility-tier calls (query
    expansion, HyDE, GraphRAG triple extraction, RAG Triad judging)."""
    return AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )
