#!/usr/bin/env python
"""Ragas eval run (ADR-017) — a second, independently-implemented quality
signal alongside the hand-rolled RAG Triad gate (`run_eval_gate.py`, ADR-009).

Deliberately standalone: this script is invoked via a *separate* Python
interpreter (see services/eval/ragas-requirements.txt and the Dockerfile's
second install stage) because Ragas requires openai>=2, which conflicts
with this project's openai<2 pin everywhere else. It cannot import
rag_core or rag_eval — everything it needs (dataset loading, HTTP calls to
retrieval/generation, env-var config) is reimplemented minimally below
rather than shared, by design.

Informational only: this script's exit code does not gate CI (see
.github/workflows/ci.yml) — only run_eval_gate.py's does, per ADR-017.

Example:
    /opt/ragas-venv/bin/python scripts/run_ragas_eval.py \\
        --dataset eval_data/synthetic_eval_set.json \\
        --output results/ragas_result.json \\
        --retrieval-url http://retrieval:8000 \\
        --generation-url http://generation:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import instructor
from openai import AsyncOpenAI
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import InstructorLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)


@dataclass(frozen=True)
class EvalDatasetItem:
    """Mirrors rag_eval.schemas.SyntheticEvalItem's shape without importing
    it — the two must stay structurally compatible since they read the same
    dataset files, but this script has no dependency on rag_eval itself."""

    question: str
    reference_context: str
    reference_answer: str
    source_document_id: str


def load_dataset(path: Path) -> list[EvalDatasetItem]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Dataset file must contain a non-empty JSON array, got: {type(raw)}")
    return [
        EvalDatasetItem(
            question=item["question"],
            reference_context=item["reference_context"],
            reference_answer=item["reference_answer"],
            source_document_id=item["source_document_id"],
        )
        for item in raw
    ]


class _SentenceTransformerRagasEmbedding(BaseRagasEmbedding):
    """Bridges the self-hosted BAAI/bge-small-en-v1.5 model (same one used
    elsewhere in this stack) into Ragas's embeddings interface, so
    AnswerRelevancy's semantic-similarity computation stays on free,
    self-hosted infrastructure rather than an external embeddings API."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()  # type: ignore[no-any-return]

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts)


class _MinimalPipelineClient:
    """Reimplements just enough of rag_eval.pipeline_client.PipelineClient
    to call the live /retrieve and /generate endpoints, without importing
    rag_core.schemas (this script's venv has no rag_core installed)."""

    def __init__(self, *, retrieval_url: str, generation_url: str, timeout: float) -> None:
        self._retrieval_url = retrieval_url.rstrip("/") + "/retrieve"
        self._generation_url = generation_url.rstrip("/") + "/generate"
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def retrieve_and_generate(
        self, question: str, *, tenant_id: str, source_domains: list[str] | None = None
    ) -> tuple[list[str], str]:
        body: dict[str, Any] = {"query": question, "tenant_id": tenant_id}
        if source_domains:
            body["source_domains"] = source_domains

        retrieve_response = await self._client.post(self._retrieval_url, json=body)
        retrieve_response.raise_for_status()
        retrieved_chunks = [item["chunk"]["text"] for item in retrieve_response.json()]

        generate_response = await self._client.post(self._generation_url, json=body)
        generate_response.raise_for_status()
        answer = generate_response.json()["answer"]

        return retrieved_chunks, answer


def build_ragas_llm(groq_api_key: str, groq_base_url: str, model: str) -> InstructorLLM:
    from ragas.llms.base import InstructorModelArgs

    client = instructor.from_openai(AsyncOpenAI(api_key=groq_api_key, base_url=groq_base_url))
    # Faithfulness generates a full per-statement NLI breakdown (each
    # extracted claim plus its reasoning), not a single score — the
    # InstructorLLM default of 1024 truncates that mid-response on anything
    # but a short answer, producing an unparseable tool-call JSON payload.
    return InstructorLLM(
        client=client,
        model=model,
        provider="groq",
        model_args=InstructorModelArgs(max_tokens=4096),
    )


async def score_item(
    item: EvalDatasetItem,
    *,
    pipeline: _MinimalPipelineClient,
    faithfulness: Faithfulness,
    answer_relevancy: AnswerRelevancy,
    context_precision: ContextPrecisionWithoutReference,
    tenant_id: str,
    source_domains: list[str] | None,
) -> dict[str, Any]:
    retrieved_chunks, answer = await pipeline.retrieve_and_generate(
        item.question, tenant_id=tenant_id, source_domains=source_domains
    )

    faithfulness_result = await faithfulness.ascore(
        user_input=item.question, response=answer, retrieved_contexts=retrieved_chunks
    )
    relevancy_result = await answer_relevancy.ascore(user_input=item.question, response=answer)
    precision_result = await context_precision.ascore(
        user_input=item.question, response=answer, retrieved_contexts=retrieved_chunks
    )

    return {
        "question": item.question,
        "source_document_id": item.source_document_id,
        "answer": answer,
        "retrieved_context": retrieved_chunks,
        "ragas_faithfulness": faithfulness_result.value,
        "ragas_answer_relevancy": relevancy_result.value,
        "ragas_context_precision": precision_result.value,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset)

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    groq_base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    llm = build_ragas_llm(groq_api_key, groq_base_url, args.judge_model)
    embeddings = _SentenceTransformerRagasEmbedding(args.embedding_model)

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecisionWithoutReference(llm=llm)

    pipeline = _MinimalPipelineClient(
        retrieval_url=args.retrieval_url,
        generation_url=args.generation_url,
        timeout=args.timeout_seconds,
    )

    results: list[dict[str, Any]] = []
    failed: list[str] = []
    try:
        for item in dataset:
            try:
                result = await score_item(
                    item,
                    pipeline=pipeline,
                    faithfulness=faithfulness,
                    answer_relevancy=answer_relevancy,
                    context_precision=context_precision,
                    tenant_id=args.tenant_id,
                    source_domains=args.source_domains,
                )
            except Exception as exc:  # noqa: BLE001 - informational run; one bad item shouldn't abort the rest
                failed.append(f"{item.source_document_id!r} ({item.question!r}): {exc}")
                continue
            results.append(result)
    finally:
        await pipeline.aclose()

    def _mean(key: str) -> float | None:
        values = [r[key] for r in results]
        return sum(values) / len(values) if values else None

    return {
        "num_items_scored": len(results),
        "failed_items": failed,
        "mean_scores": {
            "ragas_faithfulness": _mean("ragas_faithfulness"),
            "ragas_answer_relevancy": _mean("ragas_answer_relevancy"),
            "ragas_context_precision": _mean("ragas_context_precision"),
        },
        "items": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Ragas eval (ADR-017, informational-only) against a synthetic dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ragas_result.json"))
    parser.add_argument("--retrieval-url", type=str, default="http://retrieval:8000")
    parser.add_argument("--generation-url", type=str, default="http://generation:8000")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--tenant-id", type=str, default="public")
    parser.add_argument(
        "--source-domains",
        type=str,
        nargs="*",
        default=None,
        help="Optional source_domains filter passed to /retrieve and /generate.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        result = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"ERROR: ragas eval run failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result["mean_scores"], indent=2))
    print(f"Items scored: {result['num_items_scored']}, failed: {len(result['failed_items'])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Results written to {args.output}")

    # Always exits 0 (barring a hard error) — informational, not gating (ADR-017).
    return 0


if __name__ == "__main__":
    sys.exit(main())
