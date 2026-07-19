#!/usr/bin/env python
"""CI/CD gate script implementing the RAG Triad quality gate (ADR-009).

Loads a synthetic eval dataset, runs each item through the live retrieval +
generation pipeline, scores every response on the RAG Triad (faithfulness,
answer relevance, context precision), aggregates mean scores per axis, and
compares against configurable thresholds.

Exit codes (load-bearing — this is what a CI pipeline greps for):
    0 - every axis met its threshold and every item scored successfully
    1 - at least one axis missed its threshold, an item failed to score, or
        the run could not complete (bad dataset, unreachable services, etc.)

Example:
    python scripts/run_eval_gate.py \\
        --dataset eval_data/synthetic_eval_set.json \\
        --output results/eval_gate_result.json \\
        --faithfulness-threshold 0.8 \\
        --answer-relevance-threshold 0.75 \\
        --context-precision-threshold 0.7 \\
        --retrieval-url http://retrieval:8000 \\
        --generation-url http://generation:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running this script directly (`python scripts/run_eval_gate.py`) without
# having installed the rag_eval package into the environment first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402
from rag_core.llm_clients import build_groq_client  # noqa: E402

from rag_eval.config import EvalSettings  # noqa: E402
from rag_eval.eval_runner import EvalGateResult, run_eval_gate  # noqa: E402
from rag_eval.pipeline_client import PipelineClient  # noqa: E402
from rag_eval.schemas import SyntheticEvalItem  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = EvalSettings()

    parser = argparse.ArgumentParser(
        description="Run the RAG Triad CI/CD quality gate against a synthetic eval dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a JSON file containing a list of SyntheticEvalItem records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_gate_result.json"),
        help="Path to write the JSON results file (default: ./eval_gate_result.json).",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=settings.faithfulness_threshold,
        help=(
            f"Minimum mean faithfulness score to pass (default: {settings.faithfulness_threshold})."
        ),
    )
    parser.add_argument(
        "--answer-relevance-threshold",
        type=float,
        default=settings.answer_relevance_threshold,
        help=(
            "Minimum mean answer-relevance score to pass "
            f"(default: {settings.answer_relevance_threshold})."
        ),
    )
    parser.add_argument(
        "--context-precision-threshold",
        type=float,
        default=settings.context_precision_threshold,
        help=(
            "Minimum mean context-precision score to pass "
            f"(default: {settings.context_precision_threshold})."
        ),
    )
    parser.add_argument(
        "--retrieval-url",
        type=str,
        default=settings.retrieval_service_url,
        help=(f"Base URL of the retrieval service (default: {settings.retrieval_service_url})."),
    )
    parser.add_argument(
        "--generation-url",
        type=str,
        default=settings.generation_service_url,
        help=f"Base URL of the generation service (default: {settings.generation_service_url}).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=settings.utility_model,
        help=f"Model used for RAG Triad judging (default: {settings.utility_model}).",
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        default="public",
        help="Tenant ID to use for QueryRequests sent to the pipeline (default: public).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.http_timeout_seconds,
        help=(
            "HTTP timeout for pipeline calls in seconds "
            f"(default: {settings.http_timeout_seconds})."
        ),
    )
    return parser.parse_args(argv)


def load_dataset(path: Path) -> list[SyntheticEvalItem]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Dataset file must contain a JSON array of items, got {type(raw).__name__}"
        )
    if not raw:
        raise ValueError("Dataset file contains an empty list — nothing to evaluate.")

    try:
        return [SyntheticEvalItem.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"Dataset file failed SyntheticEvalItem validation: {exc}") from exc


def render_summary_table(result: EvalGateResult) -> str:
    rows = [
        ("faithfulness", result.faithfulness),
        ("answer_relevance", result.answer_relevance),
        ("context_precision", result.context_precision),
    ]
    header = f"{'axis':<20}{'mean score':>12}{'threshold':>12}{'result':>10}"
    lines = [header, "-" * len(header)]
    for name, axis in rows:
        status = "PASS" if axis.passed else "FAIL"
        lines.append(f"{name:<20}{axis.mean_score:>12.3f}{axis.threshold:>12.3f}{status:>10}")

    lines.append("")
    lines.append(f"Items scored: {len(result.item_results)}")
    if result.failed_items:
        lines.append(f"Items FAILED to score ({len(result.failed_items)}):")
        for failure in result.failed_items:
            lines.append(f"  - {failure}")

    lines.append("")
    lines.append(f"GATE RESULT: {'PASS' if result.passed else 'FAIL'}")
    return "\n".join(lines)


def result_to_json(result: EvalGateResult, args: argparse.Namespace) -> dict[str, object]:
    return {
        "gate_passed": result.passed,
        "thresholds": {
            "faithfulness": args.faithfulness_threshold,
            "answer_relevance": args.answer_relevance_threshold,
            "context_precision": args.context_precision_threshold,
        },
        "aggregates": {
            "faithfulness": {
                "mean_score": result.faithfulness.mean_score,
                "threshold": result.faithfulness.threshold,
                "passed": result.faithfulness.passed,
            },
            "answer_relevance": {
                "mean_score": result.answer_relevance.mean_score,
                "threshold": result.answer_relevance.threshold,
                "passed": result.answer_relevance.passed,
            },
            "context_precision": {
                "mean_score": result.context_precision.mean_score,
                "threshold": result.context_precision.threshold,
                "passed": result.context_precision.passed,
            },
        },
        "num_items_scored": len(result.item_results),
        "failed_items": result.failed_items,
        "items": [
            {
                "question": item.question,
                "source_document_id": item.source_document_id,
                "answer": item.answer,
                "retrieved_context": item.retrieved_context,
                "faithfulness": item.triad.faithfulness.model_dump(),
                "answer_relevance": item.triad.answer_relevance.model_dump(),
                "context_precision": item.triad.context_precision.model_dump(),
            }
            for item in result.item_results
        ],
    }


async def _run(args: argparse.Namespace) -> EvalGateResult:
    settings = EvalSettings()
    dataset = load_dataset(args.dataset)

    async with (
        build_groq_client(settings) as judge_client,
        PipelineClient(
            retrieval_base_url=args.retrieval_url,
            generation_base_url=args.generation_url,
            timeout_seconds=args.timeout_seconds,
        ) as pipeline_client,
    ):
        return await run_eval_gate(
            dataset,
            pipeline_client=pipeline_client,
            judge_client=judge_client,
            judge_model=args.judge_model,
            faithfulness_threshold=args.faithfulness_threshold,
            answer_relevance_threshold=args.answer_relevance_threshold,
            context_precision_threshold=args.context_precision_threshold,
            tenant_id=args.tenant_id,
            judge_max_retries=settings.judge_max_retries,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        result = asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary; any failure -> exit 1
        print(f"ERROR: eval gate run failed: {exc}", file=sys.stderr)
        return 1

    print(render_summary_table(result))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result_to_json(result, args), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nResults written to {args.output}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
