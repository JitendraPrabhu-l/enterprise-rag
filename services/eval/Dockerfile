# syntax=docker/dockerfile:1
# Multi-stage build for the rag-eval service.
#
# Default CMD runs the uvicorn production-sampling API (POST /score, GET /healthz).
# The CI/CD gate script is meant to be run as a separate, short-lived invocation,
# e.g.:
#   docker run --rm rag-eval:latest python scripts/run_eval_gate.py \
#       --dataset /data/synthetic_eval_set.json --output /data/result.json
#
# ADR-017: run_ragas_eval.py (informational, non-gating) uses the SEPARATE
# /opt/ragas-venv interpreter, not the image's main python:
#   docker run --rm rag-eval:latest /opt/ragas-venv/bin/python scripts/run_ragas_eval.py \
#       --dataset /data/synthetic_eval_set.json --output /data/ragas_result.json

FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir hatchling

# Build context is expected to be the monorepo root so the local rag-core
# path dependency resolves; copy just what's needed for the build.
COPY packages/rag_core /build/packages/rag_core
COPY services/eval /build/services/eval

WORKDIR /build/services/eval
# Single pip invocation so rag-core (a local path dependency, not on PyPI) is
# resolved from ../../packages/rag_core instead of failing to find it on PyPI
# when rag-eval's own dependency resolution runs as a separate command.
RUN pip install --no-cache-dir --prefix=/install ../../packages/rag_core .

FROM python:3.11-slim AS ragas-builder
# ADR-017: Ragas requires openai>=2, which conflicts with this project's
# openai<2 pin (rag_core and every service depend on it) — built into its
# own venv here, never merged with the main /install prefix above, so the
# two dependency trees never have to be reconciled into one resolvable set.
WORKDIR /build
COPY services/eval/ragas-requirements.txt /build/ragas-requirements.txt
RUN python -m venv /opt/ragas-venv \
    && /opt/ragas-venv/bin/pip install --no-cache-dir -r /build/ragas-requirements.txt

FROM python:3.11-slim AS runtime

RUN groupadd --gid 1000 raguser \
    && useradd --uid 1000 --gid raguser --shell /bin/bash --create-home raguser

COPY --from=builder /install /usr/local
COPY --from=ragas-builder /opt/ragas-venv /opt/ragas-venv

WORKDIR /app
COPY --from=builder /build/services/eval/rag_eval /app/rag_eval
COPY --from=builder /build/services/eval/scripts /app/scripts

RUN chown -R raguser:raguser /app /opt/ragas-venv
USER raguser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3).read()" || exit 1

CMD ["python", "-m", "rag_eval.main"]
