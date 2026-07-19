# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip hatch

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

# sentence-transformers pulls in torch; installed CPU-only up front, then pinned
# via a constraints file for the main install below. `pip install --target` always
# re-resolves requirements against its configured index regardless of what's
# already on disk in the target dir, so without the pin the second command would
# silently re-pull torch's default Linux (CUDA/triton-bundled, several-GB) wheel.
RUN pip install --no-cache-dir --target=/deps \
    --index-url https://download.pytorch.org/whl/cpu torch \
    && pip freeze --path=/deps | grep -i '^torch==' > /tmp/torch-constraint.txt

COPY packages/rag_core /build/packages/rag_core
COPY services/retrieval /build/services/retrieval

# Single pip invocation so rag-core (a local path dependency, not on PyPI) is
# resolved from /build/packages/rag_core instead of failing to find it on PyPI
# when rag-retrieval's own dependency resolution runs as a separate command.
# --extra-index-url keeps the CPU-only torch build reachable so the constraint
# above resolves instead of erroring on "no matching distribution".
RUN pip install --no-cache-dir --target=/deps -c /tmp/torch-constraint.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    /build/packages/rag_core /build/services/retrieval

FROM python:3.11-slim AS runtime

RUN addgroup --system app && adduser --system --ingroup app --home /home/app app

WORKDIR /app

COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY --from=builder /build/services/retrieval/rag_retrieval /app/rag_retrieval

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "rag_retrieval.api:app", "--host", "0.0.0.0", "--port", "8000"]
