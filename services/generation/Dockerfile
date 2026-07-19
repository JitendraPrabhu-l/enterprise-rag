# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip hatch

COPY packages/rag_core /build/packages/rag_core
COPY services/generation /build/services/generation

# Single pip invocation so rag-core (a local path dependency, not on PyPI) is
# resolved from /build/packages/rag_core instead of failing to find it on PyPI
# when rag-generation's own dependency resolution runs as a separate command.
RUN pip install --no-cache-dir --target=/deps \
    /build/packages/rag_core /build/services/generation

FROM python:3.11-slim AS runtime

RUN addgroup --system app && adduser --system --ingroup app --home /home/app app

WORKDIR /app

COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY --from=builder /build/services/generation/rag_generation /app/rag_generation

ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "rag_generation.api:app", "--host", "0.0.0.0", "--port", "8000"]
