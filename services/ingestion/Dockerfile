# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip hatchling

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PYTHONPATH=/install/lib/python3.11/site-packages

# sentence-transformers pulls in torch; installed CPU-only up front (this service
# has no GPU) so the later resolve doesn't pick torch's default Linux wheel, which
# drags in several GB of unused CUDA/triton libraries as transitive deps. PYTHONPATH
# above lets the second pip install (same --prefix) see it as already satisfied.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu torch

# Copy the shared rag_core package first (path dependency) so it builds
# into the same venv before the ingestion service itself is installed.
COPY packages/rag_core /build/packages/rag_core
COPY services/ingestion /build/services/ingestion

# Single pip invocation so rag-core (a local path dependency, not on PyPI) is
# resolved from /build/packages/rag_core instead of failing to find it on PyPI
# when rag-ingestion's own dependency resolution runs as a separate command.
RUN pip install --no-cache-dir --prefix=/install \
    /build/packages/rag_core /build/services/ingestion

FROM python:3.11-slim AS runtime

# Table/figure rendering and pypdf's image extraction rely on system zlib/jpeg
# libs already present in the slim image; nothing extra needed here.
RUN groupadd --system rag && useradd --system --gid rag --create-home rag

COPY --from=builder /install /usr/local

WORKDIR /app
COPY services/ingestion/rag_ingestion /app/rag_ingestion

ENV PYTHONUNBUFFERED=1 \
    PATH=/usr/local/bin:$PATH

RUN mkdir -p /tmp/rag-ingestion-uploads && chown -R rag:rag /tmp/rag-ingestion-uploads /app

USER rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)" || exit 1

CMD ["uvicorn", "rag_ingestion.api:app", "--host", "0.0.0.0", "--port", "8000"]
