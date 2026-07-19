# Hugging Face Docker Space — single container serving the consolidated RAG
# demo (see app.py's module docstring for what maps to what in the full
# stack). Spaces routes traffic to app_port from README.md's front-matter
# (7860, the platform convention).

FROM python:3.11-slim

# Everything runs as uid 1000 — the user Spaces expects; root-owned files
# would make the runtime unable to write caches/data.
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
# CPU torch wheel index: the free tier has no GPU, and default torch wheels
# drag in ~2 GB of CUDA libraries the container can never use.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Bake the embedding model into the image so first startup doesn't spend
# minutes downloading (Spaces cold-starts would otherwise hit the health
# check timeout). HF_HOME must stay writable at runtime — the library
# takes file locks even on cache hits.
ENV HF_HOME=/app/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && chown -R appuser:appuser /app

# Ephemeral corpus storage (uploads survive requests, not Space restarts).
RUN mkdir -p /data && chown appuser:appuser /data
ENV DATA_DIR=/data

COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser static ./static

USER appuser
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
