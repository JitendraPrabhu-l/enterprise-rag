---
title: Production RAG Demo
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Production RAG — live demo

Upload a PDF, then ask questions about it — answers come back grounded with
page-level citations, or flagged when the model couldn't cite its sources.

This Space is the single-container consolidation of a four-service
production RAG stack (ingestion / retrieval / generation / eval behind an
nginx-fronted UI, with Qdrant, OpenSearch, Neo4j, Redis, MinIO, and a full
Prometheus/Grafana/Loki observability train). What's preserved here, at
demo scale:

- **Hybrid retrieval** — dense (bge-small-en-v1.5 via embedded Qdrant) +
  sparse BM25 (bm25s, in-process), fused with Reciprocal Rank Fusion
  (k=60), exactly the full stack's formula.
- **Small-to-big chunking** — ~128-token children are what's indexed,
  ~1024-token parent sections (never spanning a page boundary) are what
  the LLM reads, so citations carry exact page numbers.
- **Grounded generation** — Groq (`openai/gpt-oss-120b`, the full stack's
  model) in JSON mode with a strict citation schema; citations are
  validated against the passages actually provided (an answer can't cite
  what it wasn't shown).
- **Guardrails** — prompt-injection-looking passages are dropped from
  context, and substantive answers with zero valid citations get a visible
  "guardrail flagged" badge instead of silent confidence.

Deliberately dropped for the free tier (2 vCPU / 16 GB): the GraphRAG /
Neo4j leg, the cross-encoder reranker, Celery workers, object storage, and
the observability stack. Storage is ephemeral — uploaded corpora reset when
the Space restarts; just re-upload.

Accepted formats: PDF, TXT/Markdown, JSON/JSONL (each record becomes a
citable unit), CSV (row groups), HTML, DOCX — plus a plain-text fallback
for anything else that decodes. Limits: 25 MB / 400 pages-or-records per
upload, a few queries per minute per visitor (the LLM behind this runs on
a shared free-tier key).

## Configuration (Space secrets)

| Secret | Required | Meaning |
|---|---|---|
| `GROQ_API_KEY` | **yes** | free key from console.groq.com — generation returns 503 without it |
| `GROQ_MODEL` | no | defaults to `openai/gpt-oss-120b` (the full stack's model) |
