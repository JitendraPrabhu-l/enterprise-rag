# Production RAG

Retrieval-augmented generation over large, dense, multimodal public corpora — tables, figures, and charts included. Four independently deployable services (ingestion, retrieval, generation, eval) sharing one schema contract, running self-hosted via Docker Compose. Every LLM call (generation, vision, query expansion/HyDE, GraphRAG extraction, RAG Triad judging) runs on Groq; every other dependency (vector store, sparse search, graph store, embeddings, reranker) is self-hosted to keep the stack cheap to operate.

See [`docs/HLD.md`](docs/HLD.md) for the architecture and [`docs/adr/`](docs/adr) for the 33 architecture decision records this implementation follows.

## Architecture

```
[ Raw documents ]
       │  parsing, page classification, vision description (ADR-001)
       │  parent-child + semantic chunking (ADR-002)
       │  contextual retrieval enrichment (ADR-023, on by default)
       ▼
[ ingestion service ]──── embeds + upserts ───▶ [ Qdrant ]  (ADR-003)
       │                                        [ OpenSearch ] (ADR-004, BM25)
       │  optional triple extraction (ADR-006)   [ Neo4j ]   (ADR-006, GraphRAG)
       ▼
[ retrieval service ] ── hybrid search + RRF + rerank (ADR-004, ADR-005)
       │  query decomposition for multi-hop questions (ADR-025)
       │  document-level ACL + tenant pre-filter (ADR-010, ADR-024)
       ▼
[ generation service ] ── semantic answer cache (ADR-026, skips retrieval+LLM on a scoped hit)
                           compression + Groq generation (ADR-007, ADR-008, ADR-012)
                           input/output guardrails (ADR-010)
                           answer feedback capture (ADR-027)
       ▼
[ eval service ] ── RAG Triad judges + CI gate + production sampling (ADR-009)
```

Every service is a standalone FastAPI app; they talk to each other over HTTP using the shared Pydantic contracts in `packages/rag_core`.

### Enterprise-grade additions (ADR-023 – ADR-033)

- **Contextual retrieval** (ADR-023) — each chunk gets a short, LLM-generated situating summary prepended before embedding/BM25 indexing (`ChunkRecord.searchable_text`), cutting retrieval misses on chunks that lose their antecedent in isolation. The generator still reads the raw, unprefixed passage. On by default (`CONTEXTUAL_ENRICHMENT_ENABLED=true`); per-chunk failures fall back to raw-text indexing without failing the ingest job.
- **Document-level ACLs** (ADR-024) — `DocumentMetadata.allowed_principals` is enforced as a hard pre-filter in both Qdrant and OpenSearch, alongside the existing tenant filter (ADR-010) and with the same fail-closed guarantee: no caller principals means the `"public"` tier only, never "see everything."
- **Multi-hop query decomposition** (ADR-025) — `query_strategy=decompose` splits a comparison/multi-entity question into single-fact sub-questions, each independently hybrid-searched and RRF-fused, so "how does X compare to Y" retrieves passages about both.
- **Semantic answer cache** (ADR-026) — a query embedding-similar (≥0.95 cosine) to one already answered *within the same tenant/principal/domain scope* returns the cached answer, skipping retrieval and generation entirely. Scope partitioning — not just similarity — is what keeps a cache hit from ever crossing the ACL boundary ADR-024 enforces. Guardrail-flagged answers are never cached.
- **Answer feedback loop** (ADR-027) — 👍/👎 on every answer (`POST /feedback`) feeds `rag_answer_feedback_total` and structured logs, the production-failures-become-eval-cases signal that complements the design-time golden dataset (ADR-009).
- **Citation grounding verification** (ADR-028) — every citation's `parent_id` is checked against the chunks actually retrieved, catching schema-valid citations that name content the model never saw (a fabricated or training-recalled identifier). Deterministic, no added latency; flagged at serve time and a hard CI gate failure in the eval service, independent of the RAG Triad's graded scores.
- **ColPali visual retrieval** (ADR-029, opt-in) — table/figure-dense pages get an additional late-interaction visual index (`COLPALI_ENABLED=false` by default) alongside the existing vision-description text path, preserving fine-grained visual structure a text description can lose. A genuinely heavy, opt-in dependency (`pip install rag-ingestion[colpali]`); indexing only in this pass, query-time fusion is tracked follow-up work.
- **Vector quantization** (ADR-003, now actually wired) — a domain's Qdrant collection gets scalar INT8 quantization enabled once it crosses 50M vectors, checked idempotently after each ingest (`VectorStore.enable_quantization_if_due`).
- **Service trust boundary** (ADR-030) — no service authenticates individual requests; the real, deployment-enforced boundary is network isolation (the Oracle overlay zeroes every internal service's host port, the Helm chart exposes only the UI). A closed arbitrary-local-file-read (`file_path` in `/ingest` is now constrained to `IngestionSettings.upload_dir`) is the concrete fix that came out of auditing this boundary.
- **Coverage measurement** (ADR-031) — every package's suite runs under `pytest-cov` in CI (term-missing + an XML artifact), reporting the honest baseline (32%–84% across the five packages) without an artificial uniform gate that would either be toothless for the strong packages or immediately fail the weaker ones for pre-existing gaps.
- **Dependency lockfiles + vulnerability scanning** (ADR-032) — every package gets a `uv.lock` plus a `pip-audit` CI job (informational, matrix'd per package). Three real CVE ranges were closed this pass (`pypdf`, `torch`, `transformers` version floors, each re-verified against the full test suite); two residual findings (`starlette`, `transformers`'s post-5.0 CVEs) are documented as blocked by other direct dependencies' own version ceilings, not silently unresolved.
- **Normalized error-response contract** (ADR-033) — `rag_core.errors.install_error_handlers` flattens FastAPI's own request-validation errors (a structured list) into the same `{"detail": "<string>"}` shape every hand-raised exception in this codebase already returns, wired identically into all four services.

## Prerequisites

- Docker + Docker Compose v2
- A Groq API key ([console.groq.com/keys](https://console.groq.com/keys)) — every LLM call in the stack
- ~4 GB RAM free for Qdrant + OpenSearch + Neo4j + the four services

## Quickstart

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY

make up
# or: docker compose up -d --build
```

This starts:

| Service        | Port        | Purpose                                                   |
| -------------- | ----------- | --------------------------------------------------------- |
| `ingestion`  | 8001        | `POST /ingest` — parse, chunk, embed, index a document |
| `retrieval`  | 8002        | `POST /retrieve` — hybrid search + rerank              |
| `generation` | 8003        | `POST /generate` — grounded answer with citations      |
| `eval`       | 8004        | `POST /score` — RAG Triad scoring for one interaction  |
| `qdrant`     | 6333        | Dense vector store                                        |
| `opensearch` | 9200        | Sparse (BM25) search                                      |
| `neo4j`      | 7474 / 7687 | GraphRAG store (browser UI on 7474)                       |
| `ui`         | 8080        | Web UI — upload documents, chat with citations (ADR-021)  |
| `grafana`    | 3000        | Dashboards: metrics, logs, traces (anonymous, dev only)   |
| `prometheus` | 9090        | Metrics + alert rule evaluation (ADR-021)                 |
| `alertmanager` | 9093      | Alert routing (swap the webhook receiver to page humans)  |
| `phoenix`    | 6006        | RAG-specific trace explorer (ADR-016)                     |

Then open **http://localhost:8080** — upload a PDF, watch the ingestion job
complete, and ask questions with page-level citations. Or check services by hand:

```bash
curl http://localhost:8001/healthz
curl http://localhost:8002/healthz
curl http://localhost:8003/healthz
curl http://localhost:8004/healthz
```

### Ingest a document

```bash
curl -X POST http://localhost:8001/ingest \
  -F "file=@/path/to/document.pdf" \
  -F "source_domain=demo-corpus" \
  -F "tenant_id=public"
```

### Ask a question

```bash
curl -X POST http://localhost:8003/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about X?", "tenant_id": "public", "source_domains": ["demo-corpus"]}'
```

Response matches `GenerationResponse` in `packages/rag_core/rag_core/schemas.py`: `answer`, `citations`, `model`, `used_graph`, `guardrail_flagged`.

## Local development

`docker-compose.override.yml` is picked up automatically by `docker compose up` and mounts each service's source with `--reload`, so edits under `services/*/rag_*` take effect without a rebuild.

To run a single service outside Docker:

```bash
cd services/retrieval
pip install -e .
pip install -e ../../packages/rag_core
python -m uvicorn rag_retrieval.api:app --reload
```

You'll need Qdrant/OpenSearch/Neo4j reachable — either run the full `docker compose up` and point `QDRANT_URL` etc. at `localhost`, or run just the storage services: `docker compose up -d qdrant opensearch neo4j`.

## Testing

```bash
make test        # unit tests for every service + rag_core
make lint         # ruff check + format check
make typecheck    # mypy --strict across all packages
```

Coverage (ADR-031) is measured in CI via `pytest --cov` per package (term-missing + an XML artifact); the baseline is honestly uneven across the five packages (32%–84%) and is not yet gated — see the ADR for why a uniform floor would be wrong in either direction today. Run locally with `pytest --cov=<package> --cov-report=term-missing` from any package directory.

## Observability & alerting (ADR-021)

- **Metrics**: every service exposes `/metrics` (RED signals per endpoint plus
  `rag_guardrail_flags_total` / `rag_ingest_jobs_total`). Prometheus scrapes them;
  alert rules cover service-down, 5xx rate, p95 latency, and guardrail-flag spikes.
  Alertmanager ships with a no-op webhook receiver — point it at Slack/email to page.
- **Logs**: Promtail tails every container in this compose project into Loki,
  labeled by service; query them in Grafana or the provisioned dashboard's log panel.
- **Traces**: OTLP spans fan out to Tempo (Grafana) and Phoenix (ADR-016).
- **Dashboard**: Grafana auto-provisions the "RAG Overview" dashboard
  (`deploy/grafana/dashboards/rag-overview.json`) with all three datasources.

## CI / eval gate

`.github/workflows/ci.yml` runs lint → typecheck → unit tests (with coverage) → Docker builds on every push/PR, and on PRs additionally brings up the full stack and runs the RAG Triad eval gate (`services/eval/scripts/run_eval_gate.py`) against a synthetic dataset, failing the build if faithfulness, answer relevance, or context precision drop below threshold (ADR-009). A separate `security-audit` job runs `pip-audit` against each package's `uv.lock` (informational — see ADR-032 for the two residual findings it currently reports, both blocked by other dependencies' own version constraints).

Each package's `uv.lock` (ADR-032) must be regenerated with `uv lock` (run from that package's directory) whenever its `pyproject.toml` dependencies change.

Run the gate locally once the stack is up:

```bash
make eval-gate
```

## Repository layout

```
packages/rag_core/          # shared schemas, config, tracing, Qdrant client — every service depends on this
services/ingestion/         # parsing, chunking, embedding, GraphRAG extraction
services/retrieval/         # hybrid search, RRF, rerank, GraphRAG query routing
services/generation/        # compression, Groq generation, guardrails
services/eval/               # RAG Triad judges, synthetic data, CI gate, production sampling
deploy/docker/               # OTel collector + Tempo config
deploy/grafana/               # Grafana datasources + dashboards (provisioned as code)
deploy/prometheus/            # scrape config + alert rules
deploy/alertmanager/          # alert routing
deploy/loki/  deploy/promtail/ # log aggregation
deploy/helm/                  # Kubernetes chart (ADR-018); deploy/terraform/ = kind cluster (ADR-019)
ui/                           # web UI (static, served by nginx with /api reverse proxy)
docs/HLD.md                  # high-level design
docs/adr/                    # architecture decision records ADR-001..022
docker-compose.yml            # full stack
docker-compose.override.yml   # dev-mode hot reload (auto-applied)
.github/workflows/ci.yml      # lint, typecheck, test, build, eval-gate
```

## Configuration

All services read from environment variables via `pydantic-settings` (see `rag_core.config.BaseServiceSettings` and each service's `config.py`). `.env.example` documents every variable; `docker-compose.yml` wires them into each container. Only `GROQ_API_KEY` needs a real value to run locally — everything else defaults to the in-stack service names.

## Free public deployment (ADR-022)

The whole stack — services, stores, observability, UI — runs 24/7 for free on an
Oracle Always Free ARM VM (4 OCPU / 24 GB), publicly reachable, with only the UI
and an authenticated Grafana exposed. Full runbook: [`docs/DEPLOY-ORACLE.md`](docs/DEPLOY-ORACLE.md);
on a fresh VM it's one command:

```bash
GROQ_API_KEY=gsk_... bash deploy/oracle/setup.sh
```

## Production notes

This Compose stack is sized for development and small-scale deployment. For the 10M+ document / ~500M chunk envelope described in the HLD:

- Shard Qdrant collections per source domain (already implemented in `rag_core.vector_store`) and enable scalar quantization past ~50M vectors/collection.
- Move OpenSearch and Neo4j to managed or clustered deployments — the single-node containers here are not HA.
- Scale the Celery ingestion worker pool horizontally (`docker compose up -d --scale ingestion-worker=N`, or bump the worker Deployment's replicas in the Helm chart) — the API/queue split (ADR-015) already supports it.
- Point `OTEL_EXPORTER_OTLP_ENDPOINT` at a durable trace backend instead of the local Tempo container.

See `docs/HLD.md` § Open risks for the specific scaling gaps called out at design time.
