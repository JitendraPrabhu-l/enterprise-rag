# Deploying this demo to Hugging Face Spaces

> **⚠️ Superseded for free accounts (2026-07-13):** Hugging Face now gates
> Docker/Gradio Spaces behind PRO ($9/mo) — verified live during this
> project; only Static Spaces stay free. This guide still works if you
> have PRO or a grandfathered account. The free no-card path is Modal —
> see **DEPLOY-TO-MODAL.md** (same container, plus persistent uploads).

Everything in this `spaces/` directory is the complete, self-contained
Space. Total upload: 6 files (`README.md`, `Dockerfile`, `requirements.txt`,
`app.py`, `static/` × 3). No git required — the drag-and-drop web upload is
enough.

## 1. Account (free, no card)

huggingface.co → Sign Up → verify email.

## 2. Create the Space

1. huggingface.co/new-space
2. **Space name**: e.g. `production-rag-demo`
3. **License**: MIT
4. **SDK**: select **Docker** → "Blank" template
5. **Hardware**: CPU basic · 2 vCPU · 16 GB · **FREE**
6. **Visibility**: Public
7. Create Space.

## 3. Add the Groq secret (before uploading, so the first build already has it)

Space page → **Settings** → **Variables and secrets** → **New secret**:

- Name: `GROQ_API_KEY`
- Value: your key from console.groq.com/keys

Use a **fresh key**, not one that's been pasted into chats/repos before.
Note the demo is public: visitors' queries spend this key's free-tier
quota. The app rate-limits per visitor (8 queries/min) to keep one person
from draining it, but a popular day can exhaust the daily quota — that
surfaces to users as a polite 429, not an error page.

## 4. Upload the files

Space page → **Files** tab → **Add file → Upload files**, then drag in:

```
README.md            (replaces the auto-generated one — contains the
                      required front-matter: sdk: docker, app_port: 7860)
Dockerfile
requirements.txt
app.py
static/index.html    (upload the static/ folder's three files with the
static/app.js         folder path preserved — the uploader keeps relative
static/styles.css     paths if you drag the whole folder)
```

Commit directly to `main`. The build starts automatically — first build
takes ~5–10 min (CPU torch install + model bake). Watch it under the
**Logs** tab; when it flips to **Running**, the public URL is

```
https://<your-username>-production-rag-demo.hf.space
```

(also reachable from the Space page's "Embed/App" view). Share that URL —
no login needed for visitors.

## 5. Verify

1. Open the URL — all four health chips should turn green (they all
   report from the one consolidated container).
2. Upload a small PDF, wait for "Ingested ✓ … ready to query".
3. Ask something the document answers — expect an answer with page-level
   citations; then ask something it can't answer — expect an explicit
   refusal, or a "guardrail flagged" badge if the model answers without
   citations.

## Facts worth knowing

- **Storage is ephemeral**: a Space restart (redeploy, weekly platform
  maintenance) clears uploaded corpora. Fine for a demo; persistent disk
  is a paid Spaces feature.
- **Cold sleep**: free Spaces sleep after ~48 h without traffic and wake
  on the next visit (~30–60 s, model is pre-baked so no re-download).
- **Updating**: edit/re-upload any file in the Files tab → auto-rebuild.
- **This is the demo-scale variant.** The full stack (four services,
  Qdrant/OpenSearch/Neo4j/Redis/MinIO, Prometheus/Grafana/Loki, eval
  harness) deploys via `deploy/oracle/` once an Always-Free ARM VM is
  obtainable — see `docs/DEPLOY-ORACLE.md` at the repo root.
