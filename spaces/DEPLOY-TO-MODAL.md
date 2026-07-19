# Deploying the demo to Modal (free, no card)

> Why Modal: as of July 2026, Hugging Face gates Docker/Gradio Spaces
> behind PRO ($9/mo) — Static Spaces only on the free plan. Modal's
> Starter tier gives **$30/month in compute credits with no payment
> method**; when credits run out workloads simply stop (nothing to bill).
> With scale-to-zero and demo traffic, this app uses a small fraction of
> that. Bonus over Spaces: uploads persist on a Modal Volume instead of
> resetting on every restart.

Everything runs from your machine's terminal — Modal builds the container
remotely (no local Docker needed).

## 1. Account (free, no card)

modal.com → Sign Up (GitHub or Google auth).

## 2. One-time local setup

```powershell
D:\RAG\.venv\Scripts\python.exe -m pip install modal
D:\RAG\.venv\Scripts\python.exe -m modal setup     # opens browser to authenticate
```

## 3. Create the Groq secret

Use a **fresh** key from console.groq.com/keys (treat any key that has been
pasted into chats/repos as burned):

```powershell
D:\RAG\.venv\Scripts\python.exe -m modal secret create groq GROQ_API_KEY=gsk_...your-new-key...
```

(or dashboard → Secrets → New secret → name `groq`, key `GROQ_API_KEY`.)

## 4. Deploy

```powershell
cd D:\RAG\spaces
D:\RAG\.venv\Scripts\python.exe -m modal deploy modal_app.py
```

First deploy takes ~5 min (remote image build: CPU torch + model bake —
cached for later deploys). It prints the public URL, shaped like:

```
https://<your-workspace>--production-rag-demo-web.modal.run
```

Share that URL — no login needed for visitors.

## 5. Verify

Same checklist as any deployment of this demo: four green health chips →
upload a small PDF → "Ingested ✓" → ask a question it answers (expect
page-cited answer) → ask something off-corpus (expect an explicit refusal).

## Facts worth knowing

- **Cold starts**: after 5 idle minutes the container scales to zero;
  the next visitor waits ~15–30 s while it wakes (model is baked into the
  image, so no download — just process start + model load).
- **Persistence**: `/data` is a Modal Volume — corpora survive scale-down
  and redeploys. Wipe it with `modal volume delete rag-demo-data`.
- **Cost visibility**: dashboard → Usage. Idle = $0; a day of active demo
  traffic is cents against the $30 monthly credit.
- **Updating**: edit app.py/static and re-run `modal deploy` — same URL.
- **Logs**: `modal app logs production-rag-demo` or the dashboard.
