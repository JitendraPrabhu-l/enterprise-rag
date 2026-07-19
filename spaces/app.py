"""Production RAG — single-container Hugging Face Spaces variant.

A consolidation of the four-service stack (see the repo root) into one
process sized for Spaces' free tier (2 vCPU / 16 GB), preserving the
architecture's semantics while swapping every server dependency for an
embedded equivalent:

    full stack                          this Space
    ----------------------------------  ----------------------------------
    Qdrant server (dense vectors)       qdrant-client local mode (on disk)
    OpenSearch BM25 (sparse)            bm25s, in-process, rebuilt per ingest
    RRF fusion (ADR-004, k=60)          same formula, same k
    parent 1024 / child 128 chunking    same small-to-big scheme
    Celery ingestion workers            FastAPI BackgroundTasks + job dict
    Groq JSON-mode generation +         same prompt contract, same
      citation schema (ADR-007/012)       citations JSON shape
    guardrails (injection scan,         same checks, same
      citation validation, uncited flag)  guardrail_flagged semantics
    nginx same-origin /api proxy + UI   FastAPI serves the identical UI
    Neo4j GraphRAG / reranker / Redis / dropped — demo-scale trade-offs,
      MinIO / Prometheus stack            documented in README.md

Storage is the Space's ephemeral disk: uploads survive requests but not a
Space restart — acceptable for a public demo (re-upload and go).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import bm25s
import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag-space")

# ---------------------------------------------------------------- settings

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")  # same default as the full stack
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge v1.5 retrieval convention: queries get this instruction prefix,
# passages are embedded bare.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PARENT_TOKENS = 1024  # matches the full stack's parent_chunk_tokens
CHILD_TOKENS = 128  # matches child_chunk_tokens
RRF_K = 60  # ADR-004 standard constant
CANDIDATES_PER_RANKER = 20  # wide retrieval width before fusion (ADR-005)

MAX_UPLOAD_MB = 25
MAX_PAGES = 400
GENERATE_RATE_PER_MIN = 8  # per-IP cap: the Groq key behind this is shared

# ------------------------------------------------------------- rrf fusion


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Identical formula to the full stack's rag_retrieval.fusion —
    score(d) = Σ 1/(k + rank_L(d)) over lists containing d, 1-indexed."""
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    seq = 0
    for ranked in ranked_lists:
        seen: set[str] = set()
        for idx, doc_id in enumerate(ranked):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + idx + 1)
            if doc_id not in order:
                order[doc_id] = seq
                seq += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


# ------------------------------------------------------------ tokenizing
# Word-count proxy for tokens (×0.75 words ≈ 1 token is close enough for
# chunk sizing; the full stack uses a real tokenizer, but chunk-boundary
# placement is not precision-critical).

_WORD_RE = re.compile(r"\S+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _approx_token_len(text: str) -> int:
    return int(len(_words(text)) / 0.75)


# ------------------------------------------------------------- chunking


def _split_oversized(paragraph: str) -> list[str]:
    """Formats without blank-line structure (CSV rows, minified JSON, wall-of-
    text files) can yield one 'paragraph' far beyond PARENT_TOKENS — hard-split
    by words so no parent ever blows up the LLM context."""
    words = _words(paragraph)
    limit = int(PARENT_TOKENS * 0.75)
    if len(words) <= limit:
        return [paragraph]
    return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


def build_parents(pages: list[tuple[int | None, str]], document_id: str) -> list[dict[str, Any]]:
    """Small-to-big scheme, parent half: merge consecutive paragraphs into
    ~PARENT_TOKENS sections, never spanning a page/record boundary. Each
    page is (unit_number, text) — a PDF page, a JSON record, a CSV row
    group, or None for formats with no addressable units (plain text) —
    so citations stay exact wherever exactness exists.
    """
    parents: list[dict[str, Any]] = []
    for page_number, page_text in pages:
        raw = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        paragraphs = [piece for para in raw for piece in _split_oversized(para)]
        buf: list[str] = []
        buf_tokens = 0
        for para in paragraphs:
            para_tokens = _approx_token_len(para)
            if buf and buf_tokens + para_tokens > PARENT_TOKENS:
                parents.append(_parent(document_id, page_number, "\n\n".join(buf)))
                buf, buf_tokens = [], 0
            buf.append(para)
            buf_tokens += para_tokens
        if buf:
            parents.append(_parent(document_id, page_number, "\n\n".join(buf)))
    return parents


def _parent(document_id: str, page_number: int | None, text: str) -> dict[str, Any]:
    return {
        "parent_id": f"{document_id}:{uuid.uuid4().hex[:12]}",
        "document_id": document_id,
        "page_number": page_number,
        "text": text,
    }


def build_children(parent: dict[str, Any]) -> list[dict[str, Any]]:
    """Child half: ~CHILD_TOKENS sliding windows (25% overlap) over the
    parent's words — children are what gets embedded/BM25-indexed, parents
    are what reaches the LLM."""
    words = _words(parent["text"])
    window = max(int(CHILD_TOKENS * 0.75), 24)
    step = max(int(window * 0.75), 12)
    children = []
    for start in range(0, max(len(words) - int(window * 0.25), 1), step):
        text = " ".join(words[start : start + window])
        if not text:
            continue
        children.append(
            {
                "child_id": f"{parent['parent_id']}:{start}",
                "parent_id": parent["parent_id"],
                "text": text,
            }
        )
    return children


# -------------------------------------------------------- file extraction
# Every supported format reduces to list[(unit_number | None, text)] —
# "unit" is whatever addressable granularity the format has (PDF page,
# JSON record, CSV row group), preserved so citations point at something
# a reader can actually find.

SUPPORTED_EXTENSIONS = ".pdf .txt .md .markdown .log .json .jsonl .csv .html .htm .docx"


def _flatten_json(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            lines.extend(_flatten_json(val, f"{prefix}{key}." if isinstance(val, (dict, list)) else f"{prefix}{key}"))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            lines.extend(_flatten_json(item, f"{prefix}[{i}]." if isinstance(item, (dict, list)) else prefix))
    elif value is not None:
        text = str(value).strip()
        if text:
            lines.append(f"{prefix.rstrip('.')}: {text}" if prefix else text)
    return lines


def _json_pages(payload: bytes, *, lines_format: bool) -> list[tuple[int | None, str]]:
    if lines_format:
        records = [
            json.loads(line) for line in payload.decode("utf-8", errors="replace").splitlines() if line.strip()
        ]
    else:
        parsed = json.loads(payload.decode("utf-8", errors="replace"))
        records = parsed if isinstance(parsed, list) else [parsed]
    if len(records) > MAX_PAGES:
        raise ValueError(f"JSON has {len(records)} records; the demo caps at {MAX_PAGES}.")
    pages = []
    for i, record in enumerate(records, start=1):
        text = "\n".join(_flatten_json(record))
        if text.strip():
            pages.append((i, text))  # unit = record number -> citations say "p. <record>"
    return pages


def _csv_pages(payload: bytes) -> list[tuple[int | None, str]]:
    import csv
    import io

    rows = list(csv.reader(io.StringIO(payload.decode("utf-8", errors="replace"))))
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    group_size = 40  # rows per citable unit
    if len(data) > MAX_PAGES * group_size:
        raise ValueError(f"CSV has {len(data)} rows; the demo caps at {MAX_PAGES * group_size}.")
    pages = []
    for g in range(0, len(data), group_size):
        lines = [
            " | ".join(f"{h}={v}" for h, v in zip(header, row) if str(v).strip())
            for row in data[g : g + group_size]
        ]
        pages.append((g // group_size + 1, "\n".join(lines)))
    return pages


def _strip_html(markup: str) -> str:
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag in ("script", "style"):
                self._skip += 1
            elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4"):
                self.parts.append("\n\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style") and self._skip:
                self._skip -= 1

        def handle_data(self, data: str) -> None:
            if not self._skip:
                self.parts.append(data)

    parser = _Text()
    parser.feed(markup)
    return "".join(parser.parts)


def extract_pages(filename: str, payload: bytes) -> list[tuple[int | None, str]]:
    """Dispatch by extension; unknown extensions fall through to a plain-text
    attempt so 'all kinds of files' degrades gracefully rather than 400ing
    on e.g. .rst or extensionless README files. Genuine binaries still fail
    with a clear message."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        import fitz

        with fitz.open(stream=payload, filetype="pdf") as pdf:
            if pdf.page_count > MAX_PAGES:
                raise ValueError(f"PDF has {pdf.page_count} pages; the demo caps at {MAX_PAGES}.")
            return [(i + 1, page.get_text("text")) for i, page in enumerate(pdf)]

    if ext == ".docx":
        import io

        from docx import Document

        doc = Document(io.BytesIO(payload))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(None, text)]

    if ext in (".json", ".jsonl"):
        return _json_pages(payload, lines_format=ext == ".jsonl")

    if ext == ".csv":
        return _csv_pages(payload)

    if ext in (".html", ".htm"):
        return [(None, _strip_html(payload.decode("utf-8", errors="replace")))]

    # .txt/.md/.markdown/.log and anything unknown: treat as plain text.
    text = payload.decode("utf-8", errors="replace")
    if text.count("�") > max(20, len(text) // 100):  # mostly undecodable -> binary
        raise ValueError(
            f"{ext or 'this file type'!s} looks binary — supported formats: {SUPPORTED_EXTENSIONS}"
        )
    return [(None, text)]


# ------------------------------------------------------------ guardrails

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) (instructions|context)",
        r"disregard (your|the) (system prompt|instructions)",
        r"you are now (in )?(developer|dan|jailbreak)",
        r"reveal (your|the) (system prompt|instructions)",
    )
]


def scan_for_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


# ------------------------------------------------------------ the store


class Store:
    """All corpus state: parents/documents in one JSON file, child dense
    vectors in embedded Qdrant, child BM25 rebuilt in memory. Single-writer
    (the ingest lock) — demo-scale by design."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = data_dir / "store.json"
        self.documents: dict[str, dict[str, Any]] = {}
        self.parents: dict[str, dict[str, Any]] = {}
        self.children: list[dict[str, Any]] = []
        if self.meta_path.exists():
            blob = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.documents = blob.get("documents", {})
            self.parents = blob.get("parents", {})
            self.children = blob.get("children", [])

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.qdrant = QdrantClient(path=str(data_dir / "qdrant"))
        self._vp = VectorParams(size=384, distance=Distance.COSINE)  # bge-small dim
        if not self.qdrant.collection_exists("children"):
            self.qdrant.create_collection("children", vectors_config=self._vp)

        self._bm25: Any = None
        self._bm25_ids: list[str] = []
        self._rebuild_bm25()

    def persist(self) -> None:
        self.meta_path.write_text(
            json.dumps(
                {"documents": self.documents, "parents": self.parents, "children": self.children}
            ),
            encoding="utf-8",
        )

    def _rebuild_bm25(self) -> None:
        if not self.children:
            self._bm25, self._bm25_ids = None, []
            return
        corpus = [c["text"] for c in self.children]
        tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        self._bm25 = retriever
        self._bm25_ids = [c["child_id"] for c in self.children]

    # -- write path (mirrors ADR-020: both indexes written in one pipeline,
    #    so hybrid search can never silently degrade to dense-only)

    def add_document(
        self, doc_id: str, title: str, source_domain: str, page_count: int,
        parents: list[dict[str, Any]], children: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> None:
        from qdrant_client.models import PointStruct

        self.documents[doc_id] = {
            "title": title, "source_domain": source_domain, "page_count": page_count,
        }
        for p in parents:
            self.parents[p["parent_id"]] = p
        self.children.extend(children)
        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=vec,
                payload={"child_id": c["child_id"], "parent_id": c["parent_id"]},
            )
            for c, vec in zip(children, vectors)
        ]
        for i in range(0, len(points), 256):
            self.qdrant.upsert("children", points[i : i + 256])
        self._rebuild_bm25()
        self.persist()

    # -- read path

    def dense_search(self, vector: list[float], limit: int) -> list[str]:
        hits = self.qdrant.query_points("children", query=vector, limit=limit).points
        return [h.payload["child_id"] for h in hits if h.payload]

    def sparse_search(self, query: str, limit: int) -> list[str]:
        if self._bm25 is None:
            return []
        try:
            tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
            limit = min(limit, len(self._bm25_ids))
            ids, _scores = self._bm25.retrieve(tokens, k=limit, show_progress=False)
            return [self._bm25_ids[int(i)] for i in ids[0]]
        except Exception:
            # An all-stopwords query tokenizes to nothing and bm25s balks —
            # hybrid retrieval degrades to dense-only for that query, which
            # is the correct fallback (mirrors the full stack's behavior
            # when the sparse leg returns no hits).
            logger.warning("sparse leg failed for query %r — dense-only this query", query[:60])
            return []

    def parent_of_child(self, child_id: str) -> dict[str, Any] | None:
        return self.parents.get(child_id.rsplit(":", 1)[0])


# ---------------------------------------------------------- the embedder


class Embedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [BGE_QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


# ------------------------------------------------------------- generation

SYSTEM_PROMPT = """\
You are a careful, precise research assistant that answers questions strictly \
from the provided context documents.

## Grounding rules
- Answer using ONLY the information present in the "Context" section. Do not \
use outside knowledge, training data, or assumptions to fill gaps.
- If the context does not contain enough information, say so plainly rather \
than guessing or fabricating an answer.
- Never invent facts, figures, names, dates, or citations not directly \
supported by the context.
- If context passages conflict, note the conflict rather than silently \
picking one side.

## Citation rules
- Every factual claim must be traceable to at least one context passage.
- Populate `citations` with one entry per distinct passage actually used, \
copying parent_id/document_id/page_number exactly from that passage's header.\
"""

JSON_INSTRUCTIONS = """\
Respond with ONLY a single JSON object (no markdown fences, nothing before or \
after) of exactly this shape:

{"answer": "<string>", "citations": [{"parent_id": "<string>", \
"document_id": "<string>", "page_number": <integer or null>}]}

If the context does not support an answer, return an empty citations array.\
"""


def build_context_block(parents: list[dict[str, Any]]) -> str:
    if not parents:
        return "No relevant context passages were retrieved for this query."
    sections = []
    for i, p in enumerate(parents):
        header = (
            f"[Passage {i + 1}] parent_id={p['parent_id']} "
            f"document_id={p['document_id']}, page {p['page_number']}"
        )
        sections.append(f"{header}\n{p['text']}")
    return "## Context\n\n" + "\n\n".join(sections)


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Tolerant parse: strict first, then the outermost {...} slice —
    JSON-mode models occasionally still wrap output in prose or fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


async def generate_answer(query: str, parents: list[dict[str, Any]], store: Store) -> dict[str, Any]:
    if not GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY is not configured on this Space.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{build_context_block(parents)}\n\n## Question\n{query}\n\n{JSON_INSTRUCTIONS}",
        },
    ]
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
    if resp.status_code == 429:
        raise HTTPException(429, "The shared LLM quota is momentarily exhausted — try again shortly.")
    if resp.status_code != 200:
        logger.error("Groq error %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(502, "LLM backend error.")

    request_id = uuid.uuid4().hex
    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        parsed = _extract_json_object(raw)
    except json.JSONDecodeError:
        return {
            "request_id": request_id, "answer": raw.strip(),
            "citations": [], "model": GROQ_MODEL, "guardrail_flagged": True,
        }

    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        # Observed live: for off-corpus questions the model sometimes
        # returns "" instead of a spoken refusal — safe but renders as a
        # blank bubble in the UI. Substitute the standard refusal.
        answer = "The provided context does not contain enough information to answer this question."
    valid_parent_ids = {p["parent_id"] for p in parents}
    citations = []
    for c in parsed.get("citations", []) or []:
        if not isinstance(c, dict) or c.get("parent_id") not in valid_parent_ids:
            continue  # citation must point at context actually provided
        parent = store.parents.get(c["parent_id"], {})
        doc = store.documents.get(parent.get("document_id", ""), {})
        citations.append(
            {
                "parent_id": c["parent_id"],
                "document_id": parent.get("document_id"),
                "page_number": parent.get("page_number"),
                "title": doc.get("title") or parent.get("document_id"),
            }
        )

    # Uncited-answer guardrail (same semantics as the full stack): a
    # substantive answer with zero valid citations is flagged, a refusal
    # ("context doesn't contain...") legitimately has none.
    refusal = bool(re.search(r"(context|passages?).{0,40}(not|n't|no ).{0,40}(contain|enough|support|information)", answer, re.IGNORECASE))
    flagged = bool(answer) and not citations and not refusal

    return {
        "request_id": request_id, "answer": answer, "citations": citations,
        "model": GROQ_MODEL, "guardrail_flagged": flagged,
    }


# ------------------------------------------------------------- app state

store: Store | None = None
embedder: Embedder | None = None
jobs: dict[str, dict[str, Any]] = {}
_ingest_lock = asyncio.Lock()
_generate_sem = asyncio.Semaphore(2)
_rate: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global store, embedder
    store = Store(DATA_DIR)
    embedder = Embedder(EMBED_MODEL)  # cached in the image — no download here
    logger.info(
        "ready: %d documents, %d parents, %d children",
        len(store.documents), len(store.parents), len(store.children),
    )
    yield


app = FastAPI(title="Production RAG — Spaces demo", lifespan=lifespan)


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = [t for t in _rate.get(ip, []) if now - t < 60]
    if len(window) >= GENERATE_RATE_PER_MIN:
        raise HTTPException(429, "Rate limit: this shared demo allows a few queries per minute.")
    window.append(now)
    _rate[ip] = window


# ---------------------------------------------------------------- routes


@app.get("/api/health/{svc}")
async def health(svc: str) -> JSONResponse:
    if svc not in {"ingestion", "retrieval", "generation", "eval"}:
        raise HTTPException(404, f"unknown service {svc!r}")
    return JSONResponse({"status": "ok", "service": svc, "consolidated": True})


@app.post("/api/ingest")
async def ingest(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    source_domain: str = Form("general"),
    title: str = Form(""),
) -> dict[str, str]:
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the demo's {MAX_UPLOAD_MB} MB limit.")

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    background.add_task(
        _run_ingest, job_id, payload, file.filename or "document",
        title or (file.filename or "document"), source_domain,
    )
    return {"job_id": job_id}


def _ingest_sync(payload: bytes, filename: str, title: str, source_domain: str) -> dict[str, Any]:
    """CPU-bound pipeline half, run in a worker thread: parse (any supported
    format) → chunk → embed → index (both dense and sparse — the ADR-020
    invariant)."""
    assert store is not None and embedder is not None
    started = time.monotonic()
    pages = extract_pages(filename, payload)

    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    parents = build_parents(pages, doc_id)
    if not parents:
        raise ValueError("No extractable text found (scanned/image-only or empty file?).")
    children = [c for p in parents for c in build_children(p)]
    vectors = embedder.embed_passages([c["text"] for c in children])
    store.add_document(doc_id, title, source_domain, len(pages), parents, children, vectors)

    return {
        "document_id": doc_id,
        "page_count": len(pages),
        "parent_count": len(parents),
        "chunk_count": len(children),
        "duration_seconds": time.monotonic() - started,
    }


async def _run_ingest(
    job_id: str, payload: bytes, filename: str, title: str, source_domain: str
) -> None:
    async with _ingest_lock:  # one ingest at a time on 2 vCPUs
        jobs[job_id]["status"] = "running"
        try:
            result = await asyncio.to_thread(_ingest_sync, payload, filename, title, source_domain)
            jobs[job_id].update(status="succeeded", result=result)
            logger.info("ingest %s done: %s", job_id[:8], result)
        except Exception as exc:
            logger.exception("ingest %s failed", job_id[:8])
            jobs[job_id].update(status="failed", error=str(exc))


@app.get("/api/ingest/{job_id}")
async def ingest_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.post("/api/generate")
async def generate(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    _rate_limit(request)
    assert store is not None and embedder is not None

    query = str(body.get("query", "")).strip()
    if not query:
        raise HTTPException(400, "query is required")
    top_n = max(1, min(int(body.get("top_n", 5) or 5), 20))
    domains = body.get("source_domains") or None

    if not store.children:
        return {
            "request_id": uuid.uuid4().hex,
            "answer": "No documents have been ingested yet — upload a PDF first.",
            "citations": [], "model": GROQ_MODEL, "guardrail_flagged": False,
        }

    async with _generate_sem:
        # Hybrid retrieval: dense + BM25 over children, RRF-fused (k=60),
        # then children → parents (small-to-big), dedupe preserving order.
        query_vector = await asyncio.to_thread(embedder.embed_query, query)
        dense = store.dense_search(query_vector, CANDIDATES_PER_RANKER)
        sparse = store.sparse_search(query, CANDIDATES_PER_RANKER)
        fused = reciprocal_rank_fusion([dense, sparse])

        parents: list[dict[str, Any]] = []
        seen: set[str] = set()
        for child_id, _score in fused:
            parent = store.parent_of_child(child_id)
            if parent is None or parent["parent_id"] in seen:
                continue
            if domains:
                doc = store.documents.get(parent["document_id"], {})
                if doc.get("source_domain") not in domains:
                    continue
            if scan_for_injection(parent["text"]):
                logger.warning("guardrail: dropped injected-looking passage %s", parent["parent_id"])
                continue
            seen.add(parent["parent_id"])
            parents.append(parent)
            if len(parents) >= top_n:
                break

        return await generate_answer(query, parents, store)


@app.post("/api/feedback", status_code=202)
async def feedback(body: dict[str, Any]) -> dict[str, str]:
    """Answer feedback (mirrors the full stack's ADR-027) — logged only:
    this single-container demo has no metrics/eval pipeline to feed, so
    there's nothing to increment. Shares the full stack's UI code
    (static/app.js), which is why the shape matches exactly.
    """
    rating = body.get("rating")
    if rating not in ("up", "down"):
        raise HTTPException(422, "rating must be 'up' or 'down'")
    logger.info(
        "answer_feedback request_id=%s rating=%s query=%r",
        body.get("request_id"), rating, (body.get("query") or "")[:200],
    )
    return {"status": "recorded"}


# Static UI last, so /api/* wins the route match. html=True serves
# index.html at / — same behavior as the full stack's nginx.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.exception_handler(404)
async def spa_fallback(request: Request, exc: Any):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(Path(__file__).parent / "static" / "index.html")
