"""Modal deployment wrapper for the consolidated RAG demo (app.py).

Written when Hugging Face put Docker/Gradio Spaces behind PRO (July 2026 —
Static Spaces only on the free plan); Modal's Starter tier ($30/month
credits, no card, workloads stop rather than bill when credits run out)
became the no-card free host. app.py itself is platform-agnostic and
unchanged — this file is only packaging:

    modal deploy modal_app.py
    -> https://<workspace>--production-rag-demo-web.modal.run

Design notes:
- The embedding model is baked into the image (same trick as the
  Dockerfile) so cold starts don't download ~130 MB.
- /data lives on a modal.Volume: unlike HF Spaces' ephemeral disk, uploaded
  corpora survive scale-to-zero and redeploys.
- max_containers=1 on purpose: the store (embedded Qdrant + in-memory BM25)
  is single-writer by design; two containers sharing the volume would race
  the qdrant-local file lock. One 2-vCPU container matches the demo's
  scale anyway.
"""

import modal

app = modal.App("production-rag-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU torch wheels — ~2 GB smaller than the CUDA-bundled default.
    .pip_install(
        "torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cpu"
    )
    .pip_install(
        "sentence-transformers==3.3.1",
        "qdrant-client==1.12.1",
        "bm25s==0.3.9",
        "PyMuPDF==1.25.1",
        "python-docx==1.1.2",
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
        "httpx==0.28.1",
    )
    .env({"HF_HOME": "/root/hf-cache"})
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer;"
        " SentenceTransformer('BAAI/bge-small-en-v1.5')\""
    )
    .add_local_file("app.py", "/root/app.py")
    .add_local_dir("static", "/root/static")
)

data_volume = modal.Volume.from_name("rag-demo-data", create_if_missing=True)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("groq")],  # must contain GROQ_API_KEY
    volumes={"/data": data_volume},
    cpu=2.0,
    memory=4096,
    max_containers=1,  # single-writer store — see module docstring
    scaledown_window=300,  # idle 5 min -> scale to zero -> $0
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root")
    from app import app as fastapi_app  # noqa: PLC0415

    return fastapi_app
