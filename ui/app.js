/* Production RAG UI. All calls go through same-origin /api/* (nginx proxy). */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------- service health chips ---------- */
async function refreshHealth() {
  const chips = document.querySelectorAll(".chip[data-svc]");
  await Promise.all(
    [...chips].map(async (chip) => {
      try {
        const res = await fetch(`/api/health/${chip.dataset.svc}`, { signal: AbortSignal.timeout(4000) });
        chip.classList.toggle("up", res.ok);
        chip.classList.toggle("down", !res.ok);
      } catch {
        chip.classList.remove("up");
        chip.classList.add("down");
      }
    })
  );
}
refreshHealth();
setInterval(refreshHealth, 15000);

/* ---------- ingestion ---------- */
const uploadForm = $("upload-form");
const jobStatus = $("job-status");
const jobSpinner = $("job-spinner");
const jobText = $("job-text");
const jobMeta = $("job-meta");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file-input").files[0];
  if (!file) return;

  const body = new FormData();
  body.append("file", file);
  body.append("source_domain", $("domain-input").value.trim());
  const title = $("title-input").value.trim();
  if (title) body.append("title", title);

  $("upload-btn").disabled = true;
  jobStatus.hidden = false;
  jobSpinner.className = "spinner";
  jobText.textContent = `Uploading ${file.name}…`;
  jobMeta.textContent = "";

  try {
    const res = await fetch("/api/ingest", { method: "POST", body });
    if (!res.ok) throw new Error(`upload failed: HTTP ${res.status} ${await res.text()}`);
    const { job_id } = await res.json();
    jobText.textContent = `Ingesting… (job ${job_id.slice(0, 8)})`;
    await pollJob(job_id);
  } catch (err) {
    jobSpinner.className = "spinner failed";
    jobText.textContent = "Ingestion failed";
    jobMeta.textContent = String(err);
  } finally {
    $("upload-btn").disabled = false;
  }
});

async function pollJob(jobId) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 4000));
    const res = await fetch(`/api/ingest/${jobId}`);
    if (!res.ok) continue; // transient; keep polling
    const job = await res.json();
    if (job.status === "succeeded") {
      const r = job.result || {};
      jobSpinner.className = "spinner done";
      jobText.textContent = "Ingested ✓";
      jobMeta.textContent =
        `${r.page_count} pages → ${r.parent_count} sections → ${r.chunk_count} chunks\n` +
        `took ${Math.round(r.duration_seconds)}s — ready to query`;
      return;
    }
    if (job.status === "failed") {
      jobSpinner.className = "spinner failed";
      jobText.textContent = "Ingestion failed";
      jobMeta.textContent = job.error || "unknown error";
      return;
    }
    jobText.textContent = `Ingesting… (${job.status})`;
  }
}

/* ---------- chat ---------- */
const messages = $("messages");
const chatForm = $("chat-form");
const chatInput = $("chat-input");

function addMessage(cls) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function textEl(tag, text) {
  const el = document.createElement(tag);
  el.textContent = text;
  return el;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  chatInput.value = "";

  addMessage("user").appendChild(textEl("p", query));
  const pending = addMessage("assistant");
  pending.appendChild(textEl("p", "Thinking…"));
  $("send-btn").disabled = true;

  const domain = $("query-domain").value.trim();
  const payload = {
    query,
    top_n: Math.max(1, Math.min(20, Number($("top-n").value) || 5)),
  };
  if (domain) payload.source_domains = [domain];

  const started = performance.now();
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    data.query = query; // not echoed by the API — attach for feedback context
    renderAnswer(pending, data, performance.now() - started);
  } catch (err) {
    pending.className = "msg error";
    pending.replaceChildren(textEl("p", `Request failed — ${err}`));
  } finally {
    $("send-btn").disabled = false;
    chatInput.focus();
  }
});

function renderAnswer(el, data, elapsedMs) {
  el.replaceChildren();
  el.appendChild(textEl("p", data.answer));

  if (Array.isArray(data.citations) && data.citations.length) {
    const list = document.createElement("div");
    list.className = "citations";
    for (const c of data.citations) {
      const item = document.createElement("div");
      item.className = "citation";
      const page = c.page_number != null ? `p. ${c.page_number}` : "page n/a";
      const title = c.title || c.document_id;
      item.appendChild(textEl("b", `${title} — ${page}`));
      item.appendChild(document.createTextNode(` (${c.parent_id})`));
      list.appendChild(item);
    }
    el.appendChild(list);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.appendChild(textEl("span", `${(elapsedMs / 1000).toFixed(1)}s`));
  meta.appendChild(textEl("span", data.model || ""));
  if (data.guardrail_flagged) {
    const badge = textEl("span", "guardrail flagged");
    badge.className = "badge flagged";
    meta.appendChild(badge);
  }
  meta.appendChild(buildFeedbackControls(data));
  el.appendChild(meta);
  messages.scrollTop = messages.scrollHeight;
}

/* ---------- answer feedback (ADR-027) ---------- */
function buildFeedbackControls(data) {
  const wrap = document.createElement("span");
  wrap.className = "feedback";

  const up = document.createElement("button");
  up.type = "button";
  up.className = "feedback-btn";
  up.title = "Good answer";
  up.textContent = "👍";

  const down = document.createElement("button");
  down.type = "button";
  down.className = "feedback-btn";
  down.title = "Bad answer";
  down.textContent = "👎";

  const sendFeedback = async (rating, btn) => {
    up.disabled = true;
    down.disabled = true;
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: data.request_id,
          rating,
          query: data.query,
          answer: data.answer,
        }),
      });
      btn.classList.add("sent");
      wrap.appendChild(textEl("span", "Thanks for the feedback"));
    } catch {
      // Best-effort: feedback failing to send must never disrupt the chat.
      up.disabled = false;
      down.disabled = false;
    }
  };

  up.addEventListener("click", () => sendFeedback("up", up));
  down.addEventListener("click", () => sendFeedback("down", down));

  wrap.appendChild(up);
  wrap.appendChild(down);
  return wrap;
}
