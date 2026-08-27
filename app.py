"""FastAPI server for the vault RAG assistant.

Endpoints:
    GET  /        static chat UI (static/index.html)
    GET  /healthz index stats + API reachability
    POST /ask     retrieve top-5 -> Claude with native document citations
    POST /agent   tool loop: search_vault / read_note / list_by_tag

Modes:
    MY_ANTHROPIC_KEY set  -> live mode (Claude generation, citations)
    unset                 -> extractive mode (top chunks verbatim, clearly
                             labelled) so the pipeline is demonstrable without
                             a key. Agent mode requires the key (a model drives
                             the tools) and returns 503.

The Anthropic client pins base_url explicitly and reads a *differently named*
env var (MY_ANTHROPIC_KEY): an ambient ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
must never silently reroute this app through a third-party proxy.
"""

import json
import os

import anthropic
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rag import REFUSE_THRESHOLD, VaultIndex

MODEL = "claude-opus-5"
API_KEY = os.environ.get("MY_ANTHROPIC_KEY")

client = None
if API_KEY:
    client = anthropic.Anthropic(api_key=API_KEY, base_url="https://api.anthropic.com")

app = FastAPI(title="Vault RAG Assistant")

try:
    index = VaultIndex("vectors")
except Exception as e:  # no index built yet -- serve healthz, refuse the rest
    index = None
    _index_error = str(e)

ASK_SYSTEM = """You answer questions about a personal Obsidian vault.
Rules:
- Answer ONLY from the provided document excerpts; do not use outside knowledge.
- Cite the excerpts you used.
- If the excerpts do not contain the answer, say the vault doesn't cover it.
- Be concise and specific."""

AGENT_SYSTEM = """You answer questions about a personal Obsidian vault using tools.
Strategy: search first; if a hit looks truncated or you need a neighboring
section, read the full note; if the question is about a category, list by tag.
Then answer ONLY from what the tools returned, citing note paths. If the tools
surface nothing relevant, say the vault doesn't cover it."""


class AskRequest(BaseModel):
    question: str


def _sources(hits):
    return [
        {"path": h.path, "heading": h.heading, "score": round(h.score, 3), "excerpt": h.text[:400]}
        for h in hits
    ]


def _live_answer(question: str, hits):
    """Claude call with native citations. Document blocks carry the chunks;
    the response's text blocks carry citations back to them."""
    docs = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.text},
            "title": f"{h.path}{' :: ' + h.heading if h.heading else ''}",
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": ASK_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": docs + [{"type": "text", "text": question}]}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    # Collect actually-cited documents (dedup by index) for the sources list.
    cited = {}
    for b in resp.content:
        for c in (getattr(b, "citations", None) or []):
            cited[c.get("document_index")] = c.get("document_title")
    sources = []
    for i, title in cited.items():
        if i is not None and 0 <= i < len(hits):
            h = hits[i]
            sources.append({"path": h.path, "heading": h.heading, "score": round(h.score, 3),
                            "excerpt": h.text[:400], "cited": True})
    return answer, sources


@app.post("/ask")
def ask(req: AskRequest):
    if index is None:
        return JSONResponse(status_code=503, content={"detail": f"index not built: {_index_error}"})
    hits = index.search(req.question, k=5)
    if not hits or hits[0].score < REFUSE_THRESHOLD:
        return {"answer": "I couldn't find anything about that in this vault.",
                "sources": [], "mode": "refused"}
    if client is None:
        excerpts = "\n\n---\n\n".join(
            f"[{h.path}{' :: ' + h.heading if h.heading else ''}]\n{h.text}" for h in hits[:3]
        )
        return {
            "answer": "(extractive mode: set MY_ANTHROPIC_KEY for generated answers)\n\n" + excerpts,
            "sources": _sources(hits),
            "mode": "extractive",
        }
    try:
        answer, sources = _live_answer(req.question, hits)
    except anthropic.AuthenticationError:
        return JSONResponse(status_code=503, content={"detail": "Anthropic API rejected MY_ANTHROPIC_KEY"})
    except anthropic.APIStatusError as e:
        return JSONResponse(status_code=502, content={"detail": f"Anthropic API error {e.status_code}"})
    return {"answer": answer, "sources": sources or _sources(hits), "mode": "live"}


@app.post("/agent")
def agent(req: AskRequest):
    if index is None:
        return JSONResponse(status_code=503, content={"detail": f"index not built: {_index_error}"})
    if client is None:
        return JSONResponse(status_code=503,
                            content={"detail": "agent mode requires MY_ANTHROPIC_KEY (a model drives the tools)"})

    tool_trace = []

    def search_vault(query: str, k: int = 5) -> str:
        """Semantic search over vault notes. Returns JSON: [{path, heading, score, excerpt}]."""
        hits = index.search(query, k=k)
        tool_trace.append(f"search_vault({query!r}, k={k}) -> {len(hits)} hits")
        return json.dumps([{"path": h.path, "heading": h.heading, "score": round(h.score, 3),
                            "excerpt": h.text[:400]} for h in hits])

    def read_note(path: str) -> str:
        """Read one full note by vault-relative path. Use when a search hit looks truncated."""
        tool_trace.append(f"read_note({path!r})")
        return index.read_note(path)  # raises ValueError on path escape -> is_error tool result

    def list_by_tag(tag: str) -> str:
        """List note paths carrying a frontmatter tag."""
        notes = index.notes_by_tag(tag)
        tool_trace.append(f"list_by_tag({tag!r}) -> {len(notes)} notes")
        return json.dumps(notes)

    tools = [
        {"name": "search_vault", "description": "Semantic search over vault notes. Returns JSON: [{path, heading, score, excerpt}].",
         "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                           "k": {"type": "integer", "default": 5}},
                          "required": ["query"], "additionalProperties": False}},
        {"name": "read_note", "description": "Read one full note by vault-relative path. Use when a search hit looks truncated.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                          "required": ["path"], "additionalProperties": False}},
        {"name": "list_by_tag", "description": "List note paths carrying a frontmatter tag.",
         "input_schema": {"type": "object", "properties": {"tag": {"type": "string"}},
                          "required": ["tag"], "additionalProperties": False}},
    ]

    def run_tool(block):
        try:
            fn = {"search_vault": search_vault, "read_note": read_note, "list_by_tag": list_by_tag}[block.name]
            return fn(**block.input), False
        except Exception as e:  # tool errors go back as is_error results, not crashes
            return f"error: {e}", True

    try:
        messages = [{"role": "user", "content": req.question}]
        for _ in range(8):  # bound the loop: two-hop questions need 2-4 calls
            resp = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": AGENT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
            )
            if resp.stop_reason != "tool_use":
                answer = "".join(b.text for b in resp.content if b.type == "text")
                return {"answer": answer, "tool_trace": tool_trace, "mode": "agent"}
            messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    content, is_error = run_tool(b)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        return {"answer": "stopped: tool loop exceeded 8 iterations", "tool_trace": tool_trace, "mode": "agent"}
    except anthropic.AuthenticationError:
        return JSONResponse(status_code=503, content={"detail": "Anthropic API rejected MY_ANTHROPIC_KEY"})
    except anthropic.APIStatusError as e:
        return JSONResponse(status_code=502, content={"detail": f"Anthropic API error {e.status_code}"})


@app.get("/healthz")
def healthz():
    out = {"status": "ok", "mode": "live" if client else "extractive", "index": index.stats if index else None}
    if index is None:
        out["status"] = "degraded"
        out["index_error"] = _index_error
    if client:  # 8-token probe: catch dead keys at health-check time, not demo time
        try:
            client.messages.create(model=MODEL, max_tokens=8, messages=[{"role": "user", "content": "ping"}])
            out["api"] = "ok"
        except anthropic.APIError as e:
            out["api"] = f"error: {type(e).__name__}"
    return out


@app.get("/")
def root():
    return FileResponse("static/index.html")
