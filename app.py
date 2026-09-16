"""FastAPI server for the vault RAG assistant.

Endpoints:
    GET  /        static chat UI (static/index.html)
    GET  /healthz index stats + provider/model reachability
    POST /ask     retrieve top-5 -> LLM, citations resolved back to chunks
    POST /agent   tool loop: search_vault / read_note / list_by_tag

Providers (first configured wins):
    MY_ANTHROPIC_KEY -> Claude, native document citations (strongest grounding:
                        the API returns citation objects, so a cited source
                        cannot be invented).
    LLM_API_KEY      -> any OpenAI-compatible endpoint (OpenRouter, NVIDIA
                        Build, vLLM, ...) via LLM_BASE_URL + LLM_MODELS.
    neither          -> extractive mode: top chunks verbatim, clearly labelled,
                        so the pipeline is demonstrable with no key at all.
                        /agent needs a model to drive the tools -> 503.

Why MY_ANTHROPIC_KEY rather than ANTHROPIC_API_KEY: an ambient ANTHROPIC_API_KEY
or ANTHROPIC_BASE_URL must never silently reroute this app through a third-party
proxy. The Anthropic client also pins base_url explicitly.

OpenAI-compatible endpoints have no native citation feature, so /ask numbers the
retrieved chunks in the prompt, asks for [n] markers, and resolves them back to
hits. An out-of-range marker is dropped: a source is never fabricated.
"""

import json
import os
import re

import anthropic
import openai
from typing import Dict, List, Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rag import REFUSE_THRESHOLD, VaultIndex
from search import HybridIndex
from route import AdaptiveRouter
from events import EventLogger
from frontier import FrontierManager
from memory import MemoryManager


def _load_dotenv(path=".env"):
    """Read .env if present. Six lines of stdlib instead of python-dotenv;
    setdefault so an explicitly exported variable still wins."""
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MY_ANTHROPIC_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
LLM_KEY = os.environ.get("LLM_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").lower().strip()
LLM_MODELS = [
    m.strip()
    for m in os.environ.get(
        "LLM_MODELS",
        "nex-agi/nex-n2.5-pro:free,nvidia/nemotron-3-super-120b-a12b:free,nvidia/nemotron-3.5-lightning:free,google/gemma-4-31b-it:free",
    ).split(",")
    if m.strip()
]

client = oai = None
if LLM_PROVIDER in ("openrouter", "openai-compatible") and LLM_KEY:
    oai = openai.OpenAI(api_key=LLM_KEY, base_url=LLM_BASE_URL)
    PROVIDER, MODEL = "openai-compatible", LLM_MODELS[0]
elif ANTHROPIC_KEY and LLM_PROVIDER != "openrouter":
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY, base_url="https://api.anthropic.com")
    PROVIDER, MODEL = "anthropic", ANTHROPIC_MODEL
elif LLM_KEY:
    oai = openai.OpenAI(api_key=LLM_KEY, base_url=LLM_BASE_URL)
    PROVIDER, MODEL = "openai-compatible", LLM_MODELS[0]
else:
    PROVIDER, MODEL = "none", None

app = FastAPI(title="Vault RAG Assistant")

try:
    index = HybridIndex("vectors")
    router = AdaptiveRouter(index)
    event_logger = EventLogger("vectors/events.db")
    frontier_mgr = FrontierManager("vectors/frontier.db")
    memory_mgr = MemoryManager("vectors", "demo_vault")
except Exception as e:  # no index built yet -- serve healthz, refuse the rest
    index = None
    router = None
    event_logger = None
    frontier_mgr = None
    memory_mgr = None
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

# Appended only on the OpenAI-compatible path. Without NO_REASONING the free
# reasoning models dump their chain-of-thought into `content` (measured on
# nemotron and liquid); `reasoning: {"exclude": true}` alone does not stop it.
CITE_RULE = "\n- Cite each claim with its bracketed source number, e.g. [1]."
NO_REASONING = ("\n\nOutput ONLY the final answer. Do not output your reasoning, "
                "analysis, or planning steps.")

RETRY_STATUS = {429, 500, 502, 503, 504}


class AskRequest(BaseModel):
    question: str


def _sources(hits):
    return [
        {"path": h.path, "heading": h.heading, "score": round(h.score, 3), "excerpt": h.text[:400]}
        for h in hits
    ]


def _unusable(msg, finish_reason):
    """Why a 200 response can still be garbage, both measured on 2026-09-02:

    - finish_reason "length": a reasoning model spent the budget thinking and
      the reply is cut off mid-thought. It reads like leaked chain-of-thought
      and its [n] markers point at sources it was only *considering*.
    - control-token artifacts: liquid/lfm-2.5 emits a literal
      "<|tool_call_start|>[read(path='/home/gibbon/synthtraces/...')]" -- paths
      out of its training data -- when handed documents inline. Fine when it is
      actually driving tools, unusable as an answer.
    """
    text = (msg.content or "").strip()
    if not text and not msg.tool_calls:
        return "empty response"
    if finish_reason == "length":
        return "truncated (finish_reason=length)"
    if "<|" in text:
        return "control-token artifact in output"
    return None


def _chat(messages, tools=None, max_tokens=4000):
    """Call the OpenAI-compatible endpoint, walking LLM_MODELS as a fallback
    chain. Free tiers 429 unpredictably and free models fail in the ways
    _unusable describes, so the chain is what makes them dependable at all."""
    last = "no models configured"
    for model in LLM_MODELS:
        kwargs = {"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        try:
            choice = oai.chat.completions.create(**kwargs).choices[0]
        except openai.APIStatusError as e:
            if e.status_code in RETRY_STATUS:
                last = f"{model}: HTTP {e.status_code}"
                continue
            raise
        bad = _unusable(choice.message, choice.finish_reason)
        if bad is None:
            return choice.message, model
        last = f"{model}: {bad}"
    raise RuntimeError(f"all models failed -- last: {last}")


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
        model=ANTHROPIC_MODEL,
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
    return answer, sources, ANTHROPIC_MODEL


def _openai_answer(question: str, hits):
    """No native citations here, so number the chunks and resolve [n] markers
    back to them. An out-of-range n is dropped: a hallucinated [9] yields no
    source rather than a wrong one."""
    numbered = "\n\n".join(
        f"[{i}] {h.path}{' :: ' + h.heading if h.heading else ''}\n{h.text}"
        for i, h in enumerate(hits, 1)
    )
    msg, model = _chat([
        {"role": "system", "content": ASK_SYSTEM + CITE_RULE + NO_REASONING},
        {"role": "user", "content": f"{numbered}\n\nQuestion: {question}"},
    ])
    answer = (msg.content or "").strip()
    sources, seen = [], set()
    # Fullwidth brackets are not decoration: NVIDIA Build's openai/gpt-oss-120b
    # emits U+3010/U+3011 ("【1】") instead of ASCII, so an ASCII-only
    # pattern silently resolves every citation to nothing.
    for n in re.findall(r"[\[【](\d+)[\]】]", answer):
        i = int(n) - 1
        if 0 <= i < len(hits) and i not in seen:
            seen.add(i)
            h = hits[i]
            sources.append({"path": h.path, "heading": h.heading, "score": round(h.score, 3),
                            "excerpt": h.text[:400], "cited": True})
    return answer, sources, model


class FeedbackRequest(BaseModel):
    query_id: str
    feedback: int  # +1 for up, -1 for down


class RememberRequest(BaseModel):
    question: str
    answer: str
    citations: List[str]


@app.post("/remember")
def remember(req: RememberRequest):
    if memory_mgr is None:
        return JSONResponse(status_code=503, content={"detail": "memory manager not initialized"})
    rec = memory_mgr.save_derived_answer(req.question, req.answer, req.citations)
    if rec is None:
        return JSONResponse(status_code=400, content={"detail": "Rejected by sole citation guard: must cite primary source notes"})
    return {"status": "saved", "path": rec["path"]}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if event_logger:
        ok = event_logger.record_feedback(req.query_id, req.feedback)
        if ok and req.feedback == 1 and memory_mgr:
            # Check if query had cited sources to write back verified memory
            q_info = event_logger.get_query_event(req.query_id)
            if q_info and not q_info.get("refused") and q_info.get("response_text"):
                cited = [d["path"] for d in q_info.get("displayed", []) if d.get("cited")]
                if cited:
                    memory_mgr.save_derived_answer(q_info["query"], q_info["response_text"], cited)
        return {"status": "recorded" if ok else "not_found"}
    return {"status": "no_logger"}


@app.post("/ask")
def ask(req: AskRequest):
    if index is None:
        return JSONResponse(status_code=503, content={"detail": f"index not built: {_index_error}"})

    # Adaptive routing
    if router:
        hits, strategy_id = router.route_and_search(req.question, k=getattr(index, "default_k", 5))
    else:
        hits = index.search(req.question, k=5)
        strategy_id = "standard_hybrid"

    refuse_thresh = getattr(index, "threshold", REFUSE_THRESHOLD)
    top_conf = max((getattr(h, "dense_score", h.score) for h in hits), default=0.0) if hits else 0.0
    is_exact = strategy_id == "exact_bm25_heavy" and hits and getattr(hits[0], "bm25_score", 0.0) > 0.0

    if not hits or (top_conf < refuse_thresh and not is_exact):
        if frontier_mgr:
            frontier_mgr.enqueue_gap(req.question)
        qid = event_logger.log_query_event(req.question, hits, strategy_id=strategy_id, refused=True) if event_logger else None
        return {"answer": "I couldn't find anything about that in this vault.",
                "sources": [], "mode": "refused", "strategy": strategy_id, "query_id": qid}

    if PROVIDER == "none":
        excerpts = "\n\n---\n\n".join(
            f"[{h.path}{' :: ' + h.heading if h.heading else ''}]\n{h.text}" for h in hits[:3]
        )
        answer = ("(extractive mode: set MY_ANTHROPIC_KEY or LLM_API_KEY for "
                  "generated answers)\n\n" + excerpts)
        sources = _sources(hits)
        qid = event_logger.log_query_event(req.question, hits, strategy_id=strategy_id, refused=False, response_text=answer) if event_logger else None
        return {
            "answer": answer,
            "sources": sources,
            "mode": "extractive",
            "strategy": strategy_id,
            "query_id": qid,
        }

    mode = "live"
    try:
        if PROVIDER == "anthropic":
            answer, sources, model = _live_answer(req.question, hits)
        else:
            answer, sources, model = _openai_answer(req.question, hits)
    except (anthropic.AuthenticationError, openai.AuthenticationError):
        excerpts = "\n\n---\n\n".join(
            f"[{h.path}{' :: ' + h.heading if h.heading else ''}]\n{h.text}" for h in hits[:3]
        )
        answer = ("(extractive fallback mode: API key rejected or invalid)\n\n" + excerpts)
        sources = _sources(hits)
        model = "local-extractive"
        mode = "extractive"
    except (anthropic.APIStatusError, openai.APIStatusError) as e:
        excerpts = "\n\n---\n\n".join(
            f"[{h.path}{' :: ' + h.heading if h.heading else ''}]\n{h.text}" for h in hits[:3]
        )
        answer = (f"(fallback mode: remote API error {e.status_code})\n\n" + excerpts)
        sources = _sources(hits)
        model = "local-extractive"
        mode = "extractive"
    except RuntimeError as e:  # whole fallback chain exhausted -> graceful extractive fallback
        excerpts = "\n\n---\n\n".join(
            f"[{h.path}{' :: ' + h.heading if h.heading else ''}]\n{h.text}" for h in hits[:3]
        )
        answer = (f"(fallback mode: remote LLM chain exhausted - {e})\n\n" + excerpts)
        sources = _sources(hits)
        model = "local-extractive"
        mode = "extractive"

    qid = event_logger.log_query_event(
        req.question,
        hits,
        strategy_id=strategy_id,
        refused=False,
        response_text=answer,
        cited_paths=[s["path"] for s in (sources or [])]
    ) if event_logger else None

    return {
        "answer": answer,
        "sources": sources or _sources(hits),
        "mode": mode,
        "model": model,
        "strategy": strategy_id,
        "query_id": qid,
    }


def _anthropic_agent(question, tools, run_named):
    messages = [{"role": "user", "content": question}]
    for _ in range(8):  # bound the loop: two-hop questions need 2-4 calls
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": AGENT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text"), ANTHROPIC_MODEL
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                try:
                    content, is_error = run_named(b.name, b.input), False
                except Exception as e:  # tool errors go back as results, not crashes
                    content, is_error = f"error: {e}", True
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": content, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    return "stopped: tool loop exceeded 8 iterations", ANTHROPIC_MODEL


def _openai_agent(question, tools, run_named):
    """The same loop in OpenAI function-calling shape: tool_calls on the
    assistant message, one role="tool" message per call keyed by tool_call_id."""
    oai_tools = [{"type": "function",
                  "function": {"name": t["name"], "description": t["description"],
                               "parameters": t["input_schema"]}} for t in tools]
    messages = [{"role": "system", "content": AGENT_SYSTEM + NO_REASONING},
                {"role": "user", "content": question}]
    model = LLM_MODELS[0]
    for _ in range(8):
        msg, model = _chat(messages, tools=oai_tools)
        if not msg.tool_calls:
            return (msg.content or "").strip(), model
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                out = run_named(tc.function.name, json.loads(tc.function.arguments or "{}"))
            except Exception as e:
                out = f"error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "stopped: tool loop exceeded 8 iterations", model


@app.post("/agent")
def agent(req: AskRequest):
    if index is None:
        return JSONResponse(status_code=503, content={"detail": f"index not built: {_index_error}"})
    if PROVIDER == "none":
        return JSONResponse(status_code=503,
                            content={"detail": "agent mode requires MY_ANTHROPIC_KEY or "
                                               "LLM_API_KEY (a model drives the tools)"})

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
        return index.read_note(path)  # raises ValueError on path escape -> error tool result

    def list_by_tag(tag: str) -> str:
        """List note paths carrying a frontmatter tag."""
        notes = index.notes_by_tag(tag)
        tool_trace.append(f"list_by_tag({tag!r}) -> {len(notes)} notes")
        return json.dumps(notes)

    FUNCS = {"search_vault": search_vault, "read_note": read_note, "list_by_tag": list_by_tag}

    def run_named(name, args):
        """Single execution point for both provider loops, so the path-escape
        guard in read_note covers the OpenAI path too."""
        fn = FUNCS.get(name)
        if fn is None:
            raise ValueError(f"unknown tool {name!r}")
        return fn(**args)

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

    try:
        if PROVIDER == "openai-compatible":
            answer, model = _openai_agent(req.question, tools, run_named)
        else:
            answer, model = _anthropic_agent(req.question, tools, run_named)
    except (anthropic.AuthenticationError, openai.AuthenticationError):
        return JSONResponse(status_code=503, content={"detail": f"{PROVIDER} API rejected the key"})
    except (anthropic.APIStatusError, openai.APIStatusError) as e:
        return JSONResponse(status_code=502, content={"detail": f"{PROVIDER} API error {e.status_code}"})
    except RuntimeError as e:
        return JSONResponse(status_code=502, content={"detail": str(e)})
    return {"answer": answer, "tool_trace": tool_trace, "mode": "agent", "model": model}


@app.get("/healthz")
def healthz():
    out = {"status": "ok",
           "mode": "extractive" if PROVIDER == "none" else "live",
           "provider": PROVIDER,
           "model": MODEL,
           "index": index.stats if index else None}
    if index is None:
        out["status"] = "degraded"
        out["index_error"] = _index_error
    # Cheapest portable auth probe: catch a dead key at health-check time, not
    # demo time. max_tokens=1 is enough -- a 200 means the key was accepted.
    if PROVIDER == "anthropic":
        try:
            client.messages.create(model=ANTHROPIC_MODEL, max_tokens=8,
                                   messages=[{"role": "user", "content": "ping"}])
            out["api"] = "ok"
        except anthropic.APIError as e:
            out["api"] = f"error: {type(e).__name__}"
    elif PROVIDER == "openai-compatible":
        out["models"] = LLM_MODELS
        try:
            oai.chat.completions.create(model=LLM_MODELS[0], max_tokens=1,
                                        messages=[{"role": "user", "content": "ping"}])
            out["api"] = "ok"
        except openai.APIError as e:
            out["api"] = f"error: {type(e).__name__}"
    return out


@app.get("/")
def root():
    return FileResponse("static/index.html")
