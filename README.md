# Obsidian Vault RAG Knowledge Assistant

Ask questions in natural language; get answers grounded in an Obsidian vault,
with citations you can open and verify. Local embeddings, a numpy vector store,
Claude for generation, one tool-use loop for multi-hop questions.

## Quickstart (5 commands)

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows (source .venv/bin/activate on POSIX)
pip install -r requirements.txt
python index.py --vault ./demo_vault               # ~15s: 27 notes -> 108 chunks
set MY_ANTHROPIC_KEY=sk-ant-...                    # optional; skip for extractive mode
python -m uvicorn app:app                          # http://127.0.0.1:8000
```

First model load takes ~1 minute (downloads/loads `all-MiniLM-L6-v2`, then cached).
Point `--vault` at any Obsidian vault — the indexer is safe against hostile
directory shapes (see *Indexer safety* below).

## Architecture

```
vault/*.md
  -> index.py: walk (deny-list) -> dedup (content hash) -> chunk (heading-aware,
                180 words / 30 overlap, token-budget splitter) -> MiniLM embed
                -> vectors.npz + chunks.jsonl + meta.json
  -> app.py (FastAPI):
      POST /ask     cosine top-5 (numpy matmul) -> Claude, document blocks with
                    native citations -> cited answer
                    top score < 0.27 -> "not in this vault" (no LLM call)
      POST /agent   tool loop (max 8 turns): search_vault / read_note / list_by_tag
      GET  /healthz index stats + live API probe
      GET  /        chat UI (single static/index.html)
```

Two modes: **live** (`MY_ANTHROPIC_KEY` set — Claude generates with citations)
and **extractive** (no key — top chunks returned verbatim, clearly labelled;
`/agent` requires a key because a model drives the tools).

## Evaluation (the number this repo is built around)

`evalset.jsonl` holds 25 hand-written questions written *by reading the notes*
(never model-generated) plus 5 deliberately unanswerable ones.
`python eval.py` measures retrieval — no API key needed:

```
recall@5: 0.960 (24/25)
refusal accuracy @threshold 0.27: 1.000 (5/5)
false-refusal on answerable @threshold 0.27: 0.000 (0/25)
answerable top-score distribution:   min=0.308 median=0.551 max=0.790
unanswerable top-score distribution: min=0.072 median=0.111 max=0.199
```

The refusal threshold (0.27) sits inside the measured gap between unanswerable
(max 0.199) and answerable (min 0.308) top-hit scores.

The one recall miss is a "which decisions were made" query — a decisions-log
note whose *body* doesn't semantically resemble the question. The known fix is
hybrid BM25 + dense fusion with reciprocal rank fusion; deliberately not built
because one miss doesn't justify the dependency. `eval.py` exits non-zero below
recall 0.85 / refusal 0.80, so a regression can't slip through silently.

An earlier body-only embedding scored **recall@5 = 0.840**; prepending the note
title + heading to each chunk's embedded text lifted it to 0.960. That change
was made *because* the eval demanded it — the harness exists to order decisions
like that, not to decorate a README.

Groundedness (every citation resolves to a retrieved chunk) is enforced by
construction in live mode: sources are extracted from the API's citation
objects, which only reference the document blocks sent in the request.

## Design decisions

| Decision | Reason |
|---|---|
| Local MiniLM embeddings, not a hosted API | Anthropic has no embeddings endpoint; its partner (Voyage AI) is a separate account and bill. MiniLM is free, offline, and 384-dim vectors are sufficient at vault scale. |
| Numpy array, not a vector DB | 108 chunks × 384 dims ≈ 166 KB here; even 3,400 chunks (the full real vault this was tested against) is 5 MB. Brute-force cosine is one exact, sub-millisecond matmul. FAISS/Chroma solve a scale problem this doesn't have. |
| Native citations, not structured JSON | The API rejects `citations` + `output_config.format` together. Grounding is the product; the UI reads text. |
| `MY_ANTHROPIC_KEY`, base_url pinned | An ambient `ANTHROPIC_BASE_URL` can silently reroute `anthropic.Anthropic()` through a third-party proxy. A distinct env var name plus explicit `base_url="https://api.anthropic.com"` makes that impossible. |
| Refusal threshold at retrieval, not in the model | Below-threshold queries never reach the LLM. Measurable (evalset distributions above), unlike "the model felt the context was insufficient." |
| One tool loop, three tools | search / read / list-by-tag covers two-hop questions. More agents would be orchestration without evidence. |
| thinking: adaptive, no budget_tokens | `budget_tokens` is rejected with 400 on current models. |

## Indexer safety (verified against a messy real vault)

- **Dot-directories are skipped** — tested against a vault whose `find` reports
  858 `.md` files where 680 of them live in `.config-primary`/`.config-secondary`
  plugin caches: zero leaked into the index.
- **Content-hash dedup** — 7 exact duplicate files in that vault were skipped.
  Near-duplicates (a "Copy" folder) are *not* collapsed; exact-hash only.
- **Token budget enforced loudly** — MiniLM silently truncates input past 256
  wordpiece tokens. Chunks are split until `prefix + body` fits, and a final
  assert crashes the indexer rather than truncating. 180 words of prose is
  ~240 tokens, but 180 words of code can exceed 256 — the splitter handles both.
- **Offline-first model load** — `sentence-transformers` HEADs huggingface.co on
  every load even when cached, and dies without network. When the model is
  cached, `HF_HUB_OFFLINE=1` is set before import; a fresh install still
  downloads.

## Tool-call security

`read_note(path)` is model-driven file access — a trust boundary. Paths are
resolved and rejected unless they stay inside the indexed vault
(`../../` escapes, absolute paths, and non-note files all fail loudly).
Verified: five traversal attempts, all blocked.

## Limitations

- Dense retrieval only. Exact-identifier queries ("r45", "nc.py") underperform;
  hybrid BM25 fusion is the known, documented next step.
- `/agent`'s loop is bounded at 8 turns; genuinely multi-hop research questions
  can exhaust it.
- Extractive mode is a fallback for demos, not a product: it returns verbatim
  chunks, it does not answer.
- The eval set is 30 questions on a 27-note demo corpus — enough to order
  engineering decisions, not a benchmark.
- Live-mode answer quality (as opposed to retrieval) is unmeasured here: every
  API credential available at build time was dead (401), so generation was
  verified to the request-construction level (params serialize, the API
  responds, auth errors are handled cleanly) but no generated answer was
  produced. The first session with a valid key should run the evalset through
  `/ask` and record groundedness.

## Screenshots

*(add: chat UI in live mode with citations expanded; agent mode tool trace)*

## Files

```
index.py       indexer: walk / dedup / chunk / embed -> vectors/
rag.py         VaultIndex: load, cosine search, guarded reads, tag lookup, model loader
app.py         FastAPI: /ask /agent /healthz / (+ static)
eval.py        recall@5, refusal accuracy, false-refusal rate, score distributions
evalset.jsonl  25 answerable + 5 unanswerable questions
demo_vault/    sanitized 27-note Obsidian corpus (frontmatter, tags, wikilinks)
static/        the entire UI, one HTML file
```
