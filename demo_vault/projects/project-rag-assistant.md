---
tags: [project, rag]
---

# Project: Vault RAG Assistant

**Status:** building. The [[project-rag-assistant|dogfooding project]] — a question-answering layer over this very vault.

## Goal

Ask questions in natural language; get answers grounded in my notes with citations I can open. Not a search box — search makes me read; this should make the vault answer.

## Architecture

```
vault/*.md
  → index.py: walk → dedup → chunk (180 words / 30 overlap) → embed → vectors.npz + chunks.jsonl
  → app.py (FastAPI):
      /ask    cosine top-5 → Claude with document-block citations → cited answer
      /agent  tool loop: search_vault / read_note / list_by_tag — model decides when to widen
      /healthz index stats + API reachability
```

## Key choices

- **Local `all-MiniLM-L6-v2` embeddings** — free, offline, no second vendor. See [[embedding-models]].
- **Numpy brute-force retrieval** — the whole vault is ~4 MB of vectors; an ANN index would be a dependency solving nothing. See [[vector-similarity-search]].
- **Retrieval threshold → refusal** — below-threshold queries never reach the model. See [[hallucination-and-grounding]].
- **Eval before tuning** — 20 hand-written pairs plus 5 unanswerable; recall@5 measured before any parameter got fiddled. See [[evaluating-rag-systems]].

## Open questions

1. Hybrid BM25 fusion for identifier-heavy queries — deferred until the eval demands it ([[hybrid-search-bm25]]).
2. Multi-note synthesis quality in agent mode — needs a two-hop eval question; current evalset is single-hop.
3. Whether the demo corpus needs a public-corpus expansion pack for reviewers without my context.

The build log with dated decisions lives in [[decision-log]].
