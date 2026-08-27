---
tags: [project, decisions]
---

# Decision Log

Short-form ADRs for [[project-rag-assistant]]. Newest last. Format: decision, reason, rejected alternative.

## 2026-08-25 — Local MiniLM embeddings, not a hosted API

**Reason:** free, offline, no second vendor/key; retrieval quality at vault scale is not the bottleneck. **Rejected:** Voyage AI (recommended by Anthropic but a separate account and bill) and OpenAI embeddings — both upgrade paths, not starting points. Details in [[embedding-models]].

## 2026-08-25 — Numpy array, not a vector database

**Reason:** ~2,800 chunks × 384 dims ≈ 4.3 MB; one matmul is exact and sub-millisecond. **Rejected:** Chroma, FAISS, LanceDB — dependencies and approximate recall to solve a scale problem the vault doesn't have. Details in [[vector-similarity-search]].

## 2026-08-26 — Chunk at 180 words / 30 overlap, with a loud assert

**Reason:** MiniLM truncates silently at 256 tokens; 180 words fits with headroom, and the indexer asserts the token count so a violation crashes instead of truncating. **Rejected:** fixed 512-token windows (would silently lose the tail of every long note). Details in [[chunking-strategies]].

## 2026-08-26 — Native citations, not structured JSON output

**Reason:** the API rejects citations + output format together; grounding is the product, so citations win. The UI reads text either way. Details in [[citation-patterns]].

## 2026-08-26 — Index-time deny-list and content-hash dedup

**Reason:** `os.walk` over the real vault ingests two copies of a 340-file plugin cache and a duplicated folder; glob silently skips dot-directories and reports a third of the corpus. Both failure modes produce garbage retrieval with no error. **Rejected:** "just point it at the notes folder" — the indexer must be safe against whatever directory shape it meets.

## 2026-08-27 — Retrieval threshold set from the eval distribution

**Reason:** measured top-similarity for answerable vs. unanswerable questions separates cleanly; threshold sits in the gap. **Rejected:** asking the model to decide when context is insufficient — unmeasurable. Details in [[hallucination-and-grounding]].

## 2026-08-27 — One tool loop, three tools

**Reason:** search / read / list-by-tag covers the two-hop questions; more tools or agents is orchestration without evidence. **Rejected:** a multi-agent reviewer pass — revisit only if groundedness eval shows fabrication the citations don't catch.
