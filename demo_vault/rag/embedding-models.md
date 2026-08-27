---
tags: [rag, embeddings]
---

# Embedding Models

An embedding model maps text to a fixed-length vector such that similar meanings land near each other. It is the retrieval half of [[what-is-rag]].

## The model I use and why

**`all-MiniLM-L6-v2`** — a small sentence-transformers model:

- **384-dimensional** output vectors.
- **256 wordpiece tokens** maximum input; longer text is silently truncated (the trap [[chunking-strategies]] is built around).
- Runs locally on CPU — about 0.7 seconds per thousand short chunks on my machine — with no API key, no per-token cost, and no network dependency.

For a personal vault of a few thousand chunks, its retrieval quality is within shouting distance of the big hosted models. The bottleneck in my answers is retrieval discipline, not embedding quality.

## The hosted tier

Anthropic's API does not offer embeddings at all — their recommended partner is **Voyage AI**, a separate company with its own console and key. OpenAI's `text-embedding-3-small` and Cohere's embed family are the other usual suspects. These produce 1024–3072 dimensional vectors and generally beat MiniLM on benchmarks, at the cost of a second vendor, a second key, and a per-token bill. Upgrade when a measured [[evaluating-rag-systems|eval]] says retrieval is the bottleneck — not before.

## Practical notes

- Always **normalize** embeddings (L2) at encode time; then cosine similarity is a plain dot product — see [[vector-similarity-search]].
- Embed queries and documents with the *same* model. Mixing models produces vectors in incompatible spaces, and nothing errors — scores just drift toward noise.
