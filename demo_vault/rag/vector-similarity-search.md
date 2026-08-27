---
tags: [rag, embeddings, retrieval]
---

# Vector Similarity Search

Given a query vector and a matrix of document vectors, retrieval is finding the rows with the highest similarity. The similarity function and the search structure are two separate decisions.

## Similarity functions

- **Cosine similarity** — the angle between vectors, ignoring magnitude. The default choice for text.
- **Dot product** — magnitude-sensitive. But if vectors are **L2-normalized** (unit length), dot product and cosine are *identical*. Normalizing at encode time means the query-time operation is one matrix–vector multiplication, no per-row division.
- **Euclidean distance** — on normalized vectors it's monotonically related to cosine, so ranking is the same.

So: normalize once at index time, then score with a dot product. That's the whole trick.

## Search structures

| Scale | Structure | Trade |
|---|---|---|
| < ~100k vectors | brute force (numpy matmul) | exact, sub-millisecond, zero libraries |
| millions | ANN index (HNSW, IVF) | approximate — trades recall for speed |

My vault is ~2,800 chunks × 384 dims ≈ **4.3 MB** of float32. Brute-force over that is a single numpy matmul costing well under a millisecond. Approximate indexes (FAISS, HNSW) would add dependencies and recall loss to solve a problem the data doesn't have. The threshold question — when retrieval should refuse — is covered in [[hallucination-and-grounding]]; the exact-vs-keyword weakness of dense vectors is covered in [[hybrid-search-bm25]].
