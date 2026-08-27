---
tags: [rag, evaluation, quality]
---

# Evaluating RAG Systems

The rule that orders everything else: **build the eval before tuning anything.** Once answers look fluent, you can no longer tell retrieval failure from generation failure — both produce confident, well-written text. Numbers first, vibes never.

## The three metrics

| Metric | Measures | Isolates |
|---|---|---|
| **recall@k** | expected source appears in top-k retrieved | retrieval only — no LLM needed |
| **groundedness** | every citation in the answer resolves to a retrieved chunk | fabrication |
| **refusal accuracy** | unanswerable questions get refused, not answered | parametric-memory leakage |

Recall@k is the ceiling: generation cannot exceed what retrieval delivers, so it gets measured first and alone.

## Building the eval set

- **20+ hand-written pairs**: question plus the note that must be retrieved. Write them *by reading the notes*, never by asking a model to invent questions — model-generated eval sets optimize for what the model already finds easy.
- **5 deliberately unanswerable questions** on topics absent from the vault. These measure the refusal path in [[hallucination-and-grounding]].
- Fix the set. It's a regression suite: same questions, same expected notes, before and after every change.

## Reading the results

Report the misses by *kind*, not just the count. "recall@5 = 0.75, all five misses are short-identifier queries" points at [[hybrid-search-bm25]]; a uniform miss pattern points at [[chunking-strategies]] or [[embedding-models]]. A number without a failure taxonomy is unactionable.

This is also the discipline behind [[evals-for-prompts]] — same habit, different artifact.
