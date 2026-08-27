---
tags: [rag, llm]
---

# What is RAG

Retrieval-Augmented Generation: before the model answers, a retrieval step fetches relevant documents and puts them in the prompt. The model answers **from the provided context** instead of from its training memory. Two phases, always in this order:

1. **Indexing** (offline): documents → chunks → embeddings → vector store. See [[chunking-strategies]] and [[embedding-models]].
2. **Retrieval + generation** (per query): embed the question, find the nearest chunks, hand them to the LLM with instructions to cite and to refuse when the context is insufficient.

## Why not just fine-tune or extend the context window?

- **Freshness**: a context window can't hold a vault, and training data goes stale. Retrieval fetches what exists *now*.
- **Grounding**: with retrieved text in the prompt, answers can cite sources, which makes errors visible instead of plausible. See [[hallucination-and-grounding]].
- **Cost and control**: swapping the index is cheap; swapping a fine-tune is not. Deleting a document from the index removes it from answers immediately.

## The failure people underestimate

RAG moves the hard problem from generation to retrieval. If the right chunk isn't in the top-k, the model writes a fluent answer around the wrong context. That's why [[evaluating-rag-systems]] starts at retrieval quality, and why the retrieval layer gets a similarity threshold that refuses rather than retrieves garbage.
