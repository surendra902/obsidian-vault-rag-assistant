---
tags: [rag, retrieval]
---

# Chunking Strategies

Chunking decides what a "document" is for retrieval. Embedding models have a fixed maximum input length, so long notes must be split; how you split them decides whether the retrievable unit is a coherent idea or a fragment.

## The starting point

A common starting point is **chunks of 200–400 tokens with 10–20% overlap between neighbors**. The overlap keeps sentences that straddle a boundary retrievable from both sides.

## Rules that matter more than the numbers

1. **Split at natural boundaries first** — markdown headings, then paragraphs. A heading plus its paragraphs is a coherent unit; a fixed window that saws a paragraph in half is not.
2. **Respect the model's hard limit.** `all-MiniLM-L6-v2` truncates silently at 256 wordpiece tokens: no error, no warning — everything past the limit simply never gets embedded. Since [[embedding-models]] covers why I use that model, my chunks target ~180 words with 30 words of overlap, comfortably inside the limit, and the indexer asserts the token count so a violation crashes loudly instead of silently truncating.
3. **Carry metadata per chunk**: source path, heading, tags. Retrieval returns chunks, but users reason about notes. See [[citation-patterns]].
4. **Duplicate near-identical content** (a copied folder, a backup) poisons results: the same chunk comes back three times and crowds out genuinely different notes. Dedup by content hash at index time.

## What I rejected

Semantic chunking (embedding-based splitting) and hierarchical parent-child retrieval are real improvements at scale, but they add tuning surface I don't need for a personal vault. Plain heading-aware windows with overlap are boring and they work.
