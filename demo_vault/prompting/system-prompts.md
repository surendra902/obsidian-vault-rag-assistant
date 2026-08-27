---
tags: [prompting, llm]
---

# System Prompts

The system prompt is the model's standing orders — the part that never changes between requests. Its job is scope and behavior, not information.

## What belongs there

- Identity and scope: what to answer, what to refuse.
- Output contract: citation format, refusal wording.
- Standing rules the user shouldn't have to repeat.

## What doesn't

Facts about the world, or the documents. Retrieved context goes in the message, not the system prompt — mixing them means either rebuilding the cache on every retrieval or lying about what's stable.

## Caching: why the system prompt goes first

Prompt caching is a **prefix match**: the cache holds the request up to the last cache breakpoint, and any byte that changes invalidates everything after it. Render order is tools → system → messages, so a frozen system prompt with a `cache_control` breakpoint gives you a cached prefix that hits on every request while the volatile question changes after it. Two failure modes to check for when the cache never hits:

1. A timestamp, UUID, or `datetime.now()` anywhere in the "stable" prefix.
2. Content that *serializes* unstably — an unsorted dict of tools, a set iterated in hash order.

The API reports `usage.cache_read_input_tokens`; if it stays zero across repeated requests, one of the above is happening. Cache hits cost roughly a tenth of fresh input tokens, so on a RAG workload with a fat system prompt this is most of the bill — see [[system-prompts|the pricing notes]] in the API docs. Structure follows [[prompt-structure]]: role and constraints stable, question volatile.
