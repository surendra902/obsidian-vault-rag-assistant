---
tags: [rag, quality, ux]
---

# Citation Patterns

A citation is only as good as its round trip: the user must be able to click the source and see the sentence the claim came from. Anything short of that is decoration.

## Anthropic native citations

The Claude Messages API supports citations on **document blocks**: send each retrieved chunk as `{"type": "document", "source": {"type": "text", ...}, "title": ..., "citations": {"enabled": true}}`, and the response's text blocks arrive split at citation boundaries, each carrying `cited_text` plus a pointer to `document_index` and `document_title`. The mapping from citation back to note path falls out of the block order.

Two constraints worth remembering:

- Citations and structured JSON output are **mutually exclusive** in one request — pick one. I pick citations; grounding is the product, and the UI doesn't need JSON.
- Set `citations` on **all** document blocks or none — the API rejects mixed configurations.

## The fallback pattern

Without native support, ask the model to emit `[^n]` markers, then **validate every marker against the retrieved set** and drop unresolvable ones. Validation is the part that matters: an unresolvable citation is worse than none, because it teaches the user to stop clicking.

## Display

Under each answer, list sources as `path :: heading` with the similarity score, expandable to the exact chunk text. The score is honest UI: it shows the user how confident retrieval was, which is exactly the signal [[hallucination-and-grounding]] says should be visible.
