---
tags: [prompting]
---

# Prompt Structure

A prompt that survives contact with a real system has four parts, in this order: **role, context, task, constraints**.

1. **Role** — who the model is being: "You answer questions about my notes." Cheap, surprisingly effective at setting register and scope.
2. **Context** — the retrieved chunks, reference material, or data. The bulk of the tokens.
3. **Task** — the actual instruction, stated once, in imperative form.
4. **Constraints** — format, length, refusal rules, citation requirements.

## Why the order matters

The parts the model can't misinterpret go first; the instruction sits where attention is strongest at generation time. More practically: stable content (role, most context) **before** volatile content (the question) is what makes prompt caching work — see [[system-prompts]].

## Anti-patterns I keep seeing

- **Instruction soup**: six restatements of the same rule. Each restatement dilutes; the model weighs them all and obeys an average. One clear sentence.
- **Task buried mid-prompt**: "…and also can you summarize the above…". Tasks in the middle get dropped at rates that surprise people.
- **Format specified by example only**: the model pattern-matches the example's *content*, not just its shape. State the format, then optionally show one example — that's the boundary where [[few-shot-prompting]] starts.

Constraints last does not mean constraints least: the refusal rule in my RAG prompt is one sentence and does more work than everything above it. Verification of any prompt change goes through [[evals-for-prompts]], not vibes.
