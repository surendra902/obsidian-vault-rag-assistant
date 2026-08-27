---
tags: [prompting, evaluation]
---

# Evals for Prompts

A prompt is code. It gets edited, and every edit is a regression risk — the fix for one input quietly breaks three others. An eval is the only thing standing between "I improved the prompt" and "I changed the prompt."

## The minimum viable prompt eval

1. **A fixed set of inputs** — 20–50 real requests, saved as files, including the weird ones (empty input, hostile formatting, out-of-scope questions).
2. **A scorer** — exact-match for classification, substring/key-fact checks for generation, or a rubric if you must. Key-fact checks ("does the answer contain the threshold value and the reason?") are the sweet spot for open text.
3. **A run command** that prints pass/fail per case and a total, exit-code nonzero on regression.

## Rules

- **Write the failing case first.** When a prompt mishandles an input, that input goes into the set *before* you fix the prompt. Otherwise you'll re-break it in a month and never know.
- Version prompts alongside the eval that justifies them. A prompt diff without an eval run is unreviewable.
- Watch aggregate regressions: improving average score while breaking a refusal case is a net loss for a grounded system — weight the [[evaluating-rag-systems|refusal cases]] accordingly.

## What this looks like in practice

The RAG assistant's prompt carries three obligations — answer from context, cite, refuse when insufficient — and its evalset pins all three. Prompt edits run the suite; a change that lifts answer quality but starts answering unanswerable questions gets reverted.
