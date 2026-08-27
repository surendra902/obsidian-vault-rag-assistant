---
tags: [prompting]
---

# Few-Shot Prompting

Showing the model 2–5 input→output examples inside the prompt. It works because the model pattern-matches the transformation you're demonstrating more reliably than it follows a description of that transformation.

## When it helps

- **Format compliance** — the case where it's nearly mandatory. "Return JSON like this: …" followed by one example beats three paragraphs of schema prose.
- **Edge-case signaling** — examples *are* the spec for cases your prose can't enumerate: how to refuse, how to handle a missing field.
- **Narrow classification** — a handful of labeled examples moves a fuzzy category ("is this note actionable?") more than any definition.

## When it doesn't

- **Reasoning-heavy tasks** — examples constrain the solution space to their shape. Show three tidy solutions and the model stops looking for the untidy correct one.
- **Long or diverse inputs** — examples eat context and anchor on surface features of the specific examples chosen.

## The trap

Few-shot examples leak. The model will reuse example *content* — names, numbers, phrasing — in outputs where it doesn't belong. Use obviously fake placeholders (`ACME Corp`, `note-42`) so leakage is visible in testing instead of plausible in production.

Every example added is a permanent token cost per request — unlike the [[system-prompts|cached prefix]], examples change with the task and mostly sit outside the cache. Budget accordingly, and gate changes behind [[evals-for-prompts]].
