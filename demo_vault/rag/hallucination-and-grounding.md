---
tags: [rag, llm, quality]
---

# Hallucination and Grounding

An LLM generates the most plausible continuation, not the most truthful one. With nothing in context, "plausible" and "true" are unrelated. Grounding — answers constrained to retrieved context — is the mitigation, and it has to be designed at every layer, not requested politely in a prompt.

## Layered defense

1. **Retrieval threshold**: if the best chunk's similarity is below a floor, don't call the model at all — return "not in the vault." Garbage in, confident garbage out.
2. **Prompt contract**: the system prompt says answer *only* from the provided excerpts, cite them, and say when they're insufficient. Model instructions reduce, never eliminate, drift.
3. **Citations**: force every claim to carry a source the user can open — [[citation-patterns]]. A claim without a checkable citation isn't grounded; it's decorated.
4. **Eval**: measure refusal accuracy on deliberately unanswerable questions, and groundedness on answerable ones. See [[evaluating-rag-systems]].

## The subtle case

The worst failure isn't the model inventing from nothing — it's **parametric memory filling a gap**: context is *almost* enough, and training data silently supplies the rest. The answer reads as grounded because most of it is. This is why the refusal threshold belongs at the retrieval layer where it's measurable, not inside the model where it isn't.

## Refusal is a feature

A system that answers 90% of questions correctly and says "not in your notes" for the other 10% beats one that answers 100% with 15% fabricated. Users stop trusting the second kind permanently.
