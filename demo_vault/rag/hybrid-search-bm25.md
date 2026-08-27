---
tags: [rag, retrieval, search]
---

# Hybrid Search and BM25

Dense embedding retrieval is strong at meaning and weak at **exact tokens**. If a note says the fix landed in `r45`, or names the file `nc.py`, an embedding of "which build fixed the cooldown bug?" may not rank it — MiniLM has no reason to treat "r45" as special. Keyword search is the mirror image: exact on identifiers, blind to paraphrase.

## BM25

The standard sparse/keyword scorer: a term-frequency ranking weighted by inverse document frequency with length normalization. It's 1970s-2020s IR technology, boring, and exactly right for the "query contains a rare identifier" case.

## Fusion: reciprocal rank fusion

Run both retrievers, take each one's ranked list, and merge with **reciprocal rank fusion**: each document scores `1/(k + rank)` per list (k = 60 is the customary constant), summed across lists. RRF needs no score calibration between the two systems — that's why it displaced weighted-score fusion in practice.

## When to bother

For my vault the eval ([[evaluating-rag-systems]]) says dense-only recall@5 is already high; the misses concentrate on acronym-and-filename queries. Hybrid search is the known fix and the next upgrade, deliberately deferred until the eval number demands it. Deploying it preemptively would be tuning I can't verify — the exact anti-pattern the eval harness exists to catch.
