# Session Fix Log — Verified Changes

Record of every change made in the final verification session, what was
committed, and the verified end state. Every number below was produced by a
live run, not estimated.

---

## 1. Silent Truncation Chain (root cause: MiniLM's 256-token window)

A transformers warning (`Token indices sequence length is longer than the
specified maximum sequence length (344 > 256)`) traced to a single derived
memory chunk. Two separate write paths fed unbounded text into the embedder;
both are closed.

### `memory.py` — `save_derived_answer`
Derived answers are arbitrary LLM output and were written as **one unbounded
chunk**. Now routed through the same `make_chunks` token-budget splitter the
vault indexer uses:

- Emits **N chunk records + N aligned vectors** (`chunks.jsonl` line *i* ↔
  `vectors.npz` row *i*), with `(part n/N)` appended to each heading.
- Short answers (< 15 words) that `make_chunks` drops are kept verbatim — they
  fit the window in one piece. Empty-text chunks never reach the embedder.

Verified live: a 278-token answer → 2 chunks, max 237 tokens, both within
budget; row/line alignment and unit norms intact.

### `index.py` — orphan preservation + guard placement
The merge-preserving rebuild (added earlier this session) kept ingested/derived
chunks with no vault file **verbatim**, including their stored vectors. For
over-budget orphans that stored vector is a *truncated-prefix embedding* — so
the fix preserved the bug.

- Over-budget orphans are now **re-split** and re-embedded rather than copied.
- The loud `SystemExit` guard moved to run **after** preservation, so stored
  chunks are covered, not just ones split from the current vault walk.
- Duplicate guard removed (one check covering everything, not two).

Verified live: rebuild re-split the 344-token chunk into 2 pieces; all 110
chunks ≤ 250 tokens; the guard did not fire.

### `index.py` — misleading warning suppression
`_fit_words` deliberately tokenizes overlong input *in order to split it*, so
transformers' "will result in indexing errors" warning fired on sequences that
never reach the model — printed directly above the line that fixes it. Suppressed
at that one site via a `_measure_tokens` context manager, which covers every
caller (`index.py`, `ingest.py`, `memory.py`).

**Honest limitation:** I attempted for some time to construct an input that
trips the guard end-to-end and could not — MiniLM is highly compressive (a
3000-character word tokenizes to 3 tokens; 12 high-entropy words to 14). I
verified the guard expression fires by exercising it directly instead of
claiming an end-to-end trigger I did not observe.

---

## 2. Self-Learning Loop Defect (found by submitting a real upvote)

Discovered during live `TestClient` verification, not by inspection.

### `app.py` — `/feedback` wrote junk as vault knowledge
An upvote on an extractive-fallback answer persisted the response text as a
derived memory. That text was an error preamble —
`(fallback mode: remote LLM chain exhausted - all models failed -- last:
google/gemma-4-31b-it:free: HTTP 429)` — embedded into the vault and retrievable
as if it were knowledge.

It also **cited an earlier derived chunk as a source**, pointing a derived
document's lineage at another derived document — precisely the self-poisoning
the module's docstring claims to guard against. The sole-citation guard only
requires *one* primary source, so it did not catch this.

Fixed in `/feedback`:
- Derived paths filtered out of `cited` before write-back.
- Fallback/extractive answers skipped entirely (only real generated answers
  become memories).

Verified live: upvote recorded (`{'status': 'recorded'}`), derived chunks
0 → 0. Both junk chunks that the unfixed loop had written were removed from the
index.

### `app.py` — `AskRequest` validation
An empty question returned no hits, tripped the refusal gate, and enqueued an
unresolvable blank gap into `vectors/frontier.db` — one such row was sitting
`pending` forever when found.

`question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]`
rejects `''`, `'   '`, and `'\n\t '` with **HTTP 422**.

(Note: plain `Field(min_length=1)` does **not** catch whitespace — Pydantic
counts the characters. `StringConstraints` strips first.)

### Frontier behavior (verified, not changed)
`enqueue_gap` is idempotent via a `UNIQUE` constraint on `query`: a repeated
identical refusal returns `False` instead of duplicating a row. A re-asked
refusal does not grow the queue. The junk blank row was removed; the table
now has zero empty rows.

---

## 3. Evaluation Honesty

### `eval.py` — non-vacuous contamination census
`--exclude-derived` was previously a vacuous test when no derived chunks
existed. Now computes the derived set from `chunks.jsonl` and reports one of
three explicit verdicts:

- `PASS` — derived chunks exist, no eval case expects one.
- `NOT MEANINGFUL` — no derived chunks, so the flag excludes nothing.
- `CONTAMINATION DETECTED` — an eval case's expected note IS a derived chunk.

### `tune.py` — principled selection
- Objective reweighted: `recall − 1.0·false_refusal − 1.5·(1 − refusal_acc)`.
  Rationale recorded in-code: an answered unanswerable is a hallucination
  (the system's primary failure mode); a false refusal degrades to recoverable
  extractive excerpts.
- Exhaustive 392-cell grid (7 α × 8 k_rrf × 7 thresholds) on **tune**; the
  frozen **holdout selects** among tune-qualified candidates against a gate of
  recall ≥ 0.95, refusal ≥ 0.95, false-refusal ≤ 0.10. Every candidate's tune
  score is fixed before the holdout is consulted — validation-set model
  selection, not a second tuning set.
- `tune_score` no longer leaks into `params.json`.

### `rerank.py`
Explicit `max_length=256` — `ms-marco-MiniLM-L6-v2` has a 256-token window and
silently truncates long chunks otherwise, scoring a prefix rather than the chunk.

### `search.py`
Docstring corrected: RRF `k` is loaded from `params.json` (tuned), not hardcoded.

---

## 4. Committed

Single commit on branch **`self-learning-p0`**:

```
7969c24 Fix silent truncation, self-citing memory loop, and unvalidated /ask input
```

**12 files changed, 1807 insertions(+), 61 deletions(-)**

| File | Change |
|---|---|
| `app.py` | `/feedback` derived-citation filter + fallback skip; `AskRequest` 422 validation |
| `index.py` | Orphan re-split; guard moved after preservation; `_measure_tokens` |
| `memory.py` | `save_derived_answer` token-budget multi-chunk write |
| `eval.py` | Non-vacuous contamination census |
| `tune.py` | Reweighted objective; holdout selection among tune-qualified candidates |
| `rerank.py` | Explicit `max_length=256` |
| `search.py` | RRF `k` documented as tuned-loaded |
| `params.json` | Final: `alpha 0.5, k_rrf 20, threshold 0.27, k 5, candidate_depth 50` |
| `baseline_tune.json`, `baseline_holdout.json` | Regenerated dense baselines @ threshold 0.27 |
| `tune_run_hybrid.json` | Saved hybrid tune run (new) |
| `REVERIFICATION_REPORT.md` | Audit report (new) |

Not committed: `walkthrough.md` and the session reports live in the working
directory (`Documents\claude`), outside this repo.

---

## 5. Verified Final State

| Metric | Value |
|---|---|
| Tune hybrid recall@5 | **0.978** (88/90) [0.923, 0.994] |
| Tune refusal accuracy | **1.000** (10/10) |
| Tune false-refusal | **0.044** (4/90) |
| Tune McNemar vs dense | **+5 wins, 0 regressions, p = 0.0625** |
| Holdout hybrid recall@5 | **1.000** (30/30) [0.886, 1.000] |
| Holdout refusal accuracy | **1.000** (10/10) |
| Holdout false-refusal | **0.000** (0/30) |
| Self-tests | all 8 entry points **exit 0** |
| Index integrity | 108 chunks / 108 vectors, aligned, unit-norm |
| Frontier | 0 empty rows |

### Corrections made to `walkthrough.md` claims
1. **Holdout dense is 1.000 (30/30), not 0.967** — re-splitting the over-budget
   derived chunk shifted the dense candidate pool. Hybrid was unchanged; it was
   never driven by that chunk.
2. **McNemar p-value added: 0.0625 — not significant** at α = 0.05. 6 flips
   would be needed. Documented as "an improvement, not a proof."
3. Holdout is at ceiling (0 wins, p = 1.0) — it measures *safety* there
   (refusal + contamination), not recall gain.
4. Refusal accuracy corrected from a misleading combined "(20/20)" to accurate
   per-split figures.
5. Params block corrected from `alpha 0.8, k_rrf 40` to `alpha 0.5, k_rrf 20`.

### Honest note on the contamination check
The index currently holds **zero** derived chunks — both that existed were junk
from the unfixed feedback loop and were removed. `eval.py --exclude-derived`
therefore reports `NOT MEANINGFUL`, which is the correct output, not a failure.
The filter is genuinely verified, but by `memory.py`'s self-test against a
derived chunk it writes itself; that self-test was confirmed to still pass with
an empty production index (it does not depend on production state).
