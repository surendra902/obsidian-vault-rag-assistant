# Strict Verification Report — obsidian-vault-rag-assistant

**Scope:** every claim in `walkthrough.md`, re-executed against the live repo.
**Date:** 2026-09-16 · **Verdict:** git history and code are real; **the headline
benchmark claims do not reproduce**, and 12 defects are confirmed by execution.

Nothing below is inferred. Each finding names the file:line and the command or
output that proves it.

---

## 0. What is actually true

| Claim | Status |
|---|---|
| Repo exists at the stated path | verified |
| All 8 phase commits exist with matching hashes | verified (`75377e2`…`4bd253c`) |
| `params.json` matches the walkthrough exactly | `alpha 0.8, k_rrf 40, threshold 0.27, k 5, depth 50` |
| `test_citations.py` | 9 passed |
| `events.py`, `memory.py`, `route.py` self-tests | pass |
| `rerank.py` depth cap = 15 | `MAX_CPU_DEPTH = 15` |
| Tune split: hybrid beats dense on recall | real, +4 (0.923 → 0.967) |

Everything else below is wrong, unproven, or broken.

---

## 1. BLOCKERS — benchmark claims that do not reproduce

### B1. The holdout result is false. Hybrid gains **nothing** on holdout.

Walkthrough §3: *"Dense Baseline 0.967 (29/30) → Hybrid 1.000 (30/30)"*.

```
python eval.py --mode dense  --split holdout   ->  recall@5 = 1.000 (30/30)
python eval.py --mode hybrid --split holdout   ->  recall@5 = 1.000 (30/30)
python eval.py --mode hybrid --split holdout --compare baseline_holdout.json
                                               ->  Wins 0, Regressions 0, p = 1.0000
```

Dense is already perfect. The entire claimed holdout improvement is an artifact of
comparing against a stale file. `baseline_holdout.json` has **`"mode": null`** — it
was written before `--mode` existed, so it records an unlabeled run against a
different index. **The only frozen-holdout evidence in the project is invalid.**

### B2. "+5 Wins" is 4 wins. The 5th is a hit the system wrote for itself.

```
wins=4  regressions=0  cases_absent_from_baseline=1
  WIN: Refusal threshold calibration for unanswerable questions in retrieval
  WIN: 17*24
  WIN: bge-small-en-v1.5
  WIN: System 1 silent substitution
  NOT IN BASELINE: What is the architecture of an asynchronous actor model…
```

The 5th case is the **self-mined** row (`source: "mined"`) that `frontier.py`
appended to `evalset.jsonl` after Phase 8. It was never in the baseline, so it
cannot be a "win" — it is a new question scored against a document the system
generated. See **C3**.

### B3. McNemar p = 0.1250. The tune improvement is not statistically significant.

The walkthrough reports the +wins and omits the p-value that `eval.py` prints on
the same line. With 4 discordant pairs and 0 regressions the minimum achievable
p is 0.0625; **5 flips are needed to clear p < 0.05**. The project's own test
says the result cannot be distinguished from chance, and the walkthrough presents
it as verified.

### B4. Tune recall is 0.967 (88/91), not 0.978 (88/90).

141 rows in `evalset.jsonl`, not 140: `tune 91 · holdout 30 · unanswerable 20`.
The numerator is right, the denominator is stale — Phase 8 mined a 91st tune case
and the walkthrough was never regenerated. Excluding the mined case: **87/90 = 0.967**.

### B5. `frontier.py` **fails**. The walkthrough says PASSED.

```
$ python frontier.py
Gap enqueued: False
AssertionError: Expected test query in pending gaps     (frontier.py:136)
```

Root cause: `frontier` table has `query TEXT UNIQUE` (frontier.py:31). The test
query is already `resolved` in `vectors/frontier.db`, so `enqueue_gap` returns
False and `get_pending_gaps` (status = 'pending' only) never returns it. **The
test passes exactly once, on a clean DB, and can never be re-run.**

### B6. The contamination test is vacuous.

Walkthrough §3: *"Contamination Test (`--exclude-derived`): Zero contamination."*

```
chunks: 110 · provenance values: {None: 108, 'primary': 2} · derived count: 0
```

`--exclude-derived` filters `provenance == "derived"`. **No chunk in the index has
that value.** The flag excludes nothing, the two runs are identical by
construction, and the test would report "zero contamination" on a fully
contaminated index.

### B7. Phase 4 and Phase 6 are marked PASSED but are dead code.

```
$ grep -rn "import rerank|from rerank|import memory|from memory|CrossEncoder|MemoryManager" *.py
  (no results outside rerank.py / memory.py themselves)
```

`rerank.py` and `memory.py` have **zero importers**. Their self-tests pass in
isolation; neither participates in `/ask`. Phase 6's "accuracy verified" and
Phase 4's "PASSED" describe standalone scripts, not the system.

### B8. The "Claude Answer Generation" box in the architecture diagram is unexercised.

The end-to-end script in §5 returns `mode: extractive`, `provider: none`
(app.py:280, 449). No key is set, so every documented run is excerpt-only. The
citation path (app.py:195-213) was never executed in the verification.

### B9. Internal contradiction: Phase 0 row says 0.889, its own artifact says 0.922.

The Phase 0 table row claims `tune recall=0.889 baseline`; `baseline_tune.json`,
written by that same commit, records `recall = 0.9222 (83/90)`, and §3 repeats
0.922. 0.889 is unsourced.

---

## 2. CRITICAL DEFECTS — real bugs in the code

### C1. The eval gate was silently weakened in the commit that claims it was met.

```
$ git show 75377e2 -- eval.py
- ok = recall >= 0.85 and refusal_acc >= 0.80
+ floor_recall = 0.60 if args.split in ("tune", "holdout") else 0.70   (eval.py:173)
```

Phase 0 is titled *"headroom gate met"*. The same diff drops the recall bar from
**0.85 to 0.60**. A run at 0.62 now exits green while failing the original standard.

### C2. Threshold leakage — tune and holdout share all 20 unanswerable cases.

```python
# eval.py:66-68
cases = [c for c in all_cases if c.get("split") == args.split or (c.get("unanswerable") ...)]
if args.split == "holdout":
    cases = [c for c in all_cases if c.get("split") == "holdout" or c.get("unanswerable")]
# tune.py:124-125 — same construction
tune_cases    = [... if c.get("split") == "tune"    or c.get("unanswerable")]
holdout_cases = [... if c.get("split") == "holdout" or c.get("unanswerable")]
```

The 20 unanswerable rows carry **no `split` key at all** and are unconditionally
injected into both. `threshold` is the one parameter determined entirely by those
cases — so it is tuned on the holdout and then "validated" on the holdout.
**The holdout provides zero independent evidence for the refusal threshold**, and
the reported "Refusal Accuracy 1.000 (20/20)" is the same 20 cases, twice.

### C3. Self-poisoning loop — the eval set grows every time the suite runs.

`frontier.py:116` appends a mined case to `evalset.jsonl` on every successful run,
and `frontier.py:163` asserts it lands in **tune**. The mined case scores:

```
expect: web/distributed-actor-model-architecture.md -> HIT=True at rank 0, score=0.7399
demo_vault/web/distributed-actor-model-architecture.md exists: False
```

The system crawled a page, wrote the chunk, then added an eval case asserting that
chunk is retrievable. It is the **highest-scoring case in the entire set** and a
guaranteed hit. Recall inflates by one free point per run, and the row is now
committed in `1afae0c`. This is exactly the failure mode `self.md` warns about,
realised in the implementation.

### C4. `python index.py` silently destroys the web chunks.

```
index.py:203   (out / "chunks.jsonl").write_text(...)   <- full overwrite from vault walk
ingest.py:188  with open(self.chunks_file, "a") as f:   <- append
demo_vault/web/ : No such file or directory
```

Web chunks exist **only** inside `vectors/chunks.jsonl`, with no file on disk. Any
reindex rebuilds the file from `demo_vault/` alone, deleting them — which breaks
the mined eval case (C3) and loses all ingested content with no warning.

### C5. Hybrid systematically depresses `hits[0].score`. Both refusal results are artifacts.

```
n=141   mean(hybrid - dense rank-0 score) = -0.0113
        hybrid LOWER in 36 cases, HIGHER in 0, equal in 105
```

RRF reorders by fused rank but `search.py:180` returns the **dense** score, so the
chunk promoted to rank 0 usually has a *lower* dense similarity. One mechanism,
two consequences — the walkthrough reports only the flattering one:

* **Reported:** refusal accuracy 0.950 → 1.000. The single case dense missed sits
  at 0.2956 and hybrid pushes it to 0.2633, just under the 0.27 gate. Not better
  retrieval — a downward bias that happened to cross the line.
* **Unreported:** false refusals on tune **0.033 (3/91) → 0.066 (6/91)**, double.

### C6. All 6 hybrid false refusals retrieved the correct document.

```
hybrid: 6 false refusals
   score=0.1037  recall_hit=True   17*24
   score=0.1067  recall_hit=True   2026-08-26
   score=0.1115  recall_hit=True   2026-08-25
   score=0.2401  recall_hit=True   How do I stop an LLM from making things up?
   score=0.2667  recall_hit=True   Why fine-tuning foundation weights fails…
   score=0.2693  recall_hit=True   David Allen open loops
```

Retrieval succeeded; the gate threw the answer away. Three of these are cases
dense would have answered. Note `17*24` — celebrated as a **win** in B2 — is a
case the shipped system **refuses to answer**.

### C7. `hits[0].score` is not the maximum score. The production gate reads it anyway.

```
search.py:166  sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)   <- RRF order
search.py:180  score=preserved_dense                                                <- dense value
app.py:269     if not hits or hits[0].score < refuse_thresh:                         <- gates on rank 0
```

The returned list is ordered by one quantity and scored by another.

```
cases where hits[0].score < max(score): 16/91 tune cases (17.6%)
example: [0.3368, 0.4213, 0.2971, 0.2844, 0.3053]   <- rank 1 is stronger than rank 0
```

A strong match at rank 1 is invisible to the refusal gate.

### C8. `decomposed_union` compounds C7.

`route.py:59-65` unions per-sub-query results in sub-query order, preserving each
hit's own dense score:

```
"System 1 versus System 2 thinking" -> hits[0]=0.3091  max=0.3675
```

The gate sees 0.3091. With a marginally weaker first sub-query this refuses a
query the index answered correctly.

### C9. Shared SQLite connections, no locking — 18.4% write failure under load.

```
events.py:22 · frontier.py:25 · ingest.py:46 · search.py:39
    sqlite3.connect(..., check_same_thread=False)
grep -rn "Lock()|threading|BEGIN IMMEDIATE|WAL|isolation_level" *.py  ->  no results
```

`check_same_thread=False` disables the safety check and adds nothing in its place.
FastAPI runs sync endpoints in a threadpool, and `/ask` writes to both `events.db`
and `frontier.db`. Stressing the exact shared-connection path used by `/ask`:

```
shared-connection writes: OK=1633/2000  FAILURES=367  (18.4%)
   x118  DatabaseError: not an error
   x102  DatabaseError: no more rows available
    x79  SystemError: error return without exception set      <- C-level module corruption
    x37  OperationalError: cannot start a transaction within a transaction
    x15  OperationalError: not an error
     x7  OperationalError: cannot commit - no transaction is active
```

Python's implicit transaction management on one connection is being driven by many
threads at once. The commit message advertises **"multi-thread SQLite"**; a 64-request
smoke test passes because embedding latency serialises it, which is why this was missed.

### C10. `robots.txt` fails **open**.

```python
# ingest.py:76-78
except Exception:
    # If robots.txt cannot be fetched or parsed, default to permissive
    return True
```

A timeout, a DNS blip, or a malformed `robots.txt` silently **authorises** the
crawl. The check is enforced on the ingest path (ingest.py:220), so this is the
only thing standing between the crawler and a disallowed host. Safe default is deny.

### C11. The Scrapling fallback can never fire. It is dead twice over.

```python
# ingest.py:107
js_text = trafilatura.extract(page.text) if page and page.text else None
```

Verified against the installed scrapling 0.4.14:

```
Selector.text                     -> ''            <- guard short-circuits, js_text = None
trafilatura.extract(.text)        -> None          <- and extract() rejects de-tagged text anyway
trafilatura.extract(.html_content)-> 'Hello world retrieval augmented generation.'
```

`Response.text` is the Selector's text node (empty at document root), not the HTML
body. `except: pass` at line 110 swallows the whole thing. Every "JS-heavy page"
silently returns nothing. Correct call is `page.html_content`.

### C12. `ingest.py --test` proves nothing.

```
$ python ingest.py --test
First ingestion:  0 chunks added.
Second identical ingestion: 0 chunks added.
Idempotence verified: PASS
```

`ingest.py:264` asserts only `c2 == 0`. The document was already ingested on a
prior run, so `c1 == 0` too, and **`0 == 0` passes**. This test would go green if
`ingest_document` were `return 0`. It never checks that the first ingestion worked.

---

## 3. DESIGN ISSUES

| # | Finding | Evidence |
|---|---|---|
| D1 | **The router is decorative.** 4 of 5 branches issue the identical call `self.idx.search(query, k=k, mode="hybrid")` and differ only in the returned label. `exact_bm25_heavy` does not bias toward BM25. | route.py:48, 67, 71 |
| D2 | **Random search covers 23% of a 392-cell grid.** 7 alpha × 8 k_rrf × 7 thresholds = 392; 100 random draws *with replacement* ≈ 88 unique. Candidates are prefetched once, so scoring is cheap — an exhaustive sweep is both **faster and complete**. | tune.py:120, 147-151 |
| D3 | **Self-tests are not idempotent.** `ingest.py --test` and `frontier.py` mutate persistent state on first run. They pass once, then turn vacuous (C12) or fail outright (B5). A verification suite that cannot be re-run is not a verification suite. | — |
| D4 | **Dedup key is `text[:100]`.** Two chunks sharing a 100-char prefix collide and are fused as one document. | search.py:152, 159 |
| D5 | **Latent zero-score refusal.** A BM25-only hit is assigned `dense_score = 0.0`; if it lands at rank 0 the gate refuses unconditionally. Currently **0/141 queries** trigger this — the path is live but not firing. Reported for completeness, not as an active bug. | search.py:162, 180 |
| D6 | **Docstring says RRF k=60, code uses 40.** | search.py:3 vs :118 |

---

## 4. Recommended changes, in priority order

**Stop-the-line (do before any further claims):**

1. **Regenerate both baselines with `--mode dense` on the current index**, then re-run
   every comparison. `baseline_holdout.json` (`mode: null`) is unusable. *(B1)*
2. **Give the 20 unanswerable rows an explicit `split`** — 10 tune / 10 holdout — and
   delete the `or c.get("unanswerable")` clause in `eval.py:66,68` and `tune.py:124,125`.
   Re-tune `threshold`. The current value has never been validated. *(C2)*
3. **Make `frontier.py` mine to a quarantined file** (`evalset_mined.jsonl`), excluded
   from reported recall, and revert the mined row from `evalset.jsonl`. A system must
   not grade itself on documents it wrote. *(C3, B2)*
4. **Restore the gate to `recall >= 0.85 and refusal_acc >= 0.80`**, or document in the
   commit why 0.60 is correct. *(C1)*
5. **Rewrite the walkthrough's §1 and §3 from a fresh run.** As written it reports a
   holdout gain that does not exist, a win count inflated by a self-generated case, a
   stale denominator, and omits p = 0.1250. *(B1-B4, B9)*

**Correctness:**

6. Return the fused RRF score in `Hit.score` and gate on a separate, explicit
   `dense_score` field — or gate on `max(h.dense for h in hits)`. One field cannot be
   both the sort key and the confidence signal. Fixes C5, C7, C8 and D5 together.
7. Serialise SQLite writes: one `threading.Lock` per connection, plus
   `PRAGMA journal_mode=WAL`. *(C9)*
8. `robots.txt` → `return False` on exception; log the failure. *(C10)*
9. `trafilatura.extract(page.html_content)`; replace `except: pass` with a logged
   warning. *(C11)*
10. Make `index.py` merge rather than overwrite — preserve records whose `path` has no
    file on disk — or write ingested pages to `demo_vault/web/`. *(C4)*

**Test hygiene:**

11. `ingest.py --test`: use a unique URL per run and assert `c1 > 0 and c2 == 0`. *(C12)*
12. `frontier.py`: run against a temp DB, or `DELETE` the fixture row first. *(B5)*
13. Add a `provenance: "derived"` chunk to the index, or the contamination test stays
    vacuous forever. *(B6)*
14. Replace `tune.py`'s 100 random trials with `itertools.product` over all 392. *(D2)*

**Honesty in reporting:**

15. Mark Phase 4 and Phase 6 **"implemented, not integrated"** until something imports
    them. *(B7)*
16. Mark the Claude generation path **unverified** until a run with a key is recorded. *(B8)*
17. Report `false_refusal` beside recall everywhere. The hybrid change doubled it, and
    that number appears in no summary. *(C5, C6)*

---

## 5. Bottom line

The code exists, runs, and the commit history is genuine. One claim survives
contact with the evidence: **hybrid retrieval improves tune recall by 4 cases
(0.923 → 0.967), at p = 0.125 — suggestive, not significant.** It costs double the
false-refusal rate, which is not reported anywhere.

The holdout gain is an artifact of a stale baseline. The refusal gain is an
artifact of a scoring bug. One of the five "wins" is a document the system wrote
for itself. The frozen holdout is not frozen — the threshold was tuned on it.
The gate that certified Phase 0 was lowered in the commit that certified it.

**Recommendation: do not report these numbers anywhere until items 1-5 are done.**
