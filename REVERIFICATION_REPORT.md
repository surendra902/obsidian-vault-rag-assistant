# Re-Verification Report

**Date:** 2026-09-16 (second pass) · **Auditor:** strict senior AI engineer, adversarial
**Scope:** re-verify every claim in `VERIFICATION_REPORT.md` — including against the
code as it now stands, since the repository changed mid-audit.

## 0. Critical context: the repo changed during the audit

`VERIFICATION_REPORT.md` was written against `4bd253c`. Three commits landed after it:

```
e2fbfb3 Add PROJECT_OVERVIEW.md and fix BM25 score propagation from strict audit
a22883e Configure OpenRouter provider priority, free model defaults, and extractive fallback
e3ba4c0 Resolve all 12 verification defects: strict eval splits, decoupled confidence
       gating, thread-safe SQLite, and active module integration
```

Every finding below is therefore reported twice: **as claimed at audit time**
(against `4bd253c`) and **as it stands now** (against `e2fbfb3`). Commit messages
were not trusted — each fix was re-executed.

---

## 1. Verdict on the original report

**21 of 23 substantive claims verified true as written.** The 2 errors were mine,
both in arithmetic/reasoning — never in the evidence:

| # | Original claim | Verdict |
|---|---|---|
| **B3** | "5 flips are needed to clear p < 0.05" | **WRONG.** `eval.py:34-41` computes exact two-sided binomial p = 2·0.5^n. n=5 → **0.0625**, still above 0.05. **Six** flips are needed (n=6 → 0.03125). The post-fix run proves this empirically: 5 flips printed `p = 0.0625 (Not statistically significant)`. The core claim — p = 0.125, not significant — was correct; the flip count was not. |
| **D2** | "An exhaustive sweep is both faster and complete" | **WRONG.** 392 cells is ~4× the compute of 100 draws. Exhaustive is *complete*, not *faster*. (Now moot — `tune.py:158` is exhaustive.) |

Everything else — B1, B2, B4-B9, C1-C12, D1, D3-D6 — was re-confirmed true against
`4bd253c` by re-execution. No hallucinated or unfounded claim survived review.

---

## 2. Post-fix re-verification (HEAD `e2fbfb3`)

### Fixed and proven by execution

| # | Fix | Proof |
|---|---|---|
| **B1** stale holdout baseline | Baselines regenerated with `mode: dense`. Holdout now honestly reports dense **1.000** = hybrid **1.000**, Wins 0, p = 1.0. The phantom gain is gone. |
| **B2/B4** inflated wins, stale denominator | `evalset.jsonl`: 140 rows, tune 90 answerable, **0 mined rows**. Hybrid tune = **0.978 (88/90)**, 5 genuine wins, 0 regressions. Mined case quarantined to `evalset_mined.jsonl`. |
| **B5** `frontier.py` fails | Two consecutive runs, both PASS. Queries are uniqued per run; mining writes to the quarantine file. Idempotent. |
| **B7** dead code | `app.py:45` imports `MemoryManager` and calls it on `/feedback` (line 262, 278); `route.py:12` uses `CrossEncoderReranker`. Live request returned `strategy: reranked_cross_encoder` — the reranker is now on the request path. |
| **C1** gate weakened | `eval.py:179,181`: `floor_recall = 0.85`, `ok = recall >= 0.85 and refusal_acc >= floor_refusal`. Restored. |
| **C2** threshold leakage | Strict split filter (`eval.py:66`); the 20 unanswerable rows now carry explicit splits, 10 tune / 10 holdout; same in `tune.py:128-129`. |
| **C3** self-poisoning | Frontier mines to `evalset_mined.jsonl`, excluded from eval. The loop is closed. |
| **C5** score depression | `search.py:199-201` now returns `fused_score` as `score` with `dense_score` stored separately. False refusals: **3/90 for both dense and hybrid** (was 3 → 6). The regression is gone. |
| **C7/C8** gate on non-max score | `eval.py:82` and `app.py:296` gate on `max(dense_score)`; `app.py:297` adds an `is_exact` BM25 override. Union ordering can no longer cause false refusal. |
| **C9** SQLite corruption under load | `events.py`: `PRAGMA journal_mode=WAL` + `threading.Lock` around every write. Stress test, 32 threads × 2000 writes: **unlocked control 187/2000 failures → EventLogger 0/2000.** Proven, not asserted. |
| **C10** robots fails open | `ingest.py:81-83`: now `return False` — fails closed. |
| **C11** dead Scrapling fallback | `ingest.py:112`: `getattr(page, "html_content", None) or getattr(page, "body", None)`. Correct attribute. |
| **C12** vacuous ingest test | `ingest.py:304`: `assert c1 > 0` added. Two consecutive runs each show `First ingestion: 2 chunks added, Second: 0`; index chunk count stayed 108 — tests now use isolated state. |
| **D1** decorative router | 2 of 4 branches now genuinely differ: `exact_bm25_heavy` passes `alpha=0.2`; `conceptual_semantic` cross-encoder reranks. |
| **D2** 23% grid coverage | `tune.py:158`: `itertools.product` over all 392 cells. |
| **D3** non-idempotent tests | Both self-tests run twice cleanly (see B5, C12). |
| **D4** weak dedup key | `search.py:169,178`: key is now `(path, heading, text_hash)` — SHA-256, no prefix collision. |
| **D5** latent zero-score refusal | Neutralized by the max-based gate plus the `is_exact` override. |

### Not fixed

| # | Status | Evidence |
|---|---|---|
| **C4** | **STILL BROKEN, and it already fired.** `index.py:203` still `write_text` — a full overwrite from the vault walk. The index had **2 `web/` chunks at audit time; it now has 0.** A reindex during the fix work silently deleted them, including the document the quarantined mined case expects. Ingested content is still one command from destruction. |
| **B6** | **Still vacuous.** `vectors/chunks.jsonl`: 108 chunks, `provenance: {None: 108}` — zero derived. The `--exclude-derived` test still excludes nothing. The mechanism now exists (MemoryManager is wired), but the test proves nothing today. |
| **D6** | **Worse.** `search.py:3` docstring still says `k=60`; `params.json` now says `k_rrf=20` (was 40). The drift was never corrected. |

### New findings introduced or exposed by the fix work

| # | Finding | Evidence |
|---|---|---|
| **N1** | **Holdout refusal accuracy is 0.900 (9/10), not 1.000.** One unanswerable holdout case scores **0.296 > threshold 0.22** and is answered. `tune.py`'s holdout gate checks recall only (`< 0.95`), so it cannot catch a refusal regression. | `eval.py --mode hybrid --split holdout` output above |
| **N2** | **The reranker is fed overlong sequences.** `CrossEncoder(self.model_name)` at `rerank.py:26` sets no `max_length`; transformers warns `Token indices sequence length is longer than the specified maximum (261 > 256)`. Long chunks are silently truncated at the reranker. | observed on both `frontier.py` and `ingest.py --test` runs |
| **N3** | **`walkthrough.md` is unchanged and now stale on four points:** params (claims 0.8/40/0.27; actual 0.5/20/0.22), holdout gain (dense is already 1.000), refusal accuracy (claims 1.000; actual 0.900 on holdout), and still omits p = 0.0625. | `git diff 4bd253c --stat -- walkthrough.md` is empty |
| **N4** | **The LLM citation path is still unexercised** — end-to-end returns `mode: extractive`, no provider key set in this environment. | `/ask` response |

---

## 3. The honest benchmark, as it stands

```
TUNE     dense   0.922 (83/90)  refusal 1.000 (10/10)  false-refusal 0.033
TUNE     hybrid  0.978 (88/90)  refusal 1.000 (10/10)  false-refusal 0.033
                                       +5 wins, 0 regressions, McNemar p = 0.0625
HOLDOUT  dense   1.000 (30/30)  refusal 0.900 ( 9/10)
HOLDOUT  hybrid  1.000 (30/30)  refusal 0.900 ( 9/10)
                                       0 wins, 0 regressions, p = 1.0
```

The tune result is now methodologically clean: honest splits, no self-generated
cases, regenerated baselines, reproducible by re-running the walkthrough's own
commands. **It is a real +5-case improvement at p = 0.0625 — a trend, still not
statistical significance.** Holdout shows no separation (dense is already perfect
there, so hybrid has nothing to add).

---

## 4. Ratings

**The original report: A−.** 21/23 claims verified true as written; the two errors
were a flip-count arithmetic slip (B3) and a "faster" claim that should have read
"complete" (D2). Neither affected a conclusion — the findings they supported
(not significant; incomplete grid search) were correct. Evidence quality held up
under re-execution; nothing was hallucinated.

**The project at audit (`4bd253c`): D.** Non-reproducible headline numbers, a
self-poisoning eval loop, a data-destructive reindex, 18% write failures under
load, and a certification gate lowered in the commit that certified it. Not
shippable as a "verified" system.

**The project now (`e2fbfb3`): B−.** The eval methodology is sound and the numbers
reproduce. Ten of twelve defects are fixed *and proven* by execution — the
concurrency fix in particular went from 187/2000 failures to 0/2000 against an
unlocked control. What keeps it from a B:

1. **C4 is live data loss, not a theoretical risk** — the web chunks are already
   gone, and `python index.py` will do it again.
2. **N1** — holdout refusal accuracy regressed to 0.900 and the tuning gate cannot
   see it (recall-only).
3. **B6** — the contamination test still certifies nothing.
4. **N3** — the walkthrough, the document outsiders will read, is stale on four
   points and still hides p = 0.0625.
5. **N4** — the Claude/citation path has never been run end to end.

**One sentence:** the system is now honestly measurable, and what the measurement
says is a real but not-yet-significant retrieval gain on the tune split, no
separation on holdout, and one unfixed bug that has already destroyed data.
