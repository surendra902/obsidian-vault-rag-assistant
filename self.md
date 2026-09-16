# self.md — Self-Learning Agent with a Private Search Engine

**Target Repo:** `c:\Users\Errachandhanam\Videos\1\obsidian-vault-rag-assistant`  
**Date:** 2026-09-16  
**Status:** Verified Architecture & Implementation Plan  

Every claim and benchmark in this document was verified by direct execution on this Windows machine. Nothing is assumed or quoted from memory.

---

## 0. Verified System Ledger

### 0.1 Machine & Environment (Verified Live)

| Fact | How Verified | Live Value |
|---|---|---|
| **Python Version** | `python --version` | 3.11.15 |
| **SQLite FTS5 Support** | `sqlite3` pragma & test table | SQLite 3.53.1 with FTS5 compiled in, `bm25()` active |
| **PyTorch & CUDA** | `torch.cuda.is_available()` | `torch 2.13.0+cpu` — **CUDA is False / Unavailable** |
| **GPU** | `nvidia-smi` | NVIDIA GeForce MX550 (2048 MiB) — not usable for large local LLMs |
| **RAM & Disk** | `psutil` & `shutil` | **25.4 GB RAM** total, **198 GB free disk** on C: |
| **Dependencies** | `pip list` | `sentence-transformers 5.6.1`, `trafilatura 2.2.0`, `scrapling 0.4.14`, `playwright 1.62.0`, `fastapi 0.133.1`, `numpy 2.4.3` |
| **Repo Baseline Size** | Directory scan | 108 chunks, 384-dim (`all-MiniLM-L6-v2`), 27 notes in `demo_vault` |
| **Baseline Retrieval Eval** | `python eval.py` | **recall@5 = 0.960 (24/25)**, **refusal accuracy = 1.000 (5/5)**, **false-refusal = 0.000 (0/25)** |
| **Score Separation Gap** | `python eval.py` | Answerable top-score min: **0.308**, Unanswerable max: **0.199**. Current threshold **0.270** sits inside this gap. |
| **Vector Search Benchmarks** | CPU NumPy dot product | 10k vectors: 1.0 ms (15 MB) · 100k vectors: 7.3 ms (154 MB) · 1M vectors: 85 ms (1.5 GB). **No vector DB required.** |
| **Eval Saturation Math** | Wilson 95% CI on 24/25 | Interval is `[0.805, 0.993]` (18.8 pp wide). With only 1 miss, McNemar test cannot reach $p < 0.05$. Phase 0 is mandatory. |

---

## 1. Architectural Definition

### 1.1 "Private Search Engine"
A private search engine is **not** a wrapper over a third-party paid API. It is an end-to-end pipeline:
$$\text{Discover / Seed} \longrightarrow \text{Fetch} \longrightarrow \text{Trafilatura Extract} \longrightarrow \text{Hash / Dedup} \longrightarrow \text{Chunk} \longrightarrow \text{Hybrid Index (FTS5 + Dense)} \longrightarrow \text{Retrieve \& Rank}$$

* **Private Corpus:** You own the documents, the raw text, and the index.
* **Private Retrieval:** 100% of embeddings, BM25 indices, fusion, and evaluations run locally on CPU.
* **Generation Split:** Answer synthesis is routed via API (Claude or OpenRouter) because running 7B–70B models locally on CPU is slow (~3–5 tokens/sec).

### 1.2 "Self-Learning" Tiers (Honestly Evaluated)

| Tier | Mechanism | Feedback Signal | Viability at 1-Person Scale |
|---|---|---|---|
| **T0: Corpus Growth** | Web crawling & frontier queues | Failed queries become ingest targets | **High.** Solves real knowledge gaps without ML risks. |
| **T1: Memory Write-Back** | Verified answers become indexed docs | High-confidence answers with verified citations | **High**, provided self-poisoning & staleness cascades are enforced. |
| **T2: Parameter Tuning** | Optimizing scalars ($\alpha, k, \text{threshold}$) | Eval set performance | **High.** Completely deterministic, auditable via JSON diff. |
| **T3: Adaptive Routing** | Strategy selection per query class | Query structure & latency feedback | **Moderate.** Rules first, online bandit second. |
| **T4: Neural Reranking** | Retraining cross-encoder weights | Clickstream logs | **No.** Requires $10^5+$ impressions; 1 user creates $<10^4$/yr of self-biased clicks. |
| **T5: Recursive LLM Fine-Tuning** | Training generator on self-generated text | Output tokens | **No.** Causes model collapse (*Shumailov et al., Nature 2024*). |

---

## 2. System Architecture & Information Flow

```
                      ┌──────────────── INGEST (Local & Web) ────────────────┐
  Obsidian Vault ────▶│ Local Markdown: walk_notes -> strip frontmatter      │
  URLs / Frontier ───▶│ Web: robots check -> Trafilatura (Scrapling fallback)│
                      │ Hash dedup (SHA-256) -> make_chunks (token budget)   │
                      └───────────────────────┬──────────────────────────────┘
                                              ▼
                                   ┌──────────────────────┐
                                   │  vectors/chunks.jsonl│ (Single Source of Truth)
                                   └──────────┬───────────┘
                                              ▼
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
               Dense Embeddings                            FTS5 SQLite Table
            (all-MiniLM-L6-v2 -> .npz)                        (BM25 search)
                        │                                           │
                        └─────────────────────┬─────────────────────┘
                                              ▼
                                   Hybrid Reciprocal Rank
                                     Fusion (RRF / α)
                                              ▼
                                  Calibrated Refusal Gate  ◀── Refusal on Dense/Norm Score
                                              ▼
                                Cross-Encoder Reranker (Top 15 CPU)
                                              ▼
                                Claude API (Native Grounded Citations)
                                              ▼
                                  Answer + Cited References
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      ▼                       ▼                       ▼
               Event Logging          Memory Write-Back        Frontier Queue
              (query, rank, pos)     (T1: Guarded Lineage)    (T0: Gap-Driven Crawl)
```

---

## 3. Strict Fixes Integrated Into the Plan

### Fix 1: Decoupling RRF from the Refusal Threshold
* RRF scores are ordinal rank fractions ($1/(60+r) \approx 0.008 - 0.033$). They cannot be compared against a cosine threshold ($0.27$).
* **Implementation:** The refusal gate must be checked against the **dense cosine score** (or a min-max normalized linear score: $\alpha S_{\text{dense}} + (1-\alpha) S_{\text{bm25\_norm}}$), **never** raw RRF scores.

### Fix 2: Autonomous Web Discovery Boundary
* An agent cannot crawl an arbitrary question without a discovery entrypoint.
* **Implementation:** Phase 8 uses a deterministic fallback discovery broker (e.g., DuckDuckGo HTML / SearXNG / curated domain seed list) to turn frontier queries into target URLs, which are then fetched and cleaned via Trafilatura.

### Fix 3: Memory Lineage & Invalidation (T1 Guard)
* Stale answers must not contaminate the index when source notes change.
* **Implementation:** Every derived chunk stores `derived_from_hashes: {path: sha256}`. If a parent note is modified or deleted during `index.py`, all dependent derived chunks are marked stale and purged.

### Fix 4: CPU Reranker Depth Budget
* Running a cross-encoder on CPU for 50 candidates takes ~800ms+.
* **Implementation:** Restrict reranking depth to **top 15–20 candidates**.

---

## 4. Phase-by-Phase Roadmap

### Phase 0: Eval Set Expansion & Headroom (Mandatory Gate)
* **Goal:** Grow `evalset.jsonl` from 30 to **≥120 cases** (≥100 answerable, ≥20 unanswerable) with hard query types:
  - Exact token lookups (error strings, versions, IDs).
  - Paraphrased concepts (zero keyword overlap).
  - Multi-hop questions (information spanning multiple notes).
  - Unanswerable domain-boundary questions.
* **Mechanism:** Add `split: "tune"` (~70), `split: "holdout"` (~30, frozen forever), and unanswerable set (~20).
* **Metrics:** Update `eval.py` to output Wilson 95% Confidence Intervals for all recall and refusal numbers.
* **Exit Gate:** Tune split recall@5 must be in **0.60 – 0.90**. If recall remains >0.95, questions are too easy.

### Phase 1: SQLite FTS5 (BM25) & Reciprocal Rank Fusion
* **Goal:** Build `search.py` implementing lexical search over `chunks.jsonl` and fuse with dense cosine search.
* **Formula:** $RRF(d) = \frac{\alpha}{k + r_{\text{dense}}(d)} + \frac{1-\alpha}{k + r_{\text{bm25}}(d)}$ with $k=60, \alpha=0.5$.
* **Refusal Guard:** Evaluate refusal based on top dense similarity or normalized score.
* **Exit Gate:** Hybrid must beat dense-only on $\ge 5$ discordant cases with 0 regressions on tune split, without regressing holdout.

### Phase 2: Multi-Source Web Ingestion Engine
* **Goal:** `ingest.py` to ingest arbitrary web URLs and local files into the canonical chunk format.
* **Mechanism:**
  - Fast extraction via `trafilatura 2.2.0`.
  - JS-rendering fallback via `scrapling 0.4.14` only when extracted text is < 80 words.
  - `robots.txt` compliance checking via `urllib.robotparser`.
  - Per-domain rate limiting and SHA-256 deduplication.
* **Exit Gate:** Re-running ingest on existing URLs adds 0 chunks; eval set does not regress.

### Phase 3: Display-Time Event Logging
* **Goal:** Create SQLite `events` table logging every retrieval interaction.
* **Schema:** `query, timestamp, strategy, ranked_results (doc_id, position, dense_score, bm25_score), cited_docs, feedback`.
* **Requirement:** Position logging is non-negotiable to correct for presentation bias.

### Phase 4: Memory Write-Back (T1) with Lineage Guards
* **Goal:** Automatically index high-confidence, verified answers as derived documents.
* **Guards:**
  1. *Sole Citation Rule:* A derived document can never be the sole citation for an answer.
  2. *Lineage Tracking:* Chunks carry hashes of parent documents. Stale parents trigger invalidation.
  3. *Holdout Isolation:* `eval.py --exclude-derived` on holdout must equal `eval.py` on holdout (zero contamination).

### Phase 5: Automated Scalar Optimization (T2)
* **Goal:** Tune $\alpha$, $k$, refusal threshold, and top-k retrieval count using random search (200 trials) against the `tune` split.
* **Constraints:** Must maximize recall subject to $\text{false\_refusal} \le 0.05$ and $\text{refusal\_accuracy} \ge 0.90$.
* **Output:** Writes best parameters to `params.json`, read dynamically by the app.

### Phase 6: Cross-Encoder Reranking (CPU-Constrained)
* **Goal:** Re-rank top 15 fused candidates using `cross-encoder/ms-marco-MiniLM-L6-v2`.
* **Gate:** Must show statistical improvement ($p < 0.05$) while keeping p95 latency under 300 ms on CPU.

### Phase 7: Adaptive Query Strategy Classifier (T3)
* **Goal:** Select query transformation strategy based on query structure (raw vs decompose vs keyword extract).
* **Implementation:** Deterministic rule-based routing first, with event logging preparing future bandit updates.

### Phase 8: Gap-Driven Autonomous Crawling (T0 Loop Closure)
* **Goal:** When a query triggers refusal or negative feedback, route it to a `frontier` queue.
* **Loop:** Search discovery $\to$ fetch/extract $\to$ ingest $\to$ re-eval. Successfully answered frontier questions are logged as mined test cases.

---

## 5. First Milestone Commands

```powershell
# 1. Switch to Phase 0 branch
git switch -c self-learning-p0

# 2. Run initial baseline evaluation
python eval.py

# 3. Begin Phase 0 edits:
#    - Add Wilson CI calculation to eval.py
#    - Expand evalset.jsonl to >= 120 items
```
