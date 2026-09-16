"""Retrieval eval over evalset.jsonl with Wilson Confidence Intervals and Split support.

Prints recall@k (answerable), refusal accuracy (unanswerable, by threshold),
per-miss detail, and the top-hit score distributions with 95% Wilson CIs.

Usage:
    python eval.py [--k 5] [--threshold 0.27] [--split tune|holdout|all] [--exclude-derived]
    python eval.py --compare baseline.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

from rag import REFUSE_THRESHOLD, VaultIndex


def wilson_ci(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = hits / total
    denom = 1.0 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z ** 2) / (4 * (total ** 2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


def mcnemar_test(b: int, c: int) -> float:
    """Exact two-sided binomial test for McNemar's test when b + c is small."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-tailed exact binomial test with p = 0.5
    prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval recall and refusal accuracy.")
    ap.add_argument("--k", type=int, default=5, help="Top-k hits to consider for recall")
    ap.add_argument("--threshold", type=float, default=REFUSE_THRESHOLD, help="Refusal similarity threshold")
    ap.add_argument("--split", choices=["tune", "holdout", "all"], default="tune",
                    help="Which split of evalset.jsonl to run on (default: tune)")
    ap.add_argument("--exclude-derived", action="store_true",
                    help="Exclude derived memory chunks (used to check eval contamination)")
    ap.add_argument("--save-run", type=str, default=None,
                    help="Save query-level results to a JSON file for future diffs")
    ap.add_argument("--compare", type=str, default=None,
                    help="Compare current run against a saved run JSON using McNemar's test")
    args = ap.parse_args()

    all_cases = [json.loads(line) for line in open("evalset.jsonl", encoding="utf-8") if line.strip()]

    # Filter by split if specified
    if args.split == "all":
        cases = all_cases
    else:
        cases = [c for c in all_cases if c.get("split") == args.split or (c.get("unanswerable") and args.split != "holdout")]
        # Also ensure unanswerables are included in holdout if needed or kept separate
        if args.split == "holdout":
            cases = [c for c in all_cases if c.get("split") == "holdout" or c.get("unanswerable")]

    answerable = [c for c in cases if not c.get("unanswerable")]
    unanswerable = [c for c in cases if c.get("unanswerable")]

    idx = VaultIndex("vectors")

    # --- recall@k on answerable cases ---
    misses, ans_scores = [], []
    hits_count = 0
    query_results = {}

    for c in answerable:
        hits = idx.search(c["q"], k=args.k, exclude_derived=args.exclude_derived)
        top_score = hits[0].score if hits else 0.0
        ans_scores.append(top_score)
        paths = [h.path for h in hits]
        hit_success = c["expect_note"] in paths
        query_results[c["q"]] = {
            "hit": hit_success,
            "expect": c["expect_note"],
            "retrieved": paths[:args.k],
            "top_score": top_score,
            "split": c.get("split", "tune"),
            "kind": c.get("kind", "standard")
        }

        if hit_success:
            hits_count += 1
        else:
            misses.append((c["q"], c["expect_note"], paths[:3], c.get("kind", "standard")))

    recall = hits_count / len(answerable) if answerable else 0.0
    recall_ci = wilson_ci(hits_count, len(answerable))

    # --- refusal accuracy on unanswerable cases ---
    refused, unans_scores = 0, []
    for c in unanswerable:
        top_hits = idx.search(c["q"], k=1, exclude_derived=args.exclude_derived)
        top = top_hits[0] if top_hits else None
        score = top.score if top else 0.0
        unans_scores.append(score)
        if score < args.threshold:
            refused += 1

    refusal_acc = refused / len(unanswerable) if unanswerable else 1.0
    refusal_ci = wilson_ci(refused, len(unanswerable))

    # --- false-refusal rate on answerable cases (top hit below threshold) ---
    accepted = sum(1 for s in ans_scores if s >= args.threshold)
    false_refused_count = len(answerable) - accepted
    false_refusal = false_refused_count / len(answerable) if answerable else 0.0
    false_refusal_ci = wilson_ci(false_refused_count, len(answerable))

    # --- threshold justification: score distributions ---
    dist = lambda xs: f"min={min(xs):.3f} median={sorted(xs)[len(xs)//2]:.3f} max={max(xs):.3f}" if xs else "N/A"

    print(f"=== Retrieval Evaluation [Split: {args.split.upper()}] ===")
    print(f"Cases: {len(answerable)} answerable, {len(unanswerable)} unanswerable, k={args.k}")
    print(f"recall@{args.k}:                        {recall:.3f} ({hits_count}/{len(answerable)}) [95% CI: {recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")
    print(f"refusal accuracy @threshold {args.threshold}:   {refusal_acc:.3f} ({refused}/{len(unanswerable)}) [95% CI: {refusal_ci[0]:.3f}, {refusal_ci[1]:.3f}]")
    print(f"false-refusal on answerable:          {false_refusal:.3f} ({false_refused_count}/{len(answerable)}) [95% CI: {false_refusal_ci[0]:.3f}, {false_refusal_ci[1]:.3f}]")
    if ans_scores:
        print(f"answerable top-score distribution:    {dist(ans_scores)}")
    if unans_scores:
        print(f"unanswerable top-score distribution:  {dist(unans_scores)}")

    if misses:
        print(f"\nmisses ({len(misses)}):")
        for q, expect, got, kind in misses[:10]:
            print(f"  [{kind}] Q: {q[:70]}")
            print(f"    expected: {expect}")
            print(f"    got:      {', '.join(got)}")
        if len(misses) > 10:
            print(f"  ... and {len(misses) - 10} more misses.")

    # --- McNemar's comparison if requested ---
    if args.compare and Path(args.compare).is_file():
        baseline_data = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        base_results = baseline_data.get("queries", {})
        common_keys = set(query_results.keys()) & set(base_results.keys())
        b = sum(1 for q in common_keys if not base_results[q]["hit"] and query_results[q]["hit"])
        c = sum(1 for q in common_keys if base_results[q]["hit"] and not query_results[q]["hit"])
        p_val = mcnemar_test(b, c)
        print(f"\n=== McNemar Comparison vs {args.compare} ===")
        print(f"Common cases: {len(common_keys)}")
        print(f"Wins (improved from miss -> hit):   {b}")
        print(f"Regressions (dropped hit -> miss):  {c}")
        print(f"McNemar p-value: {p_val:.4f} ({'Statistically significant (p < 0.05)' if p_val < 0.05 else 'Not statistically significant'})")

    if args.save_run:
        save_path = Path(args.save_run)
        save_data = {
            "split": args.split,
            "k": args.k,
            "threshold": args.threshold,
            "recall": recall,
            "recall_ci": list(recall_ci),
            "refusal_acc": refusal_acc,
            "queries": query_results,
        }
        save_path.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
        print(f"\nRun saved to: {save_path}")

    # Exit code: on tune split, recall must be within headroom range [0.60, 0.90] or pass floor >= 0.60
    # On holdout split, recall must not regress below floor
    floor_recall = 0.60 if args.split in ("tune", "holdout") else 0.70
    floor_refusal = 0.80
    ok = recall >= floor_recall and refusal_acc >= floor_refusal
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
