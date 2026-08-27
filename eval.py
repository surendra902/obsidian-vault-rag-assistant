"""Retrieval eval over evalset.jsonl.

Prints recall@5 (answerable), refusal accuracy (unanswerable, by threshold),
per-miss detail, and the top-hit score distributions that justify the refusal
threshold. Exit code 1 on regression below the floors recorded in README.

    python eval.py [--k 5] [--threshold 0.40]
"""

import argparse
import json
import sys
from collections import Counter

from rag import REFUSE_THRESHOLD, VaultIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=REFUSE_THRESHOLD)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open("evalset.jsonl", encoding="utf-8") if l.strip()]
    answerable = [c for c in cases if not c.get("unanswerable")]
    unanswerable = [c for c in cases if c.get("unanswerable")]

    idx = VaultIndex("vectors")

    # --- recall@k on answerable cases ---
    misses, ans_scores = [], []
    hits_count = 0
    for c in answerable:
        hits = idx.search(c["q"], k=args.k)
        ans_scores.append(hits[0].score)
        paths = [h.path for h in hits]
        if c["expect_note"] in paths:
            hits_count += 1
        else:
            misses.append((c["q"], c["expect_note"], paths[:3]))

    recall = hits_count / len(answerable)

    # --- refusal accuracy on unanswerable cases ---
    refused, unans_scores = 0, []
    for c in unanswerable:
        top = idx.search(c["q"], k=1)[0]
        unans_scores.append(top.score)
        if top.score < args.threshold:
            refused += 1
    refusal_acc = refused / len(unanswerable)

    # --- false-refusal rate on answerable cases (top hit below threshold) ---
    accepted = sum(1 for s in ans_scores if s >= args.threshold)
    false_refusal = 1 - accepted / len(answerable)

    # --- threshold justification: score distributions ---
    dist = lambda xs: f"min={min(xs):.3f} median={sorted(xs)[len(xs)//2]:.3f} max={max(xs):.3f}"

    print(f"cases: {len(answerable)} answerable, {len(unanswerable)} unanswerable, k={args.k}")
    print(f"recall@{args.k}: {recall:.3f} ({hits_count}/{len(answerable)})")
    print(f"refusal accuracy @threshold {args.threshold}: {refusal_acc:.3f} ({refused}/{len(unanswerable)})")
    print(f"false-refusal on answerable @threshold {args.threshold}: {false_refusal:.3f} ({len(answerable) - accepted}/{len(answerable)})")
    print(f"answerable top-score distribution:   {dist(ans_scores)}")
    print(f"unanswerable top-score distribution: {dist(unans_scores)}")
    if misses:
        print(f"\nmisses ({len(misses)}):")
        for q, expect, got in misses:
            print(f"  Q: {q[:70]}")
            print(f"    expected: {expect}")
            print(f"    got:      {', '.join(got)}")
    # ponytail: floors from the recorded baseline in README -- bump them when
    # the baseline improves, never lower them to make a regression pass.
    ok = recall >= 0.85 and refusal_acc >= 0.80
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
