"""Adaptive Query Strategy Classifier & Router (T3).

Classifies queries into structural categories (exact identifier, conceptual, multi-part)
to route to the optimal retrieval configuration, with feedback logging for policy updates.
"""

import re
from typing import Dict, List, Tuple

from rag import Hit
from search import HybridIndex
from rerank import CrossEncoderReranker


class QueryClassifier:
    """Deterministic structural classifier for query routing."""

    @staticmethod
    def classify(query: str) -> str:
        q = query.strip()
        lower_q = q.lower()

        # Check for multi-part questions or comparisons first
        if any(w in lower_q for w in [" versus ", " vs ", " compare ", " difference between "]) or (lower_q.count("?") > 1):
            return "multi_part"

        # Check for exact identifiers: contains quotes, codes, version numbers, or math
        if re.search(r'["\']|\b\d+[\*x\/\+\-]\d+\b|\b\d{4}-\d{2}-\d{2}\b|\b[A-Z_]{3,}\b|\b\w+\.\w{1,4}\b', q):
            return "exact_identifier"

        # Short queries (< 4 words)
        words = q.split()
        if len(words) <= 3:
            return "short_lookup"

        return "conceptual_semantic"


class AdaptiveRouter:
    def __init__(self, hybrid_index: HybridIndex):
        self.idx = hybrid_index
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def route_and_search(self, query: str, k: int = 5) -> Tuple[List[Hit], str]:
        """Classify query, execute tailored strategy, and return (hits, strategy_id)."""
        q_class = QueryClassifier.classify(query)

        if q_class == "exact_identifier":
            # Biased toward BM25 lexical precision (alpha=0.2 gives 80% weight to BM25)
            hits = self.idx.search(query, k=k, mode="hybrid", alpha=0.2)
            return hits, "exact_bm25_heavy"

        elif q_class == "multi_part":
            # Decompose into sub-clauses and union results
            parts = re.split(r" versus | vs | and |\?|\,", query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip().split()) >= 2]
            if len(sub_queries) >= 2:
                all_hits = []
                seen = set()
                for sq in sub_queries[:2]:
                    sub_hits = self.idx.search(sq, k=k, mode="hybrid")
                    for h in sub_hits:
                        key = (h.path, h.heading)
                        if key not in seen:
                            seen.add(key)
                            all_hits.append(h)
                return all_hits[:k], "decomposed_union"
            else:
                return self.idx.search(query, k=k, mode="hybrid"), "standard_hybrid"

        elif q_class == "conceptual_semantic":
            # Retrieve candidates and rerank top-15 with neural cross-encoder
            raw_hits = self.idx.search(query, k=15, mode="hybrid")
            if raw_hits:
                reranked_hits = self.reranker.rerank(query, raw_hits, top_k=k)
                return reranked_hits, "reranked_cross_encoder"
            return [], "reranked_cross_encoder"

        else:
            # Standard tuned hybrid
            hits = self.idx.search(query, k=k, mode="hybrid")
            return hits, "standard_hybrid"


def run_self_test():
    """Verify query classification and routing strategies."""
    print("=== Running QueryClassifier & AdaptiveRouter Self-Test ===")
    
    # Classification tests
    assert QueryClassifier.classify("2026-08-25") == "exact_identifier"
    assert QueryClassifier.classify("MAX_SEQ_TOKENS assertion") == "exact_identifier"
    assert QueryClassifier.classify("System 1 versus System 2 thinking") == "multi_part"
    assert QueryClassifier.classify("deep work") == "short_lookup"
    assert QueryClassifier.classify("How does externalizing open loops create psychological clarity?") == "conceptual_semantic"
    print("Query classification patterns verified: PASS")

    # Routing execution test
    idx = HybridIndex("vectors")
    router = AdaptiveRouter(idx)
    hits, strategy = router.route_and_search("17*24 Kahneman multiplication", k=3)
    print(f"Routed strategy: {strategy} -> Top hit: {hits[0].path}")
    assert strategy == "exact_bm25_heavy"
    assert hits[0].path == "books/thinking-fast-and-slow.md"

    hits_multi, strategy_multi = router.route_and_search("Zettelkasten method versus PARA framework", k=3)
    print(f"Routed strategy: {strategy_multi} -> Hits count: {len(hits_multi)}")
    assert strategy_multi == "decomposed_union"

    hits_conc, strategy_conc = router.route_and_search("How does externalizing open loops create psychological clarity?", k=3)
    print(f"Routed strategy: {strategy_conc} -> Top hit: {hits_conc[0].path}")
    assert strategy_conc == "reranked_cross_encoder"
    assert len(hits_conc) > 0

    print("Adaptive routing execution verified: PASS")
    print("=== QueryClassifier & AdaptiveRouter Self-Test Passed Successfully ===")


if __name__ == "__main__":
    run_self_test()
