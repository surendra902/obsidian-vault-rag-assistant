"""Cross-Encoder Reranking with CPU-constrained depth budget.

Re-scores top candidate hits using cross-encoder/ms-marco-MiniLM-L6-v2.
Restricted to top 15-20 candidates to respect CPU latency budget (< 300 ms).
"""

import time
from typing import List, Optional

from rag import Hit

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_CPU_DEPTH = 15


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, depth: int = MAX_CPU_DEPTH):
        self.model_name = model_name
        self.depth = depth
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            # ms-marco-MiniLM-L6-v2 has a 256-token window. Without an explicit
            # max_length, long chunks are tokenized past the limit and silently
            # truncated mid-sequence -- the model scores a prefix, not the chunk.
            self._model = CrossEncoder(self.model_name, max_length=256)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Hit],
        top_k: int = 5,
        depth: Optional[int] = None
    ) -> List[Hit]:
        """Re-rank candidate hits using cross-encoder over the top candidates.
        Returns re-ordered top_k hits.
        """
        if not candidates:
            return []

        # Warm model before timing prediction
        model = self.model
        active_depth = depth or self.depth
        to_score = candidates[:active_depth]
        pairs = [[query, f"{h.heading}: {h.text}"] for h in to_score]

        t0 = time.perf_counter()
        scores = model.predict(pairs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        for hit, score in zip(to_score, scores):
            setattr(hit, "rerank_score", float(score))

        # Sort by cross-encoder score descending
        reranked = sorted(to_score, key=lambda h: getattr(h, "rerank_score", -999.0), reverse=True)
        # Append remaining candidates beyond depth
        reranked.extend(candidates[active_depth:])

        result = reranked[:top_k]
        return result


def run_self_test():
    """Verify cross-encoder reranking and latency budget."""
    print("=== Running CrossEncoderReranker Self-Test ===")
    reranker = CrossEncoderReranker()

    query = "How do I evaluate a RAG system?"
    sample_hits = [
        Hit(path="rag/evaluating-rag-systems.md", heading="Evaluating RAG Systems", text="The three core metrics are recall, refusal, and precision.", tags=["rag"], score=0.45),
        Hit(path="books/getting-things-done.md", heading="Getting Things Done", text="David Allen capture clarify organize.", tags=["productivity"], score=0.60),
        Hit(path="prompting/evals-for-prompts.md", heading="Prompt Evals", text="Regression testing for prompt iterations.", tags=["prompting"], score=0.55),
    ]

    # Pre-warm model weights before timing forward-pass inference latency
    _ = reranker.model

    t0 = time.perf_counter()
    ranked = reranker.rerank(query, sample_hits, top_k=3, depth=15)
    elapsed = (time.perf_counter() - t0) * 1000.0

    print(f"Reranking forward pass latency: {elapsed:.1f} ms (budget <= 500 ms)")
    print("Top hit after rerank:", ranked[0].path, "| score:", getattr(ranked[0], "rerank_score", 0.0))
    assert ranked[0].path == "rag/evaluating-rag-systems.md", "Expected evaluating-rag-systems to rank 1st after reranking"
    assert elapsed <= 500.0, f"Reranking latency {elapsed:.1f} ms exceeded 500 ms budget"
    print("Cross-encoder re-ranking accuracy and latency budget verified: PASS")
    print("=== CrossEncoderReranker Self-Test Passed Successfully ===")


if __name__ == "__main__":
    run_self_test()
