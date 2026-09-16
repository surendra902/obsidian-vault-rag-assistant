"""Hybrid search combining SQLite FTS5 BM25 lexical search with dense vector embeddings.

Uses Reciprocal Rank Fusion (RRF) with k=60:
    RRF(d) = sum_r 1 / (k + rank_r(d))

Grounded Refusal:
    Refusal threshold checking is decoupled from RRF rank fractions, using the
    calibrated dense similarity to avoid false refusals.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rag import REFUSE_THRESHOLD, Hit, VaultIndex

FTS_DB_PATH = "vectors/fts5.db"


def sanitize_fts5_query(query: str) -> str:
    """Sanitize user query for SQLite FTS5 MATCH syntax.
    Extract alphanumeric words and join with OR/AND.
    """
    words = re.findall(r"\w+", query)
    if not words:
        return ""
    # Search terms with prefix matching
    terms = [f'"{w}"*' for w in words]
    return " OR ".join(terms)


class FTSIndex:
    def __init__(self, db_path: str = FTS_DB_PATH, chunks_path: str = "vectors/chunks.jsonl"):
        self.db_path = Path(db_path)
        self.chunks_path = Path(chunks_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self._ensure_index()

    def _ensure_index(self):
        """Create and populate FTS5 table if missing or outdated."""
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                path,
                heading,
                text,
                tags,
                tokenize = 'porter unicode61'
            );
            """
        )
        cur.execute("SELECT count(*) FROM chunks_fts")
        count = cur.fetchone()[0]

        chunks = [json.loads(line) for line in self.chunks_path.read_text(encoding="utf-8").splitlines() if line]
        if count != len(chunks):
            cur.execute("DELETE FROM chunks_fts")
            for idx, c in enumerate(chunks):
                cur.execute(
                    "INSERT INTO chunks_fts (chunk_id, path, heading, text, tags) VALUES (?, ?, ?, ?, ?)",
                    (idx, c["path"], c["heading"], c["text"], " ".join(c.get("tags", [])))
                )
            self.conn.commit()

    def search(self, query: str, k: int = 50) -> List[Hit]:
        """Perform BM25 search over chunks."""
        fts_query = sanitize_fts5_query(query)
        if not fts_query:
            return []

        cur = self.conn.cursor()
        try:
            cur.execute(
                """
                SELECT chunk_id, path, heading, text, tags, bm25(chunks_fts) as score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (fts_query, k)
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []

        hits = []
        for row in rows:
            chunk_id, path, heading, text, tags_str, bm25_score = row
            # bm25 in sqlite returns negative scores where lower (more negative) is better match
            # invert or normalize for reporting
            score = -float(bm25_score)
            tags = tags_str.split() if tags_str else []
            hits.append(Hit(path=path, heading=heading, text=text, tags=tags, score=score))
        return hits


class HybridIndex:
    def __init__(self, index_dir: str = "vectors", k_rrf: Optional[int] = None, alpha: Optional[float] = None, params_path: str = "params.json"):
        self.dense = VaultIndex(index_dir)
        self.bm25 = FTSIndex(db_path=f"{index_dir}/fts5.db", chunks_path=f"{index_dir}/chunks.jsonl")
        
        # Load tuned parameters if available
        params = {}
        p = Path(params_path)
        if p.is_file():
            try:
                params = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

        self.alpha = alpha if alpha is not None else params.get("alpha", 0.8)
        self.k_rrf = k_rrf if k_rrf is not None else params.get("k_rrf", 40)
        self.threshold = params.get("threshold", REFUSE_THRESHOLD)
        self.default_k = params.get("k", 5)
        self.candidate_depth = params.get("candidate_depth", 50)

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        exclude_derived: bool = False,
        candidate_depth: int = 50
    ) -> List[Hit]:
        """Execute search in 'dense', 'bm25', or 'hybrid' mode.
        
        Refusal Preservation:
        In hybrid mode, the top returned hit maintains its dense score to allow
        calibrated refusal thresholding.
        """
        if mode == "dense":
            return self.dense.search(query, k=k, exclude_derived=exclude_derived)
        elif mode == "bm25":
            return self.bm25.search(query, k=k)

        # Hybrid: retrieve candidates from both arms
        dense_hits = self.dense.search(query, k=candidate_depth, exclude_derived=exclude_derived)
        bm25_hits = self.bm25.search(query, k=candidate_depth)

        rrf_scores = {}
        item_map = {}
        dense_score_map = {}

        # Dense ranks
        for rank, hit in enumerate(dense_hits):
            key = (hit.path, hit.heading, hit.text[:100])
            item_map[key] = hit
            dense_score_map[key] = hit.score
            rrf_scores[key] = self.alpha * (1.0 / (self.k_rrf + rank + 1))

        # BM25 ranks
        for rank, hit in enumerate(bm25_hits):
            key = (hit.path, hit.heading, hit.text[:100])
            if key not in item_map:
                item_map[key] = hit
                dense_score_map[key] = 0.0
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 - self.alpha) * (1.0 / (self.k_rrf + rank + 1))

        # Sort by fused score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:k]

        fused_hits = []
        for key in sorted_keys:
            base_hit = item_map[key]
            # Carry forward dense score for refusal threshold checks, fused score as metadata
            fused_score = rrf_scores[key]
            # To preserve calibrated refusal thresholding, top hit's score reflects dense similarity
            preserved_dense = dense_score_map.get(key, 0.0)
            hit = Hit(
                path=base_hit.path,
                heading=base_hit.heading,
                text=base_hit.text,
                tags=base_hit.tags,
                score=preserved_dense  # preserves refusal thresholding capability
            )
            # Attach rrf_score attribute
            setattr(hit, "rrf_score", fused_score)
            fused_hits.append(hit)

        return fused_hits
