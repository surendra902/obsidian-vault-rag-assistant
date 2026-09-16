"""Hybrid search combining SQLite FTS5 BM25 lexical search with dense vector embeddings.

Uses Reciprocal Rank Fusion (RRF); k is loaded from params.json (tuned, not
hardcoded -- see tune.py):
    RRF(d) = sum_r 1 / (k + rank_r(d))

Grounded Refusal:
    Refusal threshold checking is decoupled from RRF rank fractions, using the
    calibrated dense similarity to avoid false refusals.
"""

import hashlib
import json
import re
import sqlite3
import threading
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
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")
        self._ensure_index()

    def _ensure_index(self):
        """Create and populate FTS5 table if missing or outdated."""
        with self._lock:
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
            hits.append(Hit(path=path, heading=heading, text=text, tags=tags, score=score, bm25_score=score))
        return hits

    def close(self):
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass


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
        alpha: Optional[float] = None,
        exclude_derived: bool = False,
        candidate_depth: int = 50
    ) -> List[Hit]:
        """Execute search in 'dense', 'bm25', or 'hybrid' mode.
        
        Refusal Preservation:
        In hybrid mode, returned hits carry both the fused ranking score (Hit.score)
        and the dense similarity (Hit.dense_score) for calibrated refusal thresholding.
        """
        if mode == "dense":
            return self.dense.search(query, k=k, exclude_derived=exclude_derived)
        elif mode == "bm25":
            return self.bm25.search(query, k=k)

        # Hybrid: retrieve candidates from both arms
        effective_alpha = alpha if alpha is not None else self.alpha
        dense_hits = self.dense.search(query, k=candidate_depth, exclude_derived=exclude_derived)
        bm25_hits = self.bm25.search(query, k=candidate_depth)

        rrf_scores = {}
        item_map = {}
        dense_score_map = {}
        bm25_score_map = {}

        # Dense ranks
        for rank, hit in enumerate(dense_hits):
            text_hash = hashlib.sha256(hit.text.encode("utf-8")).hexdigest()[:16]
            key = (hit.path, hit.heading, text_hash)
            item_map[key] = hit
            dense_score_map[key] = getattr(hit, "dense_score", hit.score)
            bm25_score_map[key] = getattr(hit, "bm25_score", 0.0)
            rrf_scores[key] = effective_alpha * (1.0 / (self.k_rrf + rank + 1))

        # BM25 ranks
        for rank, hit in enumerate(bm25_hits):
            text_hash = hashlib.sha256(hit.text.encode("utf-8")).hexdigest()[:16]
            key = (hit.path, hit.heading, text_hash)
            if key not in item_map:
                item_map[key] = hit
                dense_score_map[key] = 0.0
            bm25_score_map[key] = getattr(hit, "bm25_score", hit.score)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 - effective_alpha) * (1.0 / (self.k_rrf + rank + 1))

        # Sort by fused score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:k]

        fused_hits = []
        for key in sorted_keys:
            base_hit = item_map[key]
            fused_score = rrf_scores[key]
            preserved_dense = dense_score_map.get(key, 0.0)
            preserved_bm25 = bm25_score_map.get(key, 0.0)
            hit = Hit(
                path=base_hit.path,
                heading=base_hit.heading,
                text=base_hit.text,
                tags=base_hit.tags,
                score=fused_score,
                dense_score=preserved_dense,
                bm25_score=preserved_bm25,
                source_url=getattr(base_hit, "source_url", ""),
                provenance=getattr(base_hit, "provenance", "primary"),
                ingested_at=getattr(base_hit, "ingested_at", ""),
                derived_from_hashes=getattr(base_hit, "derived_from_hashes", None)
            )
            setattr(hit, "rrf_score", fused_score)
            fused_hits.append(hit)

        return fused_hits

    def read_note(self, path: str) -> str:
        """Forward to dense VaultIndex read_note with traversal guards."""
        return self.dense.read_note(path)

    def notes_by_tag(self, tag: str) -> list:
        """Forward to dense VaultIndex notes_by_tag."""
        return self.dense.notes_by_tag(tag)

    @property
    def stats(self):
        """Index statistics for health check."""
        base_stats = dict(self.dense.stats)
        base_stats["mode"] = "hybrid (FTS5 BM25 + dense)"
        base_stats["alpha"] = self.alpha
        base_stats["k_rrf"] = self.k_rrf
        return base_stats

    def close(self):
        if hasattr(self, "bm25") and self.bm25:
            self.bm25.close()
