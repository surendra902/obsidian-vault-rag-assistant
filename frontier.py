"""Gap-driven autonomous crawling and frontier queue (T0).

Closes the self-learning loop:
1. Refused or low-confidence queries are automatically enqueued to a frontier queue.
2. Web ingestion targets are fetched, cleaned with Trafilatura, and indexed into canonical chunks.
3. Upon successful resolution, the query is automatically added to evalset.jsonl as a
   'mined' test case in the tune split (never the frozen holdout), hardening the eval.
"""

import json
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ingest import IngestionEngine
from search import HybridIndex

FRONTIER_DB_PATH = "vectors/frontier.db"


def get_frontier_db(db_path: str = FRONTIER_DB_PATH) -> sqlite3.Connection:
    """Connect to and initialize the frontier queue database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS frontier (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            target_urls TEXT,
            resolved_at TEXT
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_frontier_status ON frontier(status);")
    conn.commit()
    return conn


class FrontierManager:
    def __init__(
        self,
        db_path: str = FRONTIER_DB_PATH,
        evalset_path: str = "evalset_mined.jsonl",
        index_dir: str = "vectors",
        vault_dir: str = "demo_vault"
    ):
        self.db_path = db_path
        self.evalset_path = Path(evalset_path)
        self.index_dir = index_dir
        self.vault_dir = vault_dir
        self._lock = threading.Lock()
        self.conn = get_frontier_db(db_path)
        self.ingest_engine = IngestionEngine(index_dir=index_dir, vault_dir=vault_dir)
        self.hybrid_index = HybridIndex(index_dir)

    def enqueue_gap(self, query: str) -> bool:
        """Enqueue a failed or refused query as a knowledge gap to investigate."""
        with self._lock:
            cur = self.conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            try:
                cur.execute(
                    "INSERT INTO frontier (query, timestamp, status, target_urls, resolved_at) VALUES (?, ?, 'pending', '', NULL)",
                    (query.strip(), now)
                )
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Query already enqueued
                return False

    def get_pending_gaps(self, limit: int = 20) -> List[Dict]:
        """Fetch unaddressed knowledge gaps."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, query, timestamp, status FROM frontier WHERE status = 'pending' ORDER BY id ASC LIMIT ?", (limit,))
            return [{"id": r[0], "query": r[1], "timestamp": r[2], "status": r[3]} for r in cur.fetchall()]

    def resolve_gap_with_content(
        self,
        query: str,
        title: str,
        content: str,
        source_url: str = "https://frontier.internal"
    ) -> bool:
        """Ingest knowledge to address a gap, verify retrieval, and promote to mined eval case."""
        # 1. Ingest new content
        new_chunks = self.ingest_engine.ingest_document(
            identifier=source_url,
            title=title,
            text=content,
            source_kind="frontier_crawl"
        )
        if new_chunks == 0:
            return False

        # 2. Re-verify retrieval using hybrid index
        self.hybrid_index = HybridIndex(self.index_dir)
        hits = self.hybrid_index.search(query, k=5, mode="hybrid")
        top_dense = max((getattr(h, "dense_score", h.score) for h in hits), default=0.0) if hits else 0.0

        # Successful if retrieved with confidence
        if hits and top_dense >= self.hybrid_index.threshold:
            now = datetime.now(timezone.utc).isoformat()
            with self._lock:
                cur = self.conn.cursor()
                cur.execute(
                    "UPDATE frontier SET status = 'resolved', target_urls = ?, resolved_at = ? WHERE query = ?",
                    (source_url, now, query.strip())
                )
                self.conn.commit()

            # 3. Auto-append to quarantined evalset_mined.jsonl (Never contaminate primary benchmark evalset)
            mined_case = {
                "q": query.strip(),
                "expect_note": hits[0].path,
                "split": "tune",
                "source": "mined",
                "kind": "gap_closed",
                "mined_at": now
            }
            with open(self.evalset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(mined_case, ensure_ascii=False) + "\n")

            print(f"Gap resolved & mined into quarantine eval set: '{query}' -> {hits[0].path}")
            return True

        return False

    def close(self):
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        if hasattr(self, "ingest_engine") and self.ingest_engine:
            self.ingest_engine.close()
        if hasattr(self, "hybrid_index") and self.hybrid_index:
            self.hybrid_index.close()


def run_self_test():
    """Verify frontier gap enqueuing, resolution, and evalset promotion in an isolated test harness."""
    print("=== Running FrontierManager Self-Test ===")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        test_dir = Path(temp_dir)
        test_db = str(test_dir / "frontier_test.db")
        test_mined = str(test_dir / "evalset_test_mined.jsonl")
        test_vectors = str(test_dir / "vectors")
        test_vault = str(test_dir / "vault")
        shutil.copytree("vectors", test_vectors)
        shutil.copytree("demo_vault", test_vault)

        frontier = FrontierManager(
            db_path=test_db,
            evalset_path=test_mined,
            index_dir=test_vectors,
            vault_dir=test_vault
        )

        test_query = "What is the architecture of an asynchronous actor model in distributed systems?"

        # 1. Enqueue gap
        enqueued = frontier.enqueue_gap(test_query)
        print(f"Gap enqueued: {enqueued}")
        assert enqueued is True, "Expected test query to be enqueued successfully"
        pending = frontier.get_pending_gaps()
        assert any(g["query"] == test_query for g in pending), "Expected test query in pending gaps"
        print("Frontier enqueuing verified: PASS")

        # 2. Resolve gap with targeted document ingest
        run_id = uuid.uuid4().hex[:6]
        doc_title = f"Distributed Actor Model Architecture {run_id}"
        doc_content = f"""
        # Distributed Actor Model Architecture {run_id}
        
        The asynchronous actor model treats actors as universal primitives of concurrent computation.
        Each actor encapsulates state and communicates exclusively through asynchronous message passing.
        In distributed systems, mailboxes buffer incoming messages, and supervisors manage fault tolerance
        using 'let it crash' philosophies popularized by Erlang/OTP.
        """ * 3
        test_url = f"https://distributed-actors.internal/spec-{run_id}"

        resolved = frontier.resolve_gap_with_content(
            query=test_query,
            title=doc_title,
            content=doc_content,
            source_url=test_url
        )
        assert resolved is True, "Expected gap resolution to succeed"
        print("Gap resolution and indexing verified: PASS")

        # 3. Verify quarantined mined file was updated
        all_cases = [json.loads(line) for line in open(test_mined, encoding="utf-8") if line.strip()]
        mined_matches = [c for c in all_cases if c["q"] == test_query]
        assert len(mined_matches) >= 1, "Mined case was not appended to test mined evalset"
        assert mined_matches[0]["split"] == "tune", "Mined case must be in tune split, never holdout!"
        assert mined_matches[0]["source"] == "mined", "Mined case must be tagged source: mined"
        print("Autonomous evalset promotion into quarantined mined file verified: PASS")
        frontier.close()

    print("=== FrontierManager Self-Test Passed Successfully ===")


if __name__ == "__main__":
    run_self_test()
