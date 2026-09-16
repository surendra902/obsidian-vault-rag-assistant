"""Gap-driven autonomous crawling and frontier queue (T0).

Closes the self-learning loop:
1. Refused or low-confidence queries are automatically enqueued to a frontier queue.
2. Web ingestion targets are fetched, cleaned with Trafilatura, and indexed into canonical chunks.
3. Upon successful resolution, the query is automatically added to evalset.jsonl as a
   'mined' test case in the tune split (never the frozen holdout), hardening the eval.
"""

import json
import sqlite3
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
    def __init__(self, db_path: str = FRONTIER_DB_PATH, evalset_path: str = "evalset.jsonl"):
        self.db_path = db_path
        self.evalset_path = Path(evalset_path)
        self.conn = get_frontier_db(db_path)
        self.ingest_engine = IngestionEngine("vectors")
        self.hybrid_index = HybridIndex("vectors")

    def enqueue_gap(self, query: str) -> bool:
        """Enqueue a failed or refused query as a knowledge gap to investigate."""
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
        # Reload search index to see new chunks
        self.hybrid_index = HybridIndex("vectors")
        hits = self.hybrid_index.search(query, k=5, mode="hybrid")
        top_score = hits[0].score if hits else 0.0

        # Successful if retrieved with confidence
        if hits and top_score >= self.hybrid_index.threshold:
            now = datetime.now(timezone.utc).isoformat()
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE frontier SET status = 'resolved', target_urls = ?, resolved_at = ? WHERE query = ?",
                (source_url, now, query.strip())
            )
            self.conn.commit()

            # 3. Auto-append to evalset.jsonl as mined case in TUNE split (Never holdout)
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

            print(f"Gap resolved & mined into tune eval set: '{query}' -> {hits[0].path}")
            return True

        return False


def run_self_test():
    """Verify frontier gap enqueuing, resolution, and evalset promotion."""
    print("=== Running FrontierManager Self-Test ===")
    frontier = FrontierManager("vectors/frontier.db", "evalset.jsonl")

    test_query = "What is the architecture of an asynchronous actor model in distributed systems?"
    
    # 1. Enqueue gap
    enqueued = frontier.enqueue_gap(test_query)
    print(f"Gap enqueued: {enqueued}")
    pending = frontier.get_pending_gaps()
    assert any(g["query"] == test_query for g in pending), "Expected test query in pending gaps"
    print("Frontier enqueuing verified: PASS")

    # 2. Resolve gap with targeted document ingest
    doc_title = "Distributed Actor Model Architecture"
    doc_content = """
    # Distributed Actor Model Architecture
    
    The asynchronous actor model treats actors as universal primitives of concurrent computation.
    Each actor encapsulates state and communicates exclusively through asynchronous message passing.
    In distributed systems, mailboxes buffer incoming messages, and supervisors manage fault tolerance
    using 'let it crash' philosophies popularized by Erlang/OTP.
    """ * 3

    resolved = frontier.resolve_gap_with_content(
        query=test_query,
        title=doc_title,
        content=doc_content,
        source_url="https://distributed-actors.internal/spec"
    )
    assert resolved is True, "Expected gap resolution to succeed"
    print("Gap resolution and indexing verified: PASS")

    # 3. Verify evalset.jsonl was updated with 'mined' case in 'tune' split
    all_cases = [json.loads(line) for line in open("evalset.jsonl", encoding="utf-8") if line.strip()]
    mined_matches = [c for c in all_cases if c["q"] == test_query]
    assert len(mined_matches) >= 1, "Mined case was not appended to evalset.jsonl"
    assert mined_matches[0]["split"] == "tune", "Mined case must be in tune split, never holdout!"
    assert mined_matches[0]["source"] == "mined", "Mined case must be tagged source: mined"
    print("Autonomous evalset promotion into tune split verified: PASS")

    print("=== FrontierManager Self-Test Passed Successfully ===")


if __name__ == "__main__":
    run_self_test()
