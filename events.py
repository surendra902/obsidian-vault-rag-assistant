"""Display-time event logging system for self-learning RAG.

Logs query events, displayed rankings with explicit display positions (0..n-1)
to correct for presentation bias (Joachims et al. 2016), cited sources, and feedback.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rag import Hit

EVENTS_DB_PATH = "vectors/events.db"


def get_events_db(db_path: str = EVENTS_DB_PATH) -> sqlite3.Connection:
    """Connect to and initialize the events database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
            query_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            params_version TEXT NOT NULL,
            refused INTEGER NOT NULL,
            response_text TEXT,
            explicit_feedback INTEGER
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS displayed_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            path TEXT NOT NULL,
            heading TEXT NOT NULL,
            dense_score REAL,
            bm25_score REAL,
            fused_score REAL,
            cited INTEGER NOT NULL,
            FOREIGN KEY (query_id) REFERENCES queries(query_id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rankings_query_id ON displayed_rankings(query_id);")
    conn.commit()
    return conn


class EventLogger:
    def __init__(self, db_path: str = EVENTS_DB_PATH):
        self.db_path = db_path
        self.conn = get_events_db(db_path)

    def log_query_event(
        self,
        query: str,
        displayed_hits: List[Hit],
        strategy_id: str = "hybrid",
        params_version: str = "1.0",
        refused: bool = False,
        response_text: str = "",
        cited_paths: Optional[List[str]] = None,
        query_id: Optional[str] = None
    ) -> str:
        """Log a displayed query event along with its entire ranked list and positions.
        Returns the unique query_id.
        """
        qid = query_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        cited_set = set(cited_paths or [])

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO queries (query_id, query, timestamp, strategy_id, params_version, refused, response_text, explicit_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (qid, query, now, strategy_id, params_version, 1 if refused else 0, response_text)
        )

        for pos, hit in enumerate(displayed_hits):
            is_cited = 1 if hit.path in cited_set else 0
            dense_s = getattr(hit, "score", 0.0)
            fused_s = getattr(hit, "rrf_score", dense_s)
            cur.execute(
                """
                INSERT INTO displayed_rankings (query_id, position, path, heading, dense_score, bm25_score, fused_score, cited)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (qid, pos, hit.path, hit.heading, dense_s, 0.0, fused_s, is_cited)
            )

        self.conn.commit()
        return qid

    def record_feedback(self, query_id: str, feedback: int) -> bool:
        """Record explicit user feedback (+1 for 👍, -1 for 👎)."""
        cur = self.conn.cursor()
        cur.execute("UPDATE queries SET explicit_feedback = ? WHERE query_id = ?", (feedback, query_id))
        self.conn.commit()
        return cur.rowcount > 0

    def get_query_event(self, query_id: str) -> Optional[Dict]:
        """Fetch a full query event with its displayed rankings."""
        cur = self.conn.cursor()
        cur.execute("SELECT query, timestamp, strategy_id, params_version, refused, response_text, explicit_feedback FROM queries WHERE query_id = ?", (query_id,))
        row = cur.fetchone()
        if not row:
            return None

        q_info = {
            "query_id": query_id,
            "query": row[0],
            "timestamp": row[1],
            "strategy_id": row[2],
            "params_version": row[3],
            "refused": bool(row[4]),
            "response_text": row[5],
            "explicit_feedback": row[6],
            "displayed": []
        }

        cur.execute("SELECT position, path, heading, dense_score, bm25_score, fused_score, cited FROM displayed_rankings WHERE query_id = ? ORDER BY position ASC", (query_id,))
        for r in cur.fetchall():
            q_info["displayed"].append({
                "position": r[0],
                "path": r[1],
                "heading": r[2],
                "dense_score": r[3],
                "bm25_score": r[4],
                "fused_score": r[5],
                "cited": bool(r[6])
            })
        return q_info


def run_self_test():
    """Verify event logging and position fidelity."""
    print("=== Running EventLogger Self-Test ===")
    logger = EventLogger("vectors/events.db")

    sample_hits = [
        Hit(path="obsidian/zettelkasten.md", heading="Three notes", text="Fleeting notes...", tags=["obsidian"], score=0.75),
        Hit(path="books/how-to-take-smart-notes.md", heading="Method", text="Ahrens note...", tags=["books"], score=0.62),
        Hit(path="productivity/para-method.md", heading="PARA", text="Tiago Forte...", tags=["productivity"], score=0.45),
    ]
    setattr(sample_hits[0], "rrf_score", 0.032)
    setattr(sample_hits[1], "rrf_score", 0.024)
    setattr(sample_hits[2], "rrf_score", 0.018)

    qid = logger.log_query_event(
        query="Explain permanent note taking",
        displayed_hits=sample_hits,
        strategy_id="hybrid",
        params_version="1.0",
        refused=False,
        response_text="Permanent notes are written as...",
        cited_paths=["obsidian/zettelkasten.md"]
    )
    print(f"Logged query event: {qid}")

    # Record feedback
    ok = logger.record_feedback(qid, feedback=1)
    assert ok, "Failed to record feedback"

    # Retrieve and verify positions are dense (0..n-1)
    event = logger.get_query_event(qid)
    assert event is not None
    assert len(event["displayed"]) == 3
    for i, item in enumerate(event["displayed"]):
        assert item["position"] == i, f"Expected position {i}, got {item['position']}"
    assert event["displayed"][0]["cited"] is True
    assert event["displayed"][1]["cited"] is False
    assert event["explicit_feedback"] == 1

    print("Position density (0..n-1) and feedback verified: PASS")
    print("=== EventLogger Self-Test Passed Successfully ===")


if __name__ == "__main__":
    run_self_test()
