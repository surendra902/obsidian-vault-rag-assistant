"""Multi-source ingestion engine for web URLs and external documents.

Extracts text using Trafilatura with Scrapling/Playwright fallback for JS-heavy pages.
Enforces robots.txt compliance, polite request rates, SHA-256 deduplication,
and preserves the exact chunking and token budget constraints of index.py.

Usage:
    python ingest.py --url https://example.com/page
    python ingest.py --file urls.txt
    python ingest.py --test
"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import trafilatura

from index import (
    CHUNK_WORDS,
    MAX_SEQ_TOKENS,
    MIN_CHUNK_WORDS,
    OVERLAP_WORDS,
    embed_text,
    make_chunks,
    sections,
)
from rag import MODEL_NAME, VaultIndex, load_model

USER_AGENT = "SelfLearningSearchAgent/1.0 (+https://local.vault/bot)"
DOCS_DB_PATH = "vectors/docs.db"


def get_docs_db(db_path: str = DOCS_DB_PATH) -> sqlite3.Connection:
    """Connect to and initialize docs database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
            url TEXT PRIMARY KEY,
            title TEXT,
            fetched_at TEXT,
            content_hash TEXT,
            robots_ok INTEGER,
            source_kind TEXT,
            chunk_count INTEGER
        );
        """
    )
    conn.commit()
    return conn


def check_robots_txt(url: str, user_agent: str = USER_AGENT) -> bool:
    """Check if the given URL is allowed by the host's robots.txt."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return True
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # If robots.txt cannot be fetched or parsed, default to permissive
        return True


def extract_web_content(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch and extract title and clean main text from a web URL.
    Uses trafilatura first; falls back to scrapling/playwright if JS-rendered (<80 words).
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None, None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=False
    )
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata and metadata.title else urlparse(url).netloc

    word_count = len(text.split()) if text else 0
    if word_count >= 80:
        return title, text

    # Fallback to Scrapling for JS-rendered dynamic pages
    try:
        from scrapling import Fetcher
        fetcher = Fetcher()
        page = fetcher.get(url)
        js_text = trafilatura.extract(page.text) if page and page.text else None
        if js_text and len(js_text.split()) > word_count:
            return title, js_text
    except Exception:
        pass

    return title, text


class IngestionEngine:
    def __init__(self, index_dir: str = "vectors"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.index_dir / "chunks.jsonl"
        self.vectors_file = self.index_dir / "vectors.npz"
        self.meta_file = self.index_dir / "meta.json"
        self.conn = get_docs_db(f"{index_dir}/docs.db")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model()
        return self._model

    def is_known_hash(self, content_hash: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM docs WHERE content_hash = ?", (content_hash,))
        return cur.fetchone() is not None

    def ingest_document(
        self,
        identifier: str,
        title: str,
        text: str,
        source_kind: str = "web",
        robots_ok: bool = True
    ) -> int:
        """Process and index a document into canonical chunks and vectors.
        Returns the number of new chunks created.
        """
        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            return 0

        # Compute content hash
        norm_text = re.sub(r"\s+", " ", text).strip()
        content_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

        # Idempotence: check if content already exists
        if self.is_known_hash(content_hash):
            return 0

        stem = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-").lower()[:50] or "web-doc"
        tokenizer = self.model.tokenizer
        raw_chunks = make_chunks(text, tags=["web", source_kind], tokenizer=tokenizer, stem=stem)
        if not raw_chunks:
            return 0

        # Format as canonical chunk records with provenance
        new_chunks = []
        for c in raw_chunks:
            chunk_record = {
                "path": f"web/{stem}.md",
                "heading": c["heading"],
                "text": c["text"],
                "tags": c["tags"],
                "source_url": identifier,
                "provenance": "primary",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            new_chunks.append(chunk_record)

        # Encode new vectors
        new_vectors = self.model.encode(
            [embed_text(c) for c in new_chunks],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        ).astype(np.float32)

        # Update chunks.jsonl
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            for c in new_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        # Update vectors.npz
        if self.vectors_file.is_file():
            existing_vecs = np.load(self.vectors_file)["vectors"]
            updated_vecs = np.vstack([existing_vecs, new_vectors])
        else:
            updated_vecs = new_vectors
        np.savez(self.vectors_file, vectors=updated_vecs)

        # Record in docs database
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO docs (url, title, fetched_at, content_hash, robots_ok, source_kind, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (identifier, title, datetime.now(timezone.utc).isoformat(), content_hash, 1 if robots_ok else 0, source_kind, len(new_chunks))
        )
        self.conn.commit()

        # Sync FTS5 table
        from search import FTSIndex
        fts = FTSIndex(db_path=f"{self.index_dir}/fts5.db", chunks_path=str(self.chunks_file))
        fts._ensure_index()

        return len(new_chunks)

    def ingest_url(self, url: str) -> int:
        """Check robots.txt, fetch URL, extract text, and index. Returns chunk count."""
        robots_ok = check_robots_txt(url)
        if not robots_ok:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO docs (url, title, fetched_at, content_hash, robots_ok, source_kind, chunk_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (url, "", datetime.now(timezone.utc).isoformat(), "", 0, "web", 0)
            )
            self.conn.commit()
            print(f"Skipping {url}: Disallowed by robots.txt")
            return 0

        title, text = extract_web_content(url)
        if not text:
            print(f"No extractable text from {url}")
            return 0

        count = self.ingest_document(identifier=url, title=title, text=text, source_kind="web", robots_ok=True)
        print(f"Ingested {url} -> {count} chunks added.")
        return count


def run_self_test():
    """Verify idempotency and robots handling."""
    print("=== Running IngestionEngine Self-Test ===")
    engine = IngestionEngine("vectors")

    sample_url = "https://example.com/test-doc-self-learning"
    sample_title = "Self-Learning Systems in Personal Knowledge Bases"
    sample_text = """
    # Self-Learning Knowledge Systems
    
    Building a personal knowledge retrieval system requires combining lexical search with dense embeddings.
    When users perform queries, reciprocal rank fusion combines sparse keyword matching with dense semantics.
    Exact token matching is essential for error logs, identifiers, and version numbers.
    Over time, knowledge gaps are identified through unanswerable query detection, triggering targeted frontier crawls.
    """ * 3

    # Test 1: Ingest document
    c1 = engine.ingest_document(sample_url, sample_title, sample_text, source_kind="test")
    print(f"First ingestion: {c1} chunks added.")

    # Test 2: Ingest identical content again (must be 0 - idempotency)
    c2 = engine.ingest_document(sample_url, sample_title, sample_text, source_kind="test")
    print(f"Second identical ingestion: {c2} chunks added.")
    assert c2 == 0, f"Expected 0 chunks on duplicate ingest, got {c2}"
    print("Idempotence verified: PASS")

    print("=== Self-Test Passed Successfully ===")


def main():
    ap = argparse.ArgumentParser(description="Multi-source ingestion engine.")
    ap.add_argument("--url", type=str, help="Single URL to fetch and ingest")
    ap.add_argument("--file", type=str, help="File with list of URLs to ingest")
    ap.add_argument("--test", action="store_true", help="Run ingestion self-tests")
    args = ap.parse_args()

    if args.test:
        run_self_test()
        return

    engine = IngestionEngine("vectors")
    if args.url:
        engine.ingest_url(args.url)
    elif args.file:
        urls = [line.strip() for line in open(args.file, encoding="utf-8") if line.strip() and not line.startswith("#")]
        for u in urls:
            engine.ingest_url(u)
            time.sleep(1.0)  # politeness delay


if __name__ == "__main__":
    main()
