"""Memory write-back (T1) with lineage tracking and self-poisoning guards.

Guards:
1. Sole Citation Guard: A derived document can never be the sole citation for an answer.
   The answer must cite at least one primary source.
2. Lineage Invalidation: Every derived document records parent file hashes.
   When a parent note changes or is removed, derived documents are automatically purged.
3. Holdout Isolation: eval.py --exclude-derived on holdout must strictly equal holdout.
"""

import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from index import embed_text
from rag import VaultIndex, load_model


class MemoryManager:
    def __init__(self, index_dir: str = "vectors", vault_dir: str = "demo_vault"):
        self.index_dir = Path(index_dir)
        self.vault_dir = Path(vault_dir)
        self.chunks_file = self.index_dir / "chunks.jsonl"
        self.vectors_file = self.index_dir / "vectors.npz"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model()
        return self._model

    def compute_file_hash(self, rel_path: str) -> Optional[str]:
        """Compute SHA-256 of normalized note text."""
        p = self.vault_dir / rel_path
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8")
        norm = re.sub(r"\s+", " ", text).strip()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def validate_citations(self, citations: List[dict]) -> bool:
        """Enforce Sole Citation Guard: An answer cannot rely solely on derived docs.
        Must have at least one primary document.
        """
        if not citations:
            return False
        has_primary = any(c.get("provenance", "primary") == "primary" for c in citations)
        return has_primary

    def save_derived_answer(
        self,
        question: str,
        distilled_answer: str,
        primary_source_paths: List[str]
    ) -> Optional[dict]:
        """Save a verified answer as a derived memory chunk with parent hash lineage."""
        if not primary_source_paths:
            print("Cannot write memory: No primary source paths provided.")
            return None

        # Build lineage map
        parent_hashes = {}
        for sp in primary_source_paths:
            h = self.compute_file_hash(sp)
            if h:
                parent_hashes[sp] = h

        if not parent_hashes:
            print("Cannot write memory: Parent sources not found in vault.")
            return None

        stem = re.sub(r"[^\w\s-]", "", question).strip().replace(" ", "-").lower()[:40] or "memory"

        # A derived answer is arbitrary LLM output and can run past MiniLM's
        # 256-token window. Split it with the same token-budget splitter as the
        # vault indexer -- otherwise the embedding silently truncates mid-sentence
        # and the memory is recalled by a prefix, not by what it actually says.
        from index import make_chunks
        answer = distilled_answer.strip()
        # make_chunks drops fragments under MIN_CHUNK_WORDS, but a short answer
        # still fits the token window in one piece -- keep it verbatim.
        pieces = make_chunks(answer, [], self.model.tokenizer, stem) or [{"text": answer}]
        chunk_records = []
        for i, piece in enumerate(pieces):
            suffix = f" (part {i + 1}/{len(pieces)})" if len(pieces) > 1 else ""
            chunk_records.append({
                "path": f"derived/{stem}.md",
                "heading": f"Q: {question}{suffix}",
                "text": piece.get("text", ""),
                "tags": ["derived", "memory"],
                "provenance": "derived",
                "derived_from_hashes": parent_hashes,
                "source_paths": primary_source_paths,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        # Encode new vectors (one per chunk, aligned row-to-line with the appends below)
        new_vecs = self.model.encode(
            [embed_text(c) for c in chunk_records],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        # Append to chunks.jsonl
        with open(self.chunks_file, "a", encoding="utf-8") as f:
            for c in chunk_records:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        # Append to vectors.npz
        if self.vectors_file.is_file():
            existing = np.load(self.vectors_file)["vectors"]
            updated = np.vstack([existing, new_vecs])
        else:
            updated = new_vecs
        np.savez(self.vectors_file, vectors=updated)

        # Re-index FTS5
        from search import FTSIndex
        fts = FTSIndex(db_path=f"{self.index_dir}/fts5.db", chunks_path=str(self.chunks_file))
        fts._ensure_index()

        return chunk_records[0]

    def invalidate_stale_memories(self) -> int:
        """Scan derived chunks and purge any whose parent documents have changed or were deleted.
        Returns the number of invalidated chunks removed.
        """
        if not self.chunks_file.is_file():
            return 0

        chunks = [json.loads(line) for line in self.chunks_file.read_text(encoding="utf-8").splitlines() if line]
        vectors = np.load(self.vectors_file)["vectors"] if self.vectors_file.is_file() else None

        valid_chunks = []
        valid_indices = []
        purged = 0

        for idx, c in enumerate(chunks):
            if c.get("provenance") != "derived":
                valid_chunks.append(c)
                valid_indices.append(idx)
                continue

            # Check lineage
            parent_hashes = c.get("derived_from_hashes", {})
            stale = False
            for path, recorded_hash in parent_hashes.items():
                current_hash = self.compute_file_hash(path)
                if current_hash != recorded_hash:
                    stale = True
                    break

            if stale:
                purged += 1
                print(f"Purging stale derived chunk: {c.get('heading')} (parent {path} changed/removed)")
            else:
                valid_chunks.append(c)
                valid_indices.append(idx)

        if purged > 0:
            # Rewrite chunks.jsonl
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                for c in valid_chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

            # Rewrite vectors.npz
            if vectors is not None and len(valid_indices) > 0:
                updated_vectors = vectors[valid_indices]
                np.savez(self.vectors_file, vectors=updated_vectors)

            # Re-index FTS5
            from search import FTSIndex
            fts = FTSIndex(db_path=f"{self.index_dir}/fts5.db", chunks_path=str(self.chunks_file))
            fts._ensure_index()

        return purged


def run_self_test():
    """Verify memory write-back, sole-citation guard, and lineage invalidation in an isolated test harness."""
    print("=== Running MemoryManager Self-Test ===")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        test_dir = Path(temp_dir)
        test_vectors = str(test_dir / "vectors")
        test_vault = str(test_dir / "vault")
        shutil.copytree("vectors", test_vectors)
        shutil.copytree("demo_vault", test_vault)

        mem = MemoryManager(test_vectors, test_vault)

        # Guard Test 1: Sole Citation Guard
        derived_only_citations = [{"provenance": "derived", "path": "derived/memo.md"}]
        mixed_citations = [{"provenance": "derived", "path": "derived/memo.md"}, {"provenance": "primary", "path": "books/getting-things-done.md"}]
        assert mem.validate_citations(derived_only_citations) is False, "Sole citation guard failed: should reject derived-only"
        assert mem.validate_citations(mixed_citations) is True, "Sole citation guard failed: should accept with primary source"
        print("Sole citation guard verified: PASS")

        # Test 2: Write derived answer
        q = "What is the two-minute rule in personal workflow?"
        ans = "In David Allen's Getting Things Done, if an action takes under two minutes, execute it immediately."
        rec = mem.save_derived_answer(q, ans, ["books/getting-things-done.md"])
        assert rec is not None, "Failed to save derived memory"
        print("Derived answer indexed with parent lineage: PASS")

        # Contamination check: actively verify --exclude-derived on index with real derived chunk
        from search import HybridIndex
        idx = HybridIndex(test_vectors)
        hits_incl = idx.search(q, k=5, exclude_derived=False)
        hits_excl = idx.search(q, k=5, exclude_derived=True)
        assert any(h.provenance == "derived" for h in hits_incl), "Expected derived chunk in hits when exclude_derived=False"
        assert not any(h.provenance == "derived" for h in hits_excl), "Expected zero derived chunks in hits when exclude_derived=True"
        print("Contamination filter (--exclude-derived) actively verified on real derived chunk: PASS")
        idx.close()

        # Test 3: Lineage check with unchanged parent
        purged_unchanged = mem.invalidate_stale_memories()
        assert purged_unchanged == 0, f"Expected 0 purges on unchanged parent, got {purged_unchanged}"
        print("Clean lineage integrity verified: PASS")

        # Test 4: Lineage invalidation check by mutating parent hash in memory record
        chunks = [json.loads(line) for line in mem.chunks_file.read_text(encoding="utf-8").splitlines() if line]
        for c in chunks:
            if c.get("path") == rec["path"]:
                c["derived_from_hashes"]["books/getting-things-done.md"] = "fake_stale_hash_12345"
        with open(mem.chunks_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        purged_stale = mem.invalidate_stale_memories()
        assert purged_stale == 1, f"Expected 1 purge on stale parent, got {purged_stale}"
        print("Automatic lineage invalidation & purge verified: PASS")

    print("=== MemoryManager Self-Test Passed Successfully ===")


if __name__ == "__main__":
    run_self_test()
