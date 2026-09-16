"""VaultIndex: load the built index, cosine search, guarded note reads, tag lookup."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
# Refusal floor: top-hit similarity below this -> "not in the vault", no LLM call.
# Calibrated on evalset.jsonl: unanswerable top-scores max at ~0.21, answerable
# min at ~0.32 -- this sits inside the gap (see README, Evaluation).
REFUSE_THRESHOLD = 0.27


def load_model():
    """Load the embedding model, offline-first.

    sentence-transformers HEADs huggingface.co on every load even when the
    model is fully cached, and dies when the network is down. If the model is
    cached, pin HF_HUB_OFFLINE=1 before the import so a local-first tool stays
    local. A fresh install (no cache) keeps default online behavior.

    # ponytail: cache sniff assumes the default HF cache dir; set HF_HUB_OFFLINE
    # yourself if your cache lives elsewhere.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--sentence-transformers--{MODEL_NAME.replace('-', '--')}"
    if cache.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@dataclass
class Hit:
    path: str
    heading: str
    text: str
    tags: list
    score: float
    source_url: str = ""
    provenance: str = "primary"
    ingested_at: str = ""
    derived_from_hashes: dict = None


class VaultIndex:
    def __init__(self, index_dir="vectors"):
        d = Path(index_dir)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.vault = Path(self.meta["vault_root"])
        self.vectors = np.load(d / "vectors.npz")["vectors"]  # (N, dim), rows L2-normalized
        self.chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        self._model = None  # lazy: SentenceTransformer import costs ~1 min cold

    @property
    def model(self):
        if self._model is None:
            self._model = load_model()
        return self._model

    def search(self, query: str, k: int = 5, exclude_derived: bool = False) -> list:
        """Cosine top-k. Rows and query are unit-normalized, so dot product == cosine."""
        qv = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
        scores = self.vectors @ qv  # (N,)
        top = np.argsort(scores)[::-1]
        hits = []
        for i in top:
            chunk = self.chunks[i]
            if exclude_derived and chunk.get("provenance") == "derived":
                continue
            hit = Hit(
                path=chunk.get("path", ""),
                heading=chunk.get("heading", ""),
                text=chunk.get("text", ""),
                tags=chunk.get("tags", []),
                score=float(scores[i]),
                source_url=chunk.get("source_url", ""),
                provenance=chunk.get("provenance", "primary"),
                ingested_at=chunk.get("ingested_at", ""),
                derived_from_hashes=chunk.get("derived_from_hashes")
            )
            hits.append(hit)
            if len(hits) == k:
                break
        return hits

    def read_note(self, path: str) -> str:
        """Read one note by vault-relative path. Path traversal is a trust boundary here:
        the caller is an LLM tool loop, so '../' escapes must fail loudly."""
        p = (self.vault / path).resolve()
        if not p.is_relative_to(self.vault.resolve()):
            raise ValueError(f"path escapes vault: {path}")
        if p.suffix != ".md" or not p.is_file():
            raise ValueError(f"not a note in this vault: {path}")
        return p.read_text(encoding="utf-8")

    def notes_by_tag(self, tag: str) -> list:
        return sorted({c["path"] for c in self.chunks if tag in c["tags"]})

    @property
    def stats(self):
        return {
            "notes": len({c["path"] for c in self.chunks}),
            "chunks": len(self.chunks),
            "model": self.meta["model"],
            "vault": str(self.vault),
        }
