"""Index an Obsidian vault into a searchable embedding store.

Usage:
    python index.py --vault ./demo_vault --out ./vectors

Outputs (in --out):
    vectors.npz   float32 matrix, shape (n_chunks, dim), rows L2-normalized
    chunks.jsonl  one JSON object per chunk: {path, heading, text, tags}
    meta.json     vault root, model name, dims -- no timestamps, so re-indexing
                  is byte-identical and diffable

Safety (see demo_vault/projects/decision-log.md, 2026-08-26):
    - dot-directories (.git, .obsidian, .config-*, ...) and deny-listed dir
      names are skipped: os.walk would ingest plugin caches, glob would skip
      them silently and under-report -- both are wrong, this is explicit.
    - exact content-hash dedup: a duplicated folder (Backups, " - Copy") must
      not make the same chunk come back twice at query time.
"""

import argparse
import contextlib
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEQ_TOKENS = 256  # MiniLM's hard limit: longer input is SILENTLY truncated
CHUNK_WORDS = 180     # ~240 wordpiece tokens, headroom under the limit
OVERLAP_WORDS = 30
MIN_CHUNK_WORDS = 15  # below this a chunk carries no signal; tail fragments merged away by overlap anyway

DENY_DIRS = {"node_modules", "site-packages", "__pycache__", "venv", ".venv", ".venv-tools"}

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


def walk_notes(vault: Path):
    """Yield .md files, skipping dot-directories and deny-listed dirs."""
    for p in sorted(vault.rglob("*.md")):
        rel_parts = p.relative_to(vault).parts
        if any(part.startswith(".") or part in DENY_DIRS for part in rel_parts[:-1]):
            continue
        yield p


def parse_note(text: str):
    """Return (tags, body). Tags-only frontmatter parser -- no dependency.

    Handles 'tags: [a, b]' and the YAML list form. Other frontmatter keys are
    ignored (indexing only needs tags).
    """
    m = _FRONTMATTER.match(text)
    if not m:
        return [], text
    fm, body = m.group(1), text[m.end():]
    tags, in_tags = [], False
    for line in fm.splitlines():
        s = line.strip()
        if s.startswith("tags:"):
            rest = s[5:].strip()
            if rest.startswith("[") and "]" in rest:
                tags += [t.strip() for t in rest[1:rest.index("]")].split(",") if t.strip()]
                in_tags = False
            else:
                in_tags = True
        elif in_tags and s.startswith("- "):
            tags.append(s[2:].strip())
        elif s and not line.startswith((" ", "\t", "-")):
            in_tags = False
    return [t for t in tags if t], body


def sections(body: str):
    """Split body into (heading, text) at markdown headings. Preamble -> heading ''."""
    matches = list(_HEADING.finditer(body))
    if not matches:
        return [("", body.strip())]
    out = []
    if matches[0].start() > 0:
        out.append(("", body[: matches[0].start()].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((m.group(2).strip(), body[m.end():end].strip()))
    return [(h, t) for h, t in out if t]


def chunk_text(text: str):
    """Sliding word window: CHUNK_WORDS with OVERLAP_WORDS overlap."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text] if len(words) >= MIN_CHUNK_WORDS else []
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [
        " ".join(words[i:i + CHUNK_WORDS])
        for i in range(0, len(words) - OVERLAP_WORDS, step)
    ]


def _prefix(stem: str, heading: str) -> str:
    return f"{stem} :: {heading}\n" if heading else f"{stem}\n"


def embed_text(chunk: dict) -> str:
    """What actually gets embedded: note identity + body. Single source for
    both the budget check in make_chunks and the encode in main."""
    return _prefix(Path(chunk["path"]).stem.replace("-", " "), chunk["heading"]) + chunk["text"]


@contextlib.contextmanager
def _measure_tokens(tokenizer):
    """Measure-only tokenization. _fit_words tokenizes overlong input precisely
    to split it, so transformers' "will result in indexing errors" warning fires
    here even though no overlong sequence ever reaches the model. The real guard
    is the SystemExit in main() over the final chunks."""
    import logging
    log = logging.getLogger("transformers")
    prior = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(prior)


def _fit_words(prefix: str, words: list, tokenizer):
    """Split words in half until prefix+body fits MAX_SEQ_TOKENS. Prefix is kept
    on every piece so each sub-chunk still knows its note identity."""
    text = prefix + " ".join(words)
    with _measure_tokens(tokenizer):
        ids = tokenizer(text)["input_ids"]
    if len(ids) <= MAX_SEQ_TOKENS:
        return [text]
    if len(words) < 2:  # single unsplittable token blob
        return [tokenizer.decode(ids[:MAX_SEQ_TOKENS])]
    mid = len(words) // 2
    return _fit_words(prefix, words[:mid], tokenizer) + _fit_words(prefix, words[mid:], tokenizer)


def make_chunks(body: str, tags: list, tokenizer, stem: str):
    """Heading-aware chunks: [{heading, text, tags}, ...].

    The note stem + heading is prefixed to every chunk's embedded text: body-only
    vectors make a named note lose to notes that discuss it (all 4 recall misses
    in the eval were title-style queries). 180 words of prose is ~240 tokens but
    code-dense text can exceed 256 -- hence the token-budget splitter.
    """
    out = []
    for heading, sec_text in sections(body):
        prefix = _prefix(stem, heading)
        for window in chunk_text(sec_text):
            for piece in _fit_words(prefix, window.split(), tokenizer):
                body_piece = piece[len(prefix):]
                if len(body_piece.split()) >= MIN_CHUNK_WORDS:
                    out.append({"heading": heading, "text": body_piece, "tags": tags})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vault", required=True, type=Path, help="vault directory to index")
    ap.add_argument("--out", default=Path("vectors"), type=Path, help="output directory")
    args = ap.parse_args()
    vault = args.vault.resolve()
    if not vault.is_dir():
        raise SystemExit(f"vault not found: {vault}")

    t0 = time.time()
    chunks, seen_hashes, dupes = [], set(), 0
    files_indexed = 0
    from rag import load_model  # deferred: model import costs ~1 min cold, keep --help fast

    model = load_model()
    tokenizer = model.tokenizer

    for p in walk_notes(vault):
        raw = p.read_text(encoding="utf-8")
        tags, body = parse_note(raw)
        h = hashlib.sha256(re.sub(r"\s+", " ", raw).strip().encode()).hexdigest()
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)
        stem = p.stem.replace("-", " ")
        note_chunks = make_chunks(body, tags, tokenizer, stem)
        if not note_chunks:
            continue
        files_indexed += 1
        rel = p.relative_to(vault).as_posix()
        for c in note_chunks:
            chunks.append({"path": rel, "heading": c["heading"], "text": c["text"], "tags": c["tags"]})

    if not chunks:
        raise SystemExit("no chunks produced -- empty vault?")

    vectors = model.encode(
        [embed_text(c) for c in chunks],
        normalize_embeddings=True,  # unit rows: query-time cosine == dot product
        batch_size=64,
        show_progress_bar=False,
    ).astype(np.float32)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # Preserve chunks with no file in the vault: ingested web pages and derived
    # memory writes live only in chunks.jsonl. A rebuild from the vault walk
    # would overwrite the file and silently delete them (vectors row i aligns
    # with chunks line i -- ingest appends both in lockstep, so orphans keep
    # their exact embedded vectors). Orphans written before the token-budget
    # splitter existed are re-split here rather than kept -- their stored vector
    # is a truncated-prefix embedding, and preserving it would keep the bug.
    chunks_file = out / "chunks.jsonl"
    vectors_file = out / "vectors.npz"
    vault_paths = {c["path"] for c in chunks}
    if chunks_file.is_file() and vectors_file.is_file():
        try:
            old = [json.loads(l) for l in chunks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            old_vecs = np.load(vectors_file)["vectors"]
            if len(old) == len(old_vecs):
                keep = [i for i, c in enumerate(old) if c.get("path") not in vault_paths]
                kept_vecs, resplit = [], []
                for i in keep:
                    c = old[i]
                    with _measure_tokens(tokenizer):
                        fits = len(tokenizer(embed_text(c))["input_ids"]) <= MAX_SEQ_TOKENS
                    if fits:
                        kept_vecs.append(old_vecs[i])
                        chunks.append(c)
                    else:
                        stem = Path(c["path"]).stem.replace("-", " ")
                        pieces = make_chunks(c["text"], c.get("tags", []), tokenizer, stem) or [c["text"]]
                        for j, p in enumerate(pieces):
                            rec = dict(c)
                            rec["text"] = p["text"] if isinstance(p, dict) else p
                            if len(pieces) > 1:
                                rec["heading"] = f"{c['heading']} (part {j + 1}/{len(pieces)})"
                            resplit.append(rec)
                if resplit:
                    vectors = np.vstack([vectors, model.encode(
                        [embed_text(c) for c in resplit], normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False).astype(np.float32)])
                    chunks.extend(resplit)
                    print(f"re-split {len(resplit)} over-budget preserved chunk(s) into token-window pieces")
                if kept_vecs:
                    vectors = np.vstack([vectors, np.array(kept_vecs, dtype=np.float32)])
                    print(f"preserved {len(kept_vecs)} ingested/derived chunk(s) with no vault file")
        except Exception as e:
            print(f"WARNING: could not preserve ingested chunks: {e}")

    # Loud, not silent: MiniLM truncates past 256 tokens without warning. Runs
    # after orphan preservation so stored overlong chunks are caught too, not
    # just the ones split from this vault walk.
    with _measure_tokens(tokenizer):
        over = [i for i, c in enumerate(chunks) if len(tokenizer(embed_text(c))["input_ids"]) > MAX_SEQ_TOKENS]
    if over:
        c = chunks[over[0]]
        raise SystemExit(
            f"chunk {over[0]} ({c['path']} :: {c['heading']}) exceeds {MAX_SEQ_TOKENS} tokens -- "
            f"this is a bug in _fit_words"
        )

    np.savez(vectors_file, vectors=vectors)
    chunks_file.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8"
    )
    (out / "meta.json").write_text(
        json.dumps(
            {"vault_root": str(vault), "model": MODEL_NAME, "chunks": len(chunks), "dim": int(vectors.shape[1])},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"indexed {files_indexed} notes ({dupes} exact dupes skipped) -> "
        f"{len(chunks)} chunks, dim {vectors.shape[1]}, {time.time() - t0:.1f}s -> {out}/"
    )


if __name__ == "__main__":
    main()
