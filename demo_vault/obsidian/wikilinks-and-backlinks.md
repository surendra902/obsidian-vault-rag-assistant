---
tags: [obsidian, knowledge-graph]
---

# Wikilinks and Backlinks

A wikilink is `[[target note]]`. Obsidian resolves it by filename, vault-wide, regardless of which folder the target lives in. This is the single feature that turns a folder of files into a graph.

## How the mechanism works

- Links are stored **in the note text**, not in a separate database. Grepping the vault finds every link.
- **Backlinks** are computed, not stored: the graph engine scans all notes for `[[this note]]` and shows the inverse edges in the backlinks pane.
- **Unresolved links** (target doesn't exist yet) show up greyed out. They are cheap placeholders for future notes — I deliberately leave them around as to-dos.

## Patterns that work

1. **Link as you write, don't link afterwards.** Retrofitting links never happens.
2. Mention links in running prose (`...as [[Ahrens]] argues...`) rather than dumping a link farm at the bottom. The sentence context around a link is what makes it meaningful later.
3. Use link aliases when the prose reads better with different words: `[[zettelkasten|slip-box]]`.

## Backlinks as a thinking tool

The backlinks pane answers "what points here?", which is often the more interesting question. When I'm stuck on [[project-rag-assistant]], reading its backlinks reminds me which decisions elsewhere depend on it. The graph view is pretty but the backlinks pane is where the actual work happens — see [[vault-organization]] for how links, tags, and folders divide the labor.
