---
tags: [obsidian, tools]
---

# Obsidian Basics

Obsidian stores notes as plain Markdown files in a local folder called a **vault**. Nothing is locked into a database: if Obsidian disappeared tomorrow, every note would still open in any text editor. That property — local-first, plain text — is the main reason I chose it over Notion and Roam.

## Core concepts

- **Vault**: a folder on disk that Obsidian opens. One vault per knowledge domain works better for me than one giant vault.
- **Notes**: files ending in `.md`. The filename is the identity of the note.
- **Links**: `[[Note title]]` creates a wikilink to another note; see [[wikilinks-and-backlinks]].
- **Plugins**: thousands of community extensions. The one I actually depend on is [[dataview-plugin]].

## What makes it a knowledge base rather than a notes app

Folders organize notes for storage, but links organize them for thought. A note that is never linked is effectively lost. My rule: a note isn't done until it links to at least one other note — that's the [[zettelkasten]] habit bleeding through.

## Settings worth changing on day one

1. Turn on "Detect all file extensions" only if you store non-note files in the vault — I keep PDFs out and reference them by path instead.
2. Set "Default location for new notes" to a single inbox folder, then process that inbox during the [[weekly-review]].
3. Enable the backlinks pane in the sidebar; it is the cheapest discovery mechanism in the whole app.

See [[vault-organization]] for how I structure folders versus tags.
