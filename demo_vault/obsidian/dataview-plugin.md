---
tags: [obsidian, plugins]
---

# Dataview Plugin

Dataview turns the vault into a queryable database. Notes become rows; frontmatter fields and tags become columns. It ships two query languages; I only use the inline one.

## A real query from my vault

````
```dataview
TABLE status, due
FROM "projects"
WHERE status != "done"
SORT due ASC
```
````

That renders a live table of every project note with a `status` and `due` field in frontmatter. The table re-renders whenever a note changes — no maintenance.

## What it's good for

- Dashboards: all open items with a given tag.
- Finding stale notes: `WHERE file.mtime < date(today) - 30`.
- Aggregations the graph view can't do, like counting notes per folder.

## What it is not

Dataview is read-only. It renders, it doesn't automate — writing notes programmatically still goes through Templater or a script. And because queries depend on frontmatter discipline, a sloppy tag breaks the table silently. That's the real cost: the plugin rewards consistent metadata, so my frontmatter convention in [[vault-organization]] exists as much for Dataview as for me.

For anything heavier than display, I now reach for the [[project-rag-assistant]] instead — natural language over the whole vault beats writing a query.
