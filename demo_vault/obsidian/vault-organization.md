---
tags: [obsidian, organization]
---

# Vault Organization

The rule that took longest to learn: **folders, tags, and links answer different questions.** Using all three for the same job produces a vault nobody can navigate.

## The division of labor

| Mechanism | Answers | Good for |
|---|---|---|
| Folders | "Where does this live?" | Coarse storage, archiving |
| Tags | "What is this?" | Cross-cutting categories, status |
| Links | "What is this related to?" | Actual thinking |

## My layout

```
00-inbox/       # unprocessed captures
10-projects/    # active work, one folder per project
20-areas/       # ongoing responsibilities (PARA "areas")
30-resources/   # reference material by topic
90-archive/     # dead projects, kept for archaeology
```

This is PARA ([[para-method]]) with Zettelkasten notes ([[zettelkasten]]) living flat inside `30-resources/`.

## Conventions

- Frontmatter on every note: `tags:` plus any fields Dataview queries need ([[dataview-plugin]]). Status fields only on project notes.
- A note moves to the archive when it's dead, not when it's messy. Messy-but-alive notes get a link instead.
- Folder depth capped at two levels. If I need deeper nesting, the real problem is usually that the note should be split.

## The test

A new note must be findable two ways: by browsing its folder, and by following a link. If the second fails, I skipped the linking step of [[zettelkasten]] and the note will rot.
