---
name: learner
description: "Use this skill after resolving a tricky bug or discovering a non-obvious workaround. It extracts knowledge that teaches the agent how to think about similar problems in the future, focusing on principles and decision-making heuristics rather than code snippets."
---

# Learner: Knowledge Extraction from Debugging Sessions

## Goal

Extract durable, transferable knowledge from hard-won debugging victories and non-obvious workarounds. The output should teach future agent instances how to **think about** similar problems — not just memorize solutions.

## Instructions

### When to Activate

Activate this skill when ALL of the following are true:
- A bug or issue was resolved that required **real debugging effort** (not a trivial typo fix)
- The solution involved a **non-obvious insight** — something that couldn't be found via a simple search
- The knowledge is **transferable** to future similar situations (not hyper-specific to one code location)

### When to STOP — Do NOT Extract Knowledge If:

- The fix is easily Googleable (e.g., standard library usage, common error messages)
- The knowledge is purely specific to the current codebase layout and won't generalize
- The fix didn't require real debugging effort (typo, missing import, wrong variable name)
- The information already exists in a skill, rule, or the project's `AGENTS.md`

### Extraction Checklist

For each valid learning, produce a structured output with these fields:

1. **Problem Statement:** What exactly went wrong? What were the symptoms? What made it hard to diagnose?
2. **Root Cause:** The actual underlying issue — why did the obvious solution not work?
3. **Solution:** The specific fix or workaround that resolved it.
4. **Triggers:** What signals in future conversations should make the agent recall this knowledge? (e.g., "When seeing error X with library Y", "When migrating between versions of Z")
5. **Principle:** The general heuristic or mental model to apply, stated as a decision-making rule (e.g., "Always check the connection pool size before assuming a timeout is network-related")

### Output Format

After extraction, determine the correct persistence mechanism:
- **If the learning is a universal directive** → Append to `AGENTS.md` under `## Lessons Learned (Knowledge Base)`
- **If the learning is a reusable procedure** → Create a new workflow in `.agents/workflows/`
- **If the learning requires deep specialized knowledge** → Create a new skill in `.agents/skills/`

Use the `meta-skill-architect` skill to ensure any generated artifacts have correct YAML frontmatter.

## Examples

### Input Scenario
The agent spent 4 iterations trying to fix an import error. The user pointed out that the module had been refactored into a package with `__init__.py` re-exports, and the import path needed updating.

### Extracted Learning
- **Problem Statement:** ImportError when importing `extract_segments` from `text_extractors` — the function existed but Python couldn't find it.
- **Root Cause:** Module was refactored from a single file into a package. The `__init__.py` didn't re-export the function, so the old import path broke silently.
- **Solution:** Updated `__init__.py` to re-export public functions, or updated import to use the new submodule path.
- **Triggers:** "When encountering ImportError after a module refactoring", "When a function exists in code but Python can't import it"
- **Principle:** After any module-to-package refactoring, always verify that `__init__.py` re-exports all public symbols that external code depends on.

## Constraints

- Do NOT extract knowledge that is trivially obvious or standard practice.
- Do NOT create duplicate entries — always check existing rules and skills before persisting.
- Do NOT store raw code snippets as learnings — focus on the **why** and the **decision heuristic**, not the **what**.
- Keep each learning concise. If an explanation exceeds 200 words, it should probably be a skill, not a rule entry.
- Always inform the user what was extracted and where it was persisted.
