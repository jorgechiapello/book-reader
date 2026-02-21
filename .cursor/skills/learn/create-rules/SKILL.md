---
name: learn-create-rules
description: Create or update Cursor rules from extracted learnings. Use when persisting rules, workflows, or guidelines to .cursor/rules/ or AGENTS.md; when the learn skill outputs rules to be saved.
---
# Create Rules (from Learnings)

Create or update rules from extracted learnings. Use this skill when persisting rules, workflows, or guidelines to `.cursor/rules/` or AGENTS.md.

## Targets

| Target | When | Format |
|--------|------|--------|
| **AGENTS.md** | Project-wide constraints, dependency rules | Append or merge into existing sections |
| **.cursor/rules/*.mdc** | File-specific patterns, detailed conventions | New or updated .mdc with frontmatter |

## Rule File Format (.mdc)

```markdown
---
description: Brief description
globs: **/*.py     # optional - file pattern
alwaysApply: false # true for universal rules
---

# Rule Title

Rule content...
```

## Guidelines

- Prefer AGENTS.md for broad project constraints (e.g., dependency management).
- Use `.cursor/rules/` for file-scoped rules (globs) or detailed patterns.
- Keep each rule under 50 lines, one concern per rule.
- Ask before overwriting existing content.
