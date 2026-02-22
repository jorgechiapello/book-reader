---
name: learning-create-skills
description: Create Antigravity skills from extracted learnings. Use when persisting workflows, procedures, or guidelines as .agent/skills/*/SKILL.md; when the learning skill outputs a workflow suited for a reusable skill.
---
# Create Skills (from Learnings)

Create skills from extracted learnings. Use this skill when persisting workflows, procedures, or guidelines to `.agent/skills/<name>/SKILL.md`.

## Location

`.agent/skills/<skill-name>/SKILL.md`

## Format

```markdown
---
name: skill-name
description: Brief description. Use when [trigger scenarios].
---

# Skill Title

## Instructions
Step-by-step guidance.

## Examples
[Optional: concrete examples]
```

## Guidelines

- Skills teach workflows; keep instructions actionable.
- Write description in third person, include trigger terms.
- Prefer project skills (`.agent/skills/`) for team sharing.
- Keep SKILL.md under 500 lines; use reference.md for details.
- Follow create-skill best practices when available.
