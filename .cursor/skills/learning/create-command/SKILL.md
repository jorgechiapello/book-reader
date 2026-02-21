---
name: learning-create-command
description: Create Cursor slash commands from extracted learnings. Use when persisting workflows or procedures as .cursor/commands/*.md; when the learning skill outputs a workflow to be saved as a command.
---
# Create Commands (from Learnings)

Create slash commands from extracted learnings. Use this skill when persisting workflows or procedures to `.cursor/commands/`.

## Location

`.cursor/commands/<name>.md`

## Format

Commands are markdown files. Keep them light—objective, steps, optional output format. Avoid duplicating full methodology; reference the learning skill or other skills when needed.

```markdown
# Command Name

Brief objective.

## Instructions
1. [Step]
2. [Step]
3. [Step]

## Output
[Optional: expected format or structure]
```

## Guidelines

- Use descriptive filenames (e.g., `learn-and-update.md`, `code-review.md`).
- Keep commands focused on a single objective.
- Reference skills for methodology rather than embedding it all.
