---
description: Analyze recent messages for gaps between agent output and user corrections, then propose skill or workflow improvements.
---

# Learning

Follow the **learning** skill (`.agent/skills/learning/SKILL.md`). Analyze the last 10 messages (or N if specified), find gaps (agent output vs user corrections), and output them under `## Gaps`.

For each gap, determine the right target:

- **Skill** (`.agent/skills/<name>/SKILL.md`) — knowledge about *how* to do something (coding conventions, agent design principles, etc.)
- **Workflow** (`.agent/workflows/<name>.md`) — step-by-step *procedure* an agent should follow (deploy, learn, review, etc.)
- **Both** — if the gap reveals a missing step in a workflow *and* a missing principle in a skill

Apply the **Semantic Match Check** from the learning skill before updating any existing file: if the gap's semantics don't align with the file's stated purpose, create a new one instead.

Ask before creating or updating any skill or workflow files.
