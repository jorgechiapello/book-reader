---
description: Analyze recent messages for gaps between agent output and user corrections, then propose skill improvements.
---

# Learning

Follow the **learning** skill (`.agent/skills/learning/SKILL.md`). Analyze the last 10 messages (or N if specified), find gaps (agent output vs user corrections), and output them under `## Gaps`.

For persistence, use the **learning/create-skills** skill (`.agent/skills/learning/create-skills/SKILL.md`) to improve skills. Ask before creating or updating skill files.
