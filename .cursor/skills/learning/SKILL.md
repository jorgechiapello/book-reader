---
name: learning
description: Find gaps between agent output and user corrections to improve skills. Use when analyzing conversation history; when the user wants to learn from past interactions.
---
# Learning

You are the "Learner" of this agentic system. Your job is to find **gaps** and use them to **improve skills**: differences between what the agent did and what the user corrected, where the correction indicates a skill should be extended or refined.

## Process

### 1. Find Corrections

Review the last N messages (default: 10). Identify:

- **User edits** – User changed code, text, or structure the agent produced
- **User rejections** – User said "no", "that's wrong", "not like that", or reverted changes
- **User redirects** – User asked for something different, clarified, or specified a different approach
- **User fixes** – User completed or corrected what the agent left incomplete

### 2. Identify the Gap

For each correction, state the gap:

- **What the agent did** – The agent produced X
- **What the user wanted** – The user expected Y (or corrected to Y)
- **Skill gap** – Which skill lacked knowledge that Y was required

### 3. Map to Skill Extension

Determine which skill(s) should be improved and how:

- **Existing skill** – Update `.cursor/skills/<name>/SKILL.md` with the new knowledge
- **New skill** – Create `.cursor/skills/<name>/SKILL.md` for a new workflow or procedure

### 4. Format for Skills

- **Instructions** – Add or refine step-by-step guidance
- **Examples** – Add concrete examples where the correction applies
- **Trigger terms** – Extend the description with when to use this knowledge

## Example Output

```markdown
## Gaps (Agent → User correction)

### Gap 1
- **Agent did:** Suggested Google Search for news
- **User corrected:** Asked to use Tavily instead
- **Extension:** Update research skill – add CONSTRAINT: Use Tavily for real-time news

### Gap 2
- **Agent did:** Generated code without adding package to requirements.txt
- **User corrected:** Added package to requirements.txt manually
- **Extension:** Update Python/dependency skill – add step: run `pip freeze | grep package_name` and append to requirements.txt
```

## Persistence

When asked to persist, use **learning/create-skills** to create or update skills under `.cursor/skills/`. Ask before creating new skills or overwriting existing ones.
