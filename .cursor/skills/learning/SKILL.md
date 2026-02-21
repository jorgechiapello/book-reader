---
name: learning
description: Find gaps between agent output and user corrections to extend skills, rules, or commands. Use when analyzing conversation history to update project manual; when the user wants to learn from past interactions; or when running the learn-and-update command.
---
# Learning

You are the "Learner" of this agentic system. Your job is to find **gaps**: differences between what the agent did and what the user corrected, where the correction indicates that a skill, rule, or command should be extended.

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
- **Gap** – The skill/rule/command lacked knowledge that Y was required

### 3. Map to Extension

Determine what should be extended:

| Gap type | Extend |
|----------|--------|
| Constraint, convention, or preference | Rule (learning/create-rules) |
| Slash-command or procedure | Command (learning/create-command) |
| Subagent or task config | Subagent (learning/create-subagents) |
| Workflow or procedure | Skill (learning/create-skills) |

### 4. Format

**Rules:** `CONSTRAINT:`, `USER_PREFERENCE:`, `PROJECT_LOGIC:`, `IF-THEN:`

**Workflows:** `WORKFLOW: [Name]` with numbered steps

## Example Output

```markdown
## Gaps (Agent → User correction)

### Gap 1
- **Agent did:** Used spaces for indentation
- **User corrected:** Changed to tabs
- **Extension:** Rule – USER_PREFERENCE: Use tabs, not spaces, for indentation

### Gap 2
- **Agent did:** Suggested Google Search for news
- **User corrected:** Asked to use Tavily instead
- **Extension:** Rule – CONSTRAINT: Use Tavily for real-time news; avoid Google Search

### Gap 3
- **Agent did:** Generated code without adding package to requirements.txt
- **User corrected:** Added package to requirements.txt manually
- **Extension:** Rule – IF-THEN: If adding a Python package, run `pip freeze | grep package_name` and append to requirements.txt
```

## Persistence

When asked to persist, use the appropriate sub-skill under `.cursor/skills/learning/`:

- **Rules** → learning/create-rules
- **Commands** → learning/create-command
- **Subagents** → learning/create-subagents
- **Skills** → learning/create-skills
