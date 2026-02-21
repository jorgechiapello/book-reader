---
name: learn
description: Observe interactions and extract learnings into rules, workflows, and guidelines. Use when analyzing conversation history to update project manual; when the user wants to learn from past interactions; or when running the learn-and-update command.
---
# Learn

You are the "Learner" of this agentic system. Observe interactions and extract learnings into **rules**, **workflows**, and **guidelines**. When persisting, use the create-rules, create-command, create-subagents, or create-skills sub-skill as appropriate.

## Process

### 1. Analyze

Review the last N messages (default: 10). Identify:

- **User preferences** (style, format, workflow)
- **Failure points** (wrong tools, incorrect assumptions)
- **Project-specific patterns** (stack, conventions, constraints)
- **Tool/task patterns** (which tools for which tasks)
- **Repeated workflows** (sequences the user follows)
- **Format expectations** (how the user wants output)

### 2. Synthesize

Create learnings in the appropriate form:

- **Rules** – Constraints, preferences, conditional logic (actionable, specific)
- **Workflows** – Step-by-step procedures when a sequence emerged or was repeated
- **Guidelines** – Broader recommendations, best practices, decision frameworks

### 3. Format

**Rules:** `CONSTRAINT:`, `USER_PREFERENCE:`, `PROJECT_LOGIC:`, `IF-THEN:`

**Workflows:** `WORKFLOW: [Name]` with numbered steps

**Guidelines:** `GUIDELINE:` for broader recommendations

## Example Output

```markdown
## Extracted Learnings

### Rules
- CONSTRAINT: [rule]
- USER_PREFERENCE: [rule]
- PROJECT_LOGIC: [rule]
- IF-THEN: [rule]

### Workflows
WORKFLOW: [Name]
1. [Step]
2. [Step]

### Guidelines
- GUIDELINE: [recommendation]
```

## Persistence

When asked to persist, use the appropriate sub-skill under `.cursor/skills/learn/`:

- **Rules** → learn/create-rules
- **Commands** → learn/create-command
- **Subagents** → learn/create-subagents
- **Skills** → learn/create-skills
