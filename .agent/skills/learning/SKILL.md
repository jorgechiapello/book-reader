---
name: learning
description: Find gaps between agent output and user corrections to improve skills. Use when analyzing conversation history; when the user wants to learn from past interactions.
---
# Learning

You are the "Learner" of this agentic system. Your job is to find **gaps** and use them to **improve skills**: differences between what the agent did and what the user corrected, where the correction indicates a skill should be extended or refined.

## Process

### 1. Find Corrections

Review the last N messages (default: 30). Identify:

- **User edits** – User changed code, text, or structure the agent produced
- **User rejections** – User said "no", "that's wrong", "not like that", or reverted changes
- **User redirects** – User asked for something different, clarified, or specified a different approach
- **User fixes** – User completed or corrected what the agent left incomplete

### 2. Identify the Gap

For each correction, state the gap:

- **What the agent did** – The agent produced X
- **What the user wanted** – The user expected Y (or corrected to Y)
- **Skill gap** – Which skill lacked knowledge that Y was required

### 3. Map to the Right Target

For each gap, determine the correct target — it may be a **skill**, a **workflow**, or **both**:

| Target | Path | Use when the gap is about… |
|--------|------|----------------------------|
| Skill | `.agent/skills/<name>/SKILL.md` | *What* to do or *how* to reason (principles, constraints, design rules) |
| Workflow | `.agent/workflows/<name>.md` | *Steps* of a procedure an agent follows (deploy, review, learn, etc.) |
| Both | skill + workflow | The gap reveals a missing principle *and* a missing procedural step |

#### Semantic Match Check (required before updating any existing file)

Before extending any file, verify the gap's semantics **align with that file's stated purpose** (its `description` frontmatter and content scope).

- **If the gap fits** → extend that file
- **If the gap does NOT fit** → create a new skill or workflow; do not force unrelated knowledge into an existing one

```
Gap: "agents should keep pipeline tasks independent"
→ Check coding-style: covers language conventions → ❌ semantic mismatch
→ Check ai-workflows: covers agent pipeline design → ✅ match → extend it
→ If no match found → create new skill

Gap: "the learn workflow doesn't check workflows as targets, only skills"
→ Check learn.md workflow: covers the learning procedure → ✅ match → extend it
→ Also check learning skill: covers how to map gaps → ✅ match → extend it too
```

#### Skill Categories

| Category | Covers | Example skill |
|----------|--------|---------------|
| Code style | Language conventions, formatting, naming | `coding-style` |
| Agent prompting | How agents reason, design pipelines, structure tasks | `ai-workflows` |
| Tooling / procedures | Step-by-step operational workflows | `workflows/*.md` |

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

When asked to persist, use **learning/create-skills** to create or update skills under `.agent/skills/`. Ask before creating new skills or overwriting existing ones.
