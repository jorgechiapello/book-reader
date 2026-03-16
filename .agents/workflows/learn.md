---
description: "Execute a deep post-mortem analysis of the current conversation, locate operational gaps that prevented full autonomy, and orchestrate persistent creation of Rules, Workflows, or Skills to eliminate future systemic failures."
---

# Learn Workflow: Retrospective Analysis and Systemic Self-Improvement

Goal: Perform a detailed introspection of the current session's transactional history, algorithmically isolate friction points where autonomy failed or required corrective human intervention, and automatically synthesize the appropriate persistent infrastructure (Rules, Workflows, Skills, or Knowledge) to incorporate this capability into the development environment.

## Steps:

### 1. Extract and Analyze In-Session Conversational Context

- Process the entire transcript of the active conversation, from the user's initial prompt up to the invocation of this workflow.
- Isolate every development directive or objective stated by the human.
- Identify with precision all operational deviations:
  - Stack traces and terminal errors
  - File manipulation operations that required multiple rewrite attempts
  - Moments where the user had to manually inject terminal commands or correct your logic via text
  - Loops where you generated a solution that failed repeatedly (2+ attempts on the same operation)

### 2. Execute Root Cause Analysis

Determine the fundamental nature of each autonomy interruption. Classify each deviation into one of three causal taxonomies:

**a) Contextual or Directive Deficit:**
The agent had all necessary syntactic tools but failed because it was unaware of how two specific system components interacted, didn't know an internal naming convention, or violated an implicit design pattern.

**b) Orchestrated Procedural Deficit:**
The agent understood the task theoretically and knew the syntax of individual tools, but systematically erred in the execution order or state transitions during a multi-phase process.

**c) Specific Technical Capability Deficit:**
The task required manipulating complex data schemas, interacting with undocumented APIs, executing processes that need deterministic validation via scripts, or applying deep logic transformations based on examples.

**d) Structural or Architectural Deficit:**
The agent applied an architectural pattern (like scattered registries) that conflicts with the preferred repository conventions (like Composition Root/DI). The knowledge is structural rather than procedural.

### 3. Extract Knowledge via Learner Skill

Before generating artifacts, invoke the `learner` skill (`.agents/skills/learner/SKILL.md`) to run its extraction checklist against each identified failure:

- For each friction point, evaluate whether it passes the learner's filter (non-trivial, non-Googleable, transferable).
- For qualifying learnings, produce structured output: **Problem Statement**, **Root Cause**, **Solution**, **Triggers**, and **Principle**.
- Discard any learnings that don't pass the filter — not every failure warrants persistence.

### 4. Synthesize the Remediation Plan & Request User Approval

- Write a concise manifest in the chat titled **"Agent Retrospective and Evolution Plan"**.
- For each extracted learning, specify the origin of the failure and justify the choice of extensibility artifact (Rule, Workflow, Skill, or Knowledge).
- List the exact file paths and artifact types you intend to create.
- **[CRITICAL PAUSE]** You MUST stop here and explicitly ask the user: *"Do you agree with these learnings and the proposed remediation plan?"*
- Do NOT proceed to generate any files until the user explicitly confirms or provides corrections. If the user provides corrections, update the learnings and plan accordingly before proceeding.

### 5. Load Meta-Skill Architect Formatting Rules

Before writing ANY artifact files, read and apply the `meta-skill-architect` skill (`.agents/skills/meta-skill-architect/SKILL.md`). This ensures:
- Correct YAML frontmatter syntax (no nested single quotes, correct fields)
- Proper directory topology for skills (kebab-case, `SKILL.md` required)
- Character limits for workflows (12,000 max)
- Valid section headings (`## Goal`, `## Instructions`, `## Constraints`)

### 6. Generate Extensibility Artifacts

Based on the diagnosis and extracted learnings, use your file-writing capabilities to instantiate the solutions:

**SCENARIO A — Rule Creation (Contextual Deficit):**
- Create or update a file in `.agents/rules/` with passive, imperative, concise instructions.
- If updating `AGENTS.md`, append information under the `## Lessons Learned (Knowledge Base)` section WITHOUT overwriting existing architectural conventions.
- Reserve rules ONLY for directives that must apply to virtually ALL future interactions.

**SCENARIO B — Workflow Creation (Procedural Deficit):**
- Create a file at `.agents/workflows/<process-name>.md`.
- Follow `meta-skill-architect` specs: YAML Frontmatter with `description:` only, `Goal:` + `## Steps:` headings, under 12,000 chars.

**SCENARIO C — Skill Creation (Capability Deficit):**
- Follow `meta-skill-architect` specs: kebab-case directory, `SKILL.md` with `name:` + `description:` YAML, `## Goal` / `## Instructions` / `## Constraints` sections.
- Optionally add `scripts/`, `examples/`, or `resources/` subdirectories if needed.

**SCENARIO D — Knowledge Creation (Architectural Deficit):**
- Create a file at `.agents/knowledge/<pattern-name>.md`.
- Follow `meta-skill-architect` specs: YAML Frontmatter with `description:` only. Document the preferred architectural heuristics and structural patterns.

### 7. Validation and Loop Closure

- Emit a confirmation providing the direct relative links to the new files or directories created.
- Formally confirm that the environment has been enriched and the autonomic vulnerability has been persistently patched.
- Suggest the user commit these changes to version control for auditability.

> **IMPORTANT:** Always run the Root Cause Analysis BEFORE writing any files. This prevents overreaction to trivial errors that would generate unnecessary skills and saturate the semantic routing engine.