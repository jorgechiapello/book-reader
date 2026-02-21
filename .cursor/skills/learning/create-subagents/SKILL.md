---
name: learning-create-subagents
description: Create subagent or task configurations from extracted learnings. Use when persisting workflows as subagent definitions, MCP task configs, or agent-specific instructions; when the learning skill outputs a workflow suited for a specialized agent.
---
# Create Subagents (from Learnings)

Create subagent or task configurations from extracted learnings. Use this skill when persisting workflows as subagent definitions, task configs, or agent-specific instructions.

## Use Cases

- **MCP task configs** – Subagent definitions for mcp_task (explore, shell, generalPurpose).
- **Agent instructions** – Prompts or configs for specialized agents (e.g., code reviewer, test runner).
- **Workflow configs** – Structured workflows with clear steps and handoff points.

## Guidelines

- Extract the core prompt and task description from learnings.
- Specify subagent_type, description, and prompt when applicable.
- Keep prompts self-contained so the subagent can run without full conversation context.
- Document when to invoke vs. run inline.
