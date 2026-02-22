---
name: code-review-workflow
description: Orchestration workflow for plan-aligned code review. Use when a major project step is complete and needs review. Invokes code-quality, architecture, and documentation subagents.
---

# Code Review Workflow

Orchestrate a full code review by delegating to topic-specific subagents and aggregating findings. Extend this workflow as new review dimensions are added.

## Instructions

1. **Plan Alignment Analysis**
   - Compare the implementation against the original planning document or step description
   - Identify deviations from the planned approach, architecture, or requirements
   - Assess whether deviations are justified improvements or problematic departures
   - Verify that all planned functionality has been implemented

2. **Invoke Topic Subagents**
   - **Code Quality**: Invoke the code-quality subagent (uses code-quality-review skill)
   - **Architecture**: Invoke the architecture subagent (uses architecture-review skill)
   - **Documentation**: Invoke the documentation subagent (uses documentation-review skill)
   - Provide each subagent with context: changed files, plan excerpt, scope

3. **Aggregate and Categorize Issues**
   - Merge findings from all subagents
   - Categorize each issue: **Critical** (must fix), **Important** (should fix), **Suggestion** (nice to have)
   - For plan deviations, state whether they are problematic or beneficial

4. **Communication Protocol**
   - Acknowledge what was done well before highlighting issues
   - If significant plan deviations: ask the coding agent to review and confirm
   - If plan issues: recommend plan updates
   - For implementation problems: provide clear guidance on fixes

## Output

Structured, actionable feedback. Be thorough but concise. Include specific examples and code snippets where helpful.
