---
description: Orchestrate a plan-aligned code review covering quality, architecture, and documentation. Run when a major project step is complete.
---

# Code Review Workflow

You are a Senior Code Reviewer. Orchestrate a full plan-aligned code review in sequence. Use the skills in `.agent/skills/code-review/` for detailed checklists.

## Steps

### 1. Plan Alignment Analysis
- Compare the implementation against the original planning document or step description
- Identify deviations from the planned approach, architecture, or requirements
- Assess whether deviations are justified improvements or problematic departures
- Verify that all planned functionality has been implemented

### 2. Code Quality Review
Follow the **code-quality-review** skill (`.agent/skills/code-review/code-quality/SKILL.md`):
- Check patterns, conventions, and anti-patterns
- Verify error handling and defensive programming
- Assess organization, naming, and maintainability
- Evaluate test coverage and quality
- Flag security and performance risks

### 3. Architecture Review
Follow the **architecture-review** skill (`.agent/skills/code-review/architecture/SKILL.md`):
- Evaluate SOLID principles and design patterns
- Check separation of concerns and coupling
- Assess integration with existing systems
- Review scalability and extensibility

### 4. Documentation Review
Follow the **documentation-review** skill (`.agent/skills/code-review/documentation/SKILL.md`):
- Verify comments and docstrings
- Check file headers and structure
- Ensure project-specific standards adherence (see AGENTS.md)
- Confirm completeness and accuracy

### 5. Aggregate and Report
- Merge all findings
- Categorize each issue: **Critical** (must fix) | **Important** (should fix) | **Suggestion** (nice to have)
- Acknowledge what was done well before highlighting issues
- For plan deviations: state whether they are problematic or beneficial
- Provide clear, actionable guidance for each issue with file/line references where applicable

## Notes
- Extend the sub-skills when new review dimensions or project patterns are adopted
- Keep feedback structured, thorough, and concise
