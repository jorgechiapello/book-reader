---
name: documentation-review
description: Documentation and standards review checklist. Use when reviewing comments, docstrings, headers, and adherence to project standards. Invoked by the documentation subagent.
---

# Documentation Review

Assess documentation and adherence to project standards. Extend this skill as documentation standards evolve.

## Instructions

1. **Comments and Docstrings**
   - Verify appropriate comments for complex logic
   - Check that functions, classes, and modules have clear docstrings
   - Ensure comments explain why, not just what

2. **File Headers and Structure**
   - Check that file headers are present and accurate (purpose, author, context)
   - Verify consistent structure across similar files

3. **Project-Specific Standards**
   - Ensure adherence to project coding standards (see AGENTS.md, style guides)
   - Check convention compliance (naming, formatting, layout)

4. **Completeness and Accuracy**
   - Verify documentation matches current implementation
   - Flag stale, misleading, or missing documentation

## Output

Report findings with file and section references. Flag severity: Critical, Important, or Suggestion. Provide concrete examples of improvements.
