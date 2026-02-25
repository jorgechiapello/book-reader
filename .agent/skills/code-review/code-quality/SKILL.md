---
name: code-quality-review
description: Code quality assessment checklist. Use when reviewing code for patterns, error handling, maintainability, tests, security, and performance. Invoked by the code-quality subagent.
---

# Code Quality Review

Assess code quality across patterns, robustness, organization, tests, and risks. Extend this skill as new quality criteria emerge.

## Instructions

1. **Patterns and Conventions**
   - Check adherence to established patterns and conventions in the codebase
   - Identify inconsistent styles or anti-patterns

2. **Error Handling and Defensive Programming**
   - Verify proper error handling (try/except, error boundaries, fallbacks)
   - Check for type safety where applicable
   - Look for missing validation, null checks, and edge-case handling

3. **Organization and Maintainability**
   - Evaluate code organization (modules, functions, responsibilities)
   - Assess naming conventions (clarity, consistency, intent-revealing)
   - Check for duplication and opportunities for abstraction

4. **Test Coverage and Quality**
   - Assess test coverage for changed or new code
   - Evaluate quality of test implementations (meaningful assertions, isolation)
   - Flag untested critical paths

5. **Security and Performance**
   - Look for potential security vulnerabilities (injection, exposure of secrets, unsafe parsing)
   - Identify performance issues (N+1 queries, unnecessary allocations, blocking calls)

6. **Dead Code**
   - Flag no-op reassignments (e.g., re-wrapping a value in the same type it already is)
   - Identify unused variables, imports, and unreachable branches
   - Remove leftover refactor artifacts (commented-out code, orphaned helpers)

## Output

Report findings with concrete file/line references. Flag severity: Critical, Important, or Suggestion. Include actionable recommendations.
