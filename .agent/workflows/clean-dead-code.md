---
description: How to orchestrate a project-wide or component-level cleanup of dead/unused code.
---

# Dead Code Cleanup Workflow

This workflow ensures unused files, functions, and variables are safely removed without breaking dynamic or external dependencies. 

## 1. Scan for Unused Code
Use `grep` or manual AST-level analysis to discover code that appears unreferenced:
- Look for unused imports, specifically ones not exported or dynamically loaded.
- Search for classes and functions that only have definitions but no `call_` or instantiation references.
- Find orphaned files in features or `/src` that are no longer imported in the `main` execution paths.

## 2. Verify Usage
Never rely solely on static analysis. Verify that the "dead" code is not:
- Loaded dynamically (`importlib`, `eval`).
- Mapped as an integration endpoint or called by external services (e.g. CLI endpoints in `main.py`).
- Part of an unfinished feature the user expressly asked to preserve.

## 3. Draft a Deletion Plan
Always group the dead code by component and present the list to the user before deleting anything.
- "I found [X, Y, Z] which appear completely unreferenced. Should I remove them?"

## 4. Execute Removal
Safely delete the isolated files (`rm` via bash) and remove dead lines of code in surviving files. Ensure all dangling imports pointing to the deleted items are also scrubbed.

## 5. Verify Project Health
Run the project's tests or evaluate `main.py` locally to ensure no implicit dependencies were broken by the cleanup process.
