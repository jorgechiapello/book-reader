---
name: code-architecture
description: "Use this skill when designing system boundaries, refactoring code, applying design patterns, or making architectural decisions. It enforces core software principles and acts as a router to specific pattern knowledge."
---

# Code Architecture: Principles and Pattern Routing

## Goal

Ensure the system maintains a clean execution graph, high cohesion, low coupling, and explicit boundaries. This skill defines the universal principles of the codebase and routes the agent to specific `.agents/knowledge/` files for concrete implementation details.

## Instructions

When planning a refactor, designing a factory pattern, or linking core components:

1. **Start Here (Principles):** Always ensure your design adheres to the core software principles below.
2. **Consult Knowledge Base:** Identify which specific architectural pattern applies to your current task, and read the corresponding `.agents/knowledge/<pattern>.md` file for exact rules, constraints, and examples.

### Core Software Principles

- **Explicit over Implicit:** Prefer configuration, dependency wiring, and component linking to be visible and traceable from the application entrypoint, rather than hidden deep in domain logic or decorators.
- **Pure Domain Modules:** Keep business logic isolated from infrastructure. Domain modules (like `base.py`, core controllers, and schemas) should not be aware of CLI arguments, application state, or global registries.
- **Composition over Inheritance:** Build complex behaviors by composing smaller, independent classes together rather than building deep, rigid inheritance trees.

### Pattern Library (Knowledge References)

For specific implementation heuristics, you MUST look up and read the following knowledge files:

- **Dependency Injection & Centralized Wiring:** Read `.agents/knowledge/dependency-injection.md` (Covers Composition Root vs Distributed Service Locators).

## Instructions for Self-Improvement

*(Note to future `/learn` iterations: Append new overarching universal principles to the Core Software Principles section of this `SKILL.md`. When learning a specific structural pattern with before/after examples, create a new file in `.agents/knowledge/` and add the reference link above.)*
