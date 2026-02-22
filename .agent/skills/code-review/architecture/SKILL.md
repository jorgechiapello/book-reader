---
name: architecture-review
description: Architecture and design review checklist. Use when reviewing for SOLID, separation of concerns, coupling, integration, and scalability. Invoked by the architecture subagent.
---

# Architecture Review

Assess architecture and design. Extend this skill as new architectural concerns or patterns are adopted.

## Instructions

1. **SOLID and Design Principles**
   - Single responsibility: each module/class has one reason to change
   - Open/closed: extensible without modification
   - Liskov substitution: subtypes honor contracts
   - Interface segregation: focused interfaces
   - Dependency inversion: depend on abstractions

2. **Separation of Concerns and Coupling**
   - Check proper separation (UI, business logic, data, I/O)
   - Evaluate coupling: loose where possible, explicit where necessary
   - Flag circular dependencies or inappropriate coupling

3. **Integration with Existing Systems**
   - Verify the code integrates well with existing systems
   - Check consistency with current architecture and boundaries
   - Identify integration risks or violations of boundaries

4. **Scalability and Extensibility**
   - Assess scalability for expected load
   - Evaluate extensibility (adding features without large refactors)
   - Consider concurrency, distribution, and resource usage

## Output

Report findings with architectural context. Flag severity: Critical, Important, or Suggestion. Explain impact and suggest alternatives where applicable.
