# Agent Guidelines

## Dependency Management
- All installed packages MUST be added to `requirements.txt` with their exact version.
- When adding a new package, run `pip freeze | grep package_name` to get the version and append it.

## Lessons Learned (Knowledge Base)

> This section is dynamically appended by the `/learn` self-improvement workflow.
> The agent will inject critical contextual findings here that should govern all future interactions.
> **DO NOT** manually edit entries below this line — they are managed by the Kaizen process.

<!-- Kaizen entries will be appended below this line -->

- **Cross-reference completeness:** When creating components that are designed to work together (e.g., a workflow that invokes skills), always verify explicit cross-references exist before delivering. Do not rely on semantic/implicit activation as the sole integration mechanism.
  - *Trigger:* Creating multiple `.agents/` artifacts in the same session that reference each other.

- **Full-file pattern sweep:** When editing a repeated pattern in documentation or code (e.g., CLI argument order), scan the ENTIRE file for all occurrences of the same pattern — not just the line the user pointed at.
  - *Trigger:* User asks to fix a specific line in a file that contains similar lines elsewhere.

