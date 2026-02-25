---
name: ai-workflows
description: Design principles for building AI agent workflows in this project. Use when designing or reviewing multi-step pipelines, CrewAI tasks, or any agentic workflow.
---

# AI Workflow Design Principles

## Task Independence

Each task in a pipeline must be **self-contained** — it should not reference, assume, or depend on the output format or existence of a later stage.

- A task receives inputs and produces outputs; it must not need to know what comes after it.
- Tasks may depend on earlier stages (e.g. segments must exist before synthesis), but never on later ones.
- This makes tasks reusable, testable in isolation, and safe to re-run independently.

```python
# ✅ Good — segment generation knows nothing about synthesis
def generate_segments(text, output_dir, chapter_filename) -> str:
    ...
    return segments_path

# ❌ Bad — segment task references the final audio output path
def generate_segments(text, output_dir, chapter_filename, final_wav_path) -> str:
    ...
```

## Pipeline Stages

When a workflow has multiple phases (e.g. ingest → segments → synthesize):

- Each phase is invokable as a standalone CLI command.
- Intermediate artifacts (e.g. `*_segments.json`) are saved to disk between phases so any stage can be re-run without repeating earlier work.
- Phases communicate via files or clearly defined data contracts, not shared in-memory state.
