---
name: coding-style
description: Python coding conventions and best practices for this project.
---

# Good Practices

## Constants & Default Values

- **Module-level defaults** must be declared in `UPPER_SNAKE_CASE` at the top of the file, before any function or class definition.
- Never inline magic numbers or strings as function parameter defaults when the same value is reused or has semantic meaning.

```python
# ✅ Good
DEFAULT_USE_EMO_TEXT = False
DEFAULT_EMO_ALPHA = 1
DEFAULT_INTERVAL_SILENCE = 200

def load_segments(path: Path) -> SegmentsDocument:
    use_emo_text = data.get("use_emo_text", DEFAULT_USE_EMO_TEXT)
```

```python
# ❌ Bad
def load_segments(path: Path) -> SegmentsDocument:
    use_emo_text = data.get("use_emo_text", False)
```
