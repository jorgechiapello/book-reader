---
name: coding-style
description: Python coding conventions and best practices for this project.
---

# Good Practices

## Constants & Default Values

- **Module-level defaults** must be declared in `UPPER_SNAKE_CASE` at the top of the file, before any function or class definition.
- Always use the `DEFAULT_` prefix for constants that represent default values (e.g. `DEFAULT_EMO_ALPHA`, not `EMO_ALPHA`).
- Never inline magic numbers or strings as function parameter defaults when the same value is reused or has semantic meaning.
- **Before finishing any file**, audit ALL function signatures and bodies for magic literals (numbers, URLs, strings, booleans) and extract them as module-level constants.

```python
# ✅ Good
DEFAULT_USE_EMO_TEXT = False
DEFAULT_EMO_ALPHA = 1
DEFAULT_INTERVAL_SILENCE = 200

def load_segments(path: Path) -> SegmentsDocument:
    use_emo_text = data.get("use_emo_text", DEFAULT_USE_EMO_TEXT)
```

```python
# ❌ Bad — wrong prefix, and literals left inline
EMO_ALPHA = 1  # missing DEFAULT_ prefix

def load_segments(path: Path) -> SegmentsDocument:
    use_emo_text = data.get("use_emo_text", False)  # magic literal
    tts_url: str = "http://localhost:8001"  # magic string
```
