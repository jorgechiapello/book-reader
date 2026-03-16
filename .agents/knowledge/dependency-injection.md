# Dependency Injection & Component Wiring Pattern

## Goal

Enforce explicit Dependency Injection (Composition Root) instead of relying on distributed, implicit dependency resolution mechanisms or service locators.

## Directives

1. **Centralize Registries:** Define registry dictionaries (e.g., `WRITERS = {}`) explicitly in the application entrypoint (e.g., `main.py`). This serves as the single source of truth for the available implementations.
2. **Avoid Distributed Registration:** Do NOT scatter `@register_` style decorators across isolated domain modules just to build a dynamic registry. Implicit self-registration hides the total surface area of available logic from the entrypoint.
3. **Use Constructor Injection:** Core orchestration classes (like `Pipeline`) must NOT instantiate their own dependencies via internal factory logic. The entrypoint must instantiate the implementations and pass them explicitly into the conductor's constructor.

## Examples

### INCORRECT (Distributed Locator & Hidden Factory)
```python
# writers/rule_based.py
@register_writer("rule_based")
class RuleBasedWriter: ...

# pipeline.py
class Pipeline:
    @classmethod
    def build(cls, writer_name: str):
        writer = get_writer(writer_name) # Hidden dependency resolution inside domain logic
        return cls(writer)
```

### CORRECT (Composition Root Constructor Injection)
```python
# writers/rule_based.py
class RuleBasedWriter: ...

# main.py
WRITERS = {"rule_based": RuleBasedWriter}

writer_instance = WRITERS["rule_based"]()
pipeline = Pipeline(writer_instance) # Explicit constructor injection
```
