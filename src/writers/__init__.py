from .base import ScriptWriter

def get_writer(name: str) -> ScriptWriter:
    """Factory to get a ScriptWriter implementation."""
    if name == "rule_based":
        from .rule_based import RuleBasedWriter
        return RuleBasedWriter()
    elif name == "emotional_analyst":
        from .emotional_analyst import EmotionalAnalystWriter
        return EmotionalAnalystWriter()
    else:
        raise ValueError(f"Unknown ScriptWriter: {name!r}")
