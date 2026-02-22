"""Backend factory."""

from .base import ChapterContext, TTSBackend


def get_backend(name: str) -> TTSBackend:
    """Return the TTSBackend strategy for the given backend name."""
    if name == "indextts2":
        from .indextts2 import IndexTTS2Backend
        return IndexTTS2Backend()
    elif name == "styletts2":
        from .styletts2 import StyleTTS2Backend
        return StyleTTS2Backend()
    else:
        raise ValueError(f"Unknown TTS backend: {name!r}")


__all__ = ["ChapterContext", "TTSBackend", "get_backend"]
