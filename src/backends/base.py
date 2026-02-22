"""Strategy base class for TTS backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChapterContext:
    """All data needed to process one chapter."""

    text: str
    chapter_title: str
    chapter_filename: str
    output_dir: Path
    voice: Path | None = None
    ollama_model: str = "llama3.2"
    tts_url: str = "http://localhost:8001"


class TTSBackend(ABC):
    """
    Strategy interface for TTS backends.

    Each backend implements segments(), synthesize(), and run().
    Backends that do not support a command should raise NotImplementedError
    with a descriptive message.
    """

    @abstractmethod
    def segments(self, ctx: ChapterContext) -> None:
        """Phase 1: Produce emotion-annotated segment data for later synthesis."""
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, ctx: ChapterContext) -> None:
        """Phase 2: Generate audio from pre-computed segments."""
        raise NotImplementedError

    @abstractmethod
    def run(self, ctx: ChapterContext) -> None:
        """Full one-shot pipeline: produce segments and synthesize audio."""
        raise NotImplementedError
