from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SynthesisContext:
    """Context and dependencies for audio synthesis."""
    voice_name: str | None = None
    voice_sample_path: Path | None = None
    tts_url: str | None = None


class VoiceBackend(ABC):
    """
    Stage 2: Translates an annotated script and generates audio.
    
    Implementations use an LLM to read the human-readable interpretation
    and inline annotations, generate backend-specific inputs (like emo_vectors),
    and then call the underlying TTS engine.
    """

    @abstractmethod
    def synthesize(self, script_path: Path, output_path: Path, ctx: SynthesisContext) -> Path:
        """
        Read a ChapterScript, translate it, and produce a WAV file.

        Args:
            script_path: Path to the .json script file.
            output_path: Path where the resulting .wav file should be saved.
            ctx: Synthesis context with URL/voice settings.

        Returns:
            The path to the generated .wav file.
        """
        pass
