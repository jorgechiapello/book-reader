from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_backends.base import VoiceBackend, SynthesisContext
    from writers.base import ScriptWriter

from typing import List


class Pipeline:
    """Orchestrates the 3-stage TTS pipeline: Ingest -> Script -> Audio."""

    def __init__(self, writer: "ScriptWriter", voice_backend: "VoiceBackend"):
        self.writer = writer
        self.voice_backend = voice_backend

    def run_stage_1(self, segments: "List[str]", output_path: Path) -> Path:
        """Stage 1: Convert pre-split segments into an annotated ChapterScript JSON."""
        return self.writer.write(segments, output_path)

    def run_stage_2(self, script_path: Path, output_path: Path, ctx: "SynthesisContext") -> Path:
        """Stage 2: Translate script to backend segments and generate audio."""
        return self.voice_backend.synthesize(script_path, output_path, ctx)

    def run_pipeline(self, segments: "List[str]", output_dir: Path, chapter_filename: str, ctx: "SynthesisContext") -> Path:
        """Run the full pipeline End-to-End."""
        script_filename = chapter_filename.replace(".txt", ".json")
        script_path = output_dir / script_filename

        audio_filename = chapter_filename.replace(".txt", ".wav")
        audio_path = output_dir / audio_filename

        print(f"--- Stage 1: Writing Script with {self.writer.__class__.__name__} ---")
        self.run_stage_1(segments, script_path)

        print(f"--- Stage 2: Synthesis with {self.voice_backend.__class__.__name__} ---")
        self.run_stage_2(script_path, audio_path, ctx)

        return audio_path


def build_pipeline(writer_name: str, synthesizer_name: str) -> Pipeline:
    """Factory to wire dependencies based on arguments."""
    from voice_backends import get_voice_backend
    from writers import get_writer

    writer = get_writer(writer_name)
    synthesizer = get_voice_backend(synthesizer_name)
    return Pipeline(writer, synthesizer)
