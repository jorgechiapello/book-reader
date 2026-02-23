"""IndexTTS-2 backend strategy."""

from .base import ChapterContext, TTSBackend


class IndexTTS2Backend(TTSBackend):
    """Strategy for the IndexTTS-2 Docker-server pipeline."""

    def segments(self, ctx: ChapterContext) -> None:
        """Phase 1: Split text into segments and save *_segments.json."""
        from workflows.indextts2.workflow import generate_segments

        generate_segments(
            text=ctx.text,
            output_dir=ctx.output_dir,
            chapter_title=ctx.chapter_title,
            chapter_filename=ctx.chapter_filename,
        )

    def synthesize(self, ctx: ChapterContext) -> None:
        """Phase 2: Synthesize WAV audio from *_segments.json via TTS server."""
        from workflows.indextts2.workflow import synthesize_from_segments

        synthesize_from_segments(
            voice=str(ctx.voice) if ctx.voice else None,
            output_dir=ctx.output_dir,
            chapter_title=ctx.chapter_title,
            chapter_filename=ctx.chapter_filename,
            tts_url=ctx.tts_url,
        )

    def run(self, ctx: ChapterContext) -> None:
        """Full pipeline: segments then synthesize."""
        self.segments(ctx)
        self.synthesize(ctx)
