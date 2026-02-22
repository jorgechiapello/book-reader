"""StyleTTS2 backend strategy."""

from .base import ChapterContext, TTSBackend


class StyleTTS2Backend(TTSBackend):
    """
    Strategy for the StyleTTS2 local pipeline.

    StyleTTS2 processes emotion inline during synthesis; there is no separate
    segment-generation phase. Calling segments() or synthesize() independently
    is not supported.
    """

    def segments(self, ctx: ChapterContext) -> None:
        raise NotImplementedError(
            "'segments' is not supported by StyleTTS2Backend — "
            "emotion analysis runs inline during 'run'."
        )

    def synthesize(self, ctx: ChapterContext) -> None:
        raise NotImplementedError(
            "'synthesize' is not supported by StyleTTS2Backend — "
            "use 'run' to generate audio in one step."
        )

    def run(self, ctx: ChapterContext) -> None:
        """One-shot pipeline: analyse emotion and synthesize audio."""
        from workflows.styletts2.workflow import run_styletts2_workflow

        run_styletts2_workflow(
            text=ctx.text,
            ollama_model=ctx.ollama_model,
            voice_sample_path=str(ctx.voice) if ctx.voice else None,
            output_dir=ctx.output_dir,
            chapter_title=ctx.chapter_title,
            chapter_filename=ctx.chapter_filename,
        )
