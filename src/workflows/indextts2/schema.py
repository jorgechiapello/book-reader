from dataclasses import dataclass

@dataclass
class Segment:
    """Single segment for synthesis."""
    text: str


@dataclass
class SegmentsDocument:
    """Segments file document with top-level params and segments."""

    segments: list[Segment]
    use_emo_text: bool
    emo_alpha: float
    interval_silence: int