"""IndexTTS2 segments schema and validation."""

from dataclasses import dataclass
from typing import Any

EMO_VECTOR_LENGTH = 8
EMO_LABELS = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]


@dataclass
class Segment:
    """Single segment for synthesis."""

    text: str
    emo_vector: list[float] | None = None
    interval_silence: int | None = None
    role: str = "Narrator"


@dataclass
class SegmentsDocument:
    """Segments file document with top-level params and segments."""

    segments: list[Segment]
    use_emo_text: bool = False
    emo_alpha: float = 1
    interval_silence: int = 200


def validate_emo_vector(value: Any) -> tuple[bool, str | None]:
    """
    Validate emo_vector. Returns (valid, error_message).
    If valid, error_message is None.
    """
    if value is None:
        return True, None

    if not isinstance(value, list):
        return False, "emo_vector must be an array of 8 floats or null."

    if len(value) != EMO_VECTOR_LENGTH:
        return (
            False,
            f"emo_vector must have exactly 8 values {EMO_LABELS}, got {len(value)}.",
        )

    for i, v in enumerate(value):
        if not isinstance(v, (int, float)):
            return (
                False,
                f"emo_vector values must be numbers, found {type(v).__name__} at index {i}.",
            )
        if v < 0:
            return False, f"Each emo_vector value must be >= 0, found {v} at index {i}."
        if v > 1:
            return False, f"Each emo_vector value must be <= 1, found {v} at index {i}."

    return True, None


def normalize_emo_vector(value: list[float] | None) -> list[float] | None:
    """Normalize the emotion vector so its elements sum to 1.0."""
    if not value or len(value) != EMO_VECTOR_LENGTH:
        return value
        
    total = sum(value)
    if total <= 0:
        return value
        
    return [round(v / total, 3) for v in value]


def merge_segment_params(document: SegmentsDocument, segment: Segment) -> dict:
    """Merge document defaults with segment overrides for API payload."""
    emo_vec = segment.emo_vector
    use_emo = document.use_emo_text if emo_vec is None else False
    silence = segment.interval_silence if segment.interval_silence is not None else document.interval_silence

    return {
        "text": segment.text,
        "use_emo_text": use_emo,
        "emo_alpha": document.emo_alpha,
        "interval_silence": silence,
        "emo_vector": emo_vec,
    }
