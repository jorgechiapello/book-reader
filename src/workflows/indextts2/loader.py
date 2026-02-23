"""Load segments from JSON file with legacy format support."""

import json
from pathlib import Path

from .schema import SegmentsDocument, Segment


def load_segments(path: Path) -> SegmentsDocument:
    """
    Load segments from JSON file. Handles new schema and legacy flat list format.
    Returns normalized SegmentsDocument.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # New schema: top-level has "segments" key
    if "segments" in data:
        use_emo_text = data.get("use_emo_text", False)
        emo_alpha = data.get("emo_alpha", 1)
        interval_silence = data.get("interval_silence", 200)
        raw_segments = data["segments"]
    else:
        # Legacy: flat list [{text, soft_instruction, emotion, role}, ...]
        use_emo_text = True  # legacy had soft_instruction
        emo_alpha = 1
        interval_silence = 200
        raw_segments = data if isinstance(data, list) else []

    segments = []
    for s in raw_segments:
        text = s.get("text", "")
        emo_vec = s.get("emo_vector")
        interval = s.get("interval_silence")
        segments.append(
            Segment(
                text=text,
                emo_vector=emo_vec,
                interval_silence=interval,
            )
        )

    return SegmentsDocument(
        segments=segments,
        use_emo_text=use_emo_text,
        emo_alpha=emo_alpha,
        interval_silence=interval_silence,
    )
