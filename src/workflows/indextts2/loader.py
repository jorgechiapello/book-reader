import json
from pathlib import Path

from .schema import SegmentsDocument, Segment

DEFAULT_USE_EMO_TEXT = False
DEFAULT_EMO_ALPHA = 1
DEFAULT_INTERVAL_SILENCE = 200

def load_segments(path: Path) -> SegmentsDocument:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    use_emo_text = data.get("use_emo_text", DEFAULT_USE_EMO_TEXT)
    emo_alpha = data.get("emo_alpha", DEFAULT_EMO_ALPHA)
    interval_silence = data.get("interval_silence", DEFAULT_INTERVAL_SILENCE)
    raw_segments = data["segments"]

    segments = []
    for s in raw_segments:
        text = s.get("text", "")
        segments.append(Segment(text=text))

    return SegmentsDocument(
        segments=segments,
        use_emo_text=use_emo_text,
        emo_alpha=emo_alpha,
        interval_silence=interval_silence,
    )
