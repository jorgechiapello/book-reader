import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ScriptSegment:
    """A segment of text with interpretations and annotations."""
    text: str
    interpretation: str
    # Speaker is removed for now based on user feedback.


@dataclass
class ChapterScript:
    """The full script for a chapter."""
    version: int
    chapter_title: str
    summary: str
    segments: List[ScriptSegment]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ChapterScript":
        """Deserialize from JSON string."""
        parsed = json.loads(data)
        
        segments = [ScriptSegment(**s) for s in parsed.get("segments", [])]

        return cls(
            version=parsed.get("version", 1),
            chapter_title=parsed.get("chapter_title", ""),
            summary=parsed.get("summary", ""),
            segments=segments
        )

    def save(self, path: Path) -> None:
        """Save the script to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ChapterScript":
        """Load the script from a JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))
