from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chapter:
    title: str
    segments: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenated text of all segments (for backward compatibility)."""
        return "\n".join(self.segments)


class TextExtractor(ABC):
    @abstractmethod
    def extract_segments(self, text: str) -> List[str]:
        pass
