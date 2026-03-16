from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from script_schema import ChapterScript


class ScriptWriter(ABC):
    """
    Stage 1: Interprets raw text into an annotated script.
    
    Implementations of this strategy determine *how* the text is analyzed
    and paused, whether using LLMs (CrewAI) or fast heuristics.
    """

    @abstractmethod
    def write(self, segments: List[str], output_path: Path) -> Path:
        """
        Produce a ChapterScript JSON file from the given pre-split segments.

        Args:
            segments: List of text segments already extracted from the chapter.
            output_path: Where to save the resulting .json script.

        Returns:
            The output path for chaining.
        """
        pass
