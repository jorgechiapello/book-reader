from abc import ABC, abstractmethod
from pathlib import Path

from script_schema import ChapterScript

class ScriptWriter(ABC):
    """
    Stage 1: Interprets raw text into an annotated script.
    
    Implementations of this strategy determine *how* the text is analyzed
    and paused, whether using LLMs (CrewAI) or fast heuristics.
    """

    @abstractmethod
    def write(self, text: str, output_path: Path) -> Path:
        """
        Produce a ChapterScript JSON file from the given raw text.

        Args:
            text: The raw chapter text.
            output_path: Where to save the resulting .json script.

        Returns:
            The output path for chaining.
        """
        pass
