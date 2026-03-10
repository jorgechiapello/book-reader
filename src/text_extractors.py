import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod


@dataclass
class Chapter:
    title: str
    segments: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenated text of all segments (for backward compatibility)."""
        return "\n".join(self.segments)

class ReaderInterface(ABC):
    @abstractmethod
    def extract_pages(self) -> List[str]:
        pass

class PyPdfReader(ReaderInterface):
    def __init__(self, path: str):
        from pypdf import PdfReader
        self.reader = PdfReader(path)

    def extract_pages(self) -> List[str]:
        return [page.extract_text() or "" for page in self.reader.pages]

class TextExtractor(ABC):
    @abstractmethod
    def extract_segments(self, text: str) -> List[str]:
        pass

class RuleBasedTextExtractor(TextExtractor):
    def __init__(self, reader: ReaderInterface):
        self.reader = reader

    def extract_chapters(self) -> List[Chapter]:
        pages = self.reader.extract_pages()
        full_text = "\n".join(pages)

        # Step 1: Detect chapter boundaries
        chapter_blocks = self._split_into_chapter_blocks(full_text)

        # Step 2: For each chapter block, derive segments
        result = []
        for i, (chapter_title, block_text) in enumerate(chapter_blocks):
            segments = self._extract_segments(block_text)
            title = f"{i+1:03d}_{chapter_title}"
            result.append(Chapter(title=title, segments=segments))

        return result

    def extract_segments(self, text: str) -> List[str]:
        return self._extract_segments(text.split("\n"))


    def _split_into_chapter_blocks(self, text: str) -> List[tuple]:
        """Split full text into (title, body) blocks at 'Chapter N' headings."""
        chapter_re = re.compile(r"^(chapter\s+\S+.*)$", re.IGNORECASE)
        lines = text.split("\n")

        blocks: List[tuple] = []
        current_title = "chapter 1"
        current_lines: List[str] = []
        found_chapter = False

        for line in lines:
            if chapter_re.match(line.strip()):
                # Save previous block
                if current_lines or found_chapter:
                    blocks.append((current_title, current_lines))
                current_title = line.strip().lower()
                current_lines = [line.strip()]  # Include the heading as first segment
                found_chapter = True
            else:
                current_lines.append(line)

        # Save last block
        if current_lines:
            blocks.append((current_title, current_lines))

        # If no chapter headings found, treat all as chapter 1
        if not found_chapter:
            blocks = [("chapter 1", lines)]

        return blocks

    def _extract_segments(self, lines: List[str]) -> List[str]:
        """
        Merge broken PDF lines into coherent segments.

        A new segment starts when:
        - The previous line ended with sentence-ending punctuation (. ! ?)
        - OR the current line starts with an uppercase letter after a non-ending line
          (indicating a new paragraph / title)
        - OR the previous line had no continuation (was very short / a title)

        Lines that are mid-sentence continuations (line ends without punctuation and
        next line starts lowercase) get joined with a space.
        """
        segments: List[str] = []
        buffer = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Blank line: flush buffer as a segment
                if buffer:
                    segments.append(buffer)
                    buffer = ""
                continue

            if not buffer:
                buffer = stripped
                continue

            prev_ends_sentence = buffer[-1] in ".!?"
            cur_starts_upper = stripped[0].isupper()

            if prev_ends_sentence or cur_starts_upper:
                # Start a new segment
                segments.append(buffer)
                buffer = stripped
            else:
                # Continue current segment (mid-sentence line break)
                buffer = buffer + " " + stripped

        if buffer:
            segments.append(buffer)

        return segments


def extract_chapters(path: Path) -> List[Chapter]:
    if path.suffix.lower() == ".pdf":
        reader = PyPdfReader(str(path))
        extractor = RuleBasedTextExtractor(reader)
        return extractor.extract_chapters()
    raise ValueError(f"Unsupported file type: {path.suffix}")
