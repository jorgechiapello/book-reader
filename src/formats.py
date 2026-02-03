import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup
from ebooklib import epub
from pypdf import PdfReader


@dataclass
class Chapter:
    title: str
    text: str


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "book"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    normalized: List[str] = []
    previous_blank = True
    for line in lines:
        if not line:
            if not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return "\n".join(normalized).strip()


def merge_pdf_lines(text: str) -> str:
    """
    Merge PDF lines while preserving sentence boundaries.
    
    PDFs often break mid-sentence, but we want to preserve:
    - Sentence endings (. ! ?)
    - Paragraph breaks (blank lines)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    merged: List[str] = []
    buffer = ""
    
    for line in lines:
        # Blank line = paragraph break
        if not line:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append("")  # Preserve paragraph break
            continue
        
        # Handle hyphenation at line breaks
        if buffer.endswith("-") and not buffer.endswith("--"):
            buffer = buffer[:-1] + line
        # If previous line ended with sentence, start new line
        elif buffer and (buffer.endswith(".") or buffer.endswith("!") or buffer.endswith("?")):
            merged.append(buffer.strip())
            buffer = line
        # Otherwise join with space
        elif buffer:
            buffer = buffer + " " + line
        else:
            buffer = line
    
    if buffer:
        merged.append(buffer.strip())
    
    return "\n".join(merged)


def split_by_heading_markers(text: str) -> List[Tuple[str, str]]:
    heading_re = re.compile(r"^\s*(chapter|book)\s+([0-9ivxlcdm]+|\w+)\b.*", re.IGNORECASE)
    parts: List[Tuple[str, List[str]]] = []
    current_title = "Chapter 1"
    current_lines: List[str] = []
    found = False
    for line in text.split("\n"):
        if heading_re.match(line):
            found = True
            if current_lines:
                parts.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        parts.append((current_title, current_lines))
    if not found:
        return []
    return [(title, "\n".join(lines)) for title, lines in parts]


def split_fixed_size(text: str, max_chars: int = 4000) -> List[Tuple[str, str]]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        chunks.append(current.strip())
    return [(f"Section {i+1}", chunk) for i, chunk in enumerate(chunks)]


def extract_txt(path: Path) -> List[Chapter]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    normalized = normalize_text(raw)
    splits = split_by_heading_markers(normalized)
    if not splits:
        return [Chapter("Chapter 1", normalized)]
    return [Chapter(title, normalize_text(text)) for title, text in splits]


def extract_epub(path: Path) -> List[Chapter]:
    book = epub.read_epub(str(path))
    chapters: List[Chapter] = []
    for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        headings = soup.find_all(["h1", "h2", "h3"])
        if not headings:
            title = item.get_name()
            text = soup.get_text("\n", strip=True)
            if text:
                chapters.append(Chapter(title, normalize_text(text)))
            continue
        current_title = headings[0].get_text(strip=True)
        buffer: List[str] = []
        for node in soup.body.descendants if soup.body else soup.descendants:
            if getattr(node, "name", None) in ["h1", "h2", "h3"]:
                if buffer:
                    chapters.append(Chapter(current_title, normalize_text("\n".join(buffer))))
                    buffer = []
                current_title = node.get_text(strip=True)
            elif getattr(node, "name", None) is None:
                text = str(node).strip()
                if text:
                    buffer.append(text)
        if buffer:
            chapters.append(Chapter(current_title, normalize_text("\n".join(buffer))))
    return [chapter for chapter in chapters if chapter.text]


def extract_pdf(path: Path) -> List[Chapter]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    merged = merge_pdf_lines("\n\n".join(pages))
    normalized = normalize_text(merged)
    splits = split_by_heading_markers(normalized)
    if not splits:
        splits = split_fixed_size(normalized)
    return [Chapter(title, normalize_text(text)) for title, text in splits]


def extract_chapters(path: Path) -> List[Chapter]:
    if path.suffix.lower() == ".txt":
        return extract_txt(path)
    if path.suffix.lower() == ".epub":
        return extract_epub(path)
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")
