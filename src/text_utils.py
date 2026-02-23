"""Shared text utilities for TTS workflows."""

import re


def split_text_smartly(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Split text into chunks while preserving sentence and paragraph boundaries.

    Splits by paragraphs first (double newlines), then falls back to sentence
    boundaries (after . ! ?) if a paragraph exceeds max_chunk_size.

    Args:
        text: The full text to split.
        max_chunk_size: Maximum characters per chunk (soft limit).

    Returns:
        List of text chunks with preserved structure.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            continue

        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk = (current_chunk + "\n\n" + paragraph).strip() if current_chunk else paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]
