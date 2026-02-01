---
name: Local TTS Audiobook
overview: Build a local-only pipeline that parses books into chapters and generates per-chapter audio using a voice-cloning TTS model, stored under /book/chapters.
todos:
  - id: extract-chapters
    content: Parse inputs and write chapter text files
    status: pending
  - id: generate-audio
    content: Generate audio via XTTS with voice samples
    status: pending
    dependencies:
      - extract-chapters
  - id: docs
    content: Write README with install and CLI usage
    status: pending
    dependencies:
      - generate-audio
---

# Local Voice-Cloning Audiobook Plan

## Scope
- Accept book inputs (TXT/EPUB/PDF), split into chapters, and store results under `/book/chapters` in the output directory.
- Generate per-chapter audio using a local TTS model with voice cloning (single voice to start, extendable later).
- Apply light text cleanup for readability (no rewriting).

## Key Decisions
- Local voice cloning: use XTTS-style model (e.g., Coqui XTTS) with a small reference audio sample per voice.
- No LLM rewrite step; only normalization (whitespace, punctuation fixes, paragraph breaks).
- Output structure: `{output_dir}/{book_slug}/chapters/{chapter_index}_{chapter_slug}.{audio_ext}` plus raw chapter text files.

## Implementation Outline
- Create a Python CLI with commands:
  - `ingest`: parse input, split into chapters, normalize text, write chapter text files.
  - `speak`: generate audio per chapter using XTTS and a selected voice sample.
  - `run`: do all steps end-to-end.
- Build input handlers for TXT/EPUB/PDF to extract chapter text.
- Add a simple voice registry (`voices/` folder with metadata and sample WAV).
- Save structured output under `/book/chapters` with both text and audio.

## Files to Add/Change
- `[README.md](/Users/jorgealbertochiapellosaid/repos/book-reader/README.md)`
- `[src/main.py](/Users/jorgealbertochiapellosaid/repos/book-reader/src/main.py)`
- `[src/formats.py](/Users/jorgealbertochiapellosaid/repos/book-reader/src/formats.py)`
- `[src/tts.py](/Users/jorgealbertochiapellosaid/repos/book-reader/src/tts.py)`
- `[src/voices.py](/Users/jorgealbertochiapellosaid/repos/book-reader/src/voices.py)`
- `[requirements.txt](/Users/jorgealbertochiapellosaid/repos/book-reader/requirements.txt)`

## Todos
- Implement chapter extraction, normalization, and output layout under `/book/chapters`.
- Implement XTTS voice cloning audio generation and voice registry.
- Document setup, model downloads, and usage examples.
