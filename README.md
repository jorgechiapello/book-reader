# Book Reader (Local TTS)

Local audiobook generator that parses TXT/EPUB/PDF into chapters and generates per-chapter audio using voice cloning.

## Setup

1. Create a virtual environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip3 install -r requirements.txt`
2. Install PyTorch (required by Coqui TTS):
   - Follow the official instructions at https://pytorch.org/get-started/locally/

## Voice Samples

Place your voice sample in the `voices/` folder. The tool supports:
- `voices/your_voice.wav`
- `voices/your_voice/sample.wav`

Recommended: 1–5 minutes of clean audio, no background noise.

Example prompt to record your sample (aim for about five minutes, read it naturally):

See `VOICE_SAMPLE.md`.

## Usage

### Ingest (extract chapters)
`python src/main.py ingest --input /path/to/book.epub --output-dir /path/to/output`

### Speak (generate audio)
`python src/main.py speak --input /path/to/book.epub --output-dir /path/to/output --voice your_voice`

### End-to-end
`python src/main.py run --input /path/to/book.epub --output-dir /path/to/output --voice your_voice`

## Output Structure

```
output/
  book-slug/
    manifest.json
    chapters/
      001_chapter-1.txt
      001_chapter-1.wav
      002_chapter-2.txt
      002_chapter-2.wav
```

## Notes

- Chapter splitting is heuristic, especially for PDFs. If headings are missing, chapters are split by size.
- Audio output is WAV for reliable concatenation across chunks.
