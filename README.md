# Book Reader (Local TTS)

Local audiobook generator that parses TXT/EPUB/PDF into chapters and generates per-chapter audio using voice cloning.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## TTS Backends

| Backend | Strengths | Use Case |
|---------|-----------|----------|
| **StyleTTS2** | Expressiveness, style/emotion control | Default, local high-quality narration |
| **IndexTTS-2** | Natural language emotion instructions | Most expressive, cinematic narration |

## Script Writers (Text Analysis)

Before generating audio, the pipeline writes a "script" (splitting text into chapters and segments, adding pacing/emotion notes).

| Writer | Description | Speed | Requirements |
|---|---|---|---|
| **`emotional_analyst`** (default) | Uses CrewAI to read the text and produce a rich, annotated script with inline pauses and voice director notes. | Slow | Ollama running locally (`localhost:11434`) |
| **`rule_based`** | Splits text by punctuation and uses sentence length to guess pacing. No LLM used. | Instant | None |

### Setting up `emotional_analyst` (Requires Ollama)

If you want the highest quality pacing and emotion, you need to run Ollama locally to power the `emotional_analyst` writer.

1. **Install Ollama:** Download from [ollama.com](https://ollama.com/) or run `brew install ollama`.
2. **Start the Server:** Open a new terminal and run `ollama serve` (or launch the Ollama app on macOS).
3. **Pull the Model:** Before running the pipeline, you MUST pull the language model used by the script writer (by default `qwen2.5:14b`):
   ```bash
   ollama pull qwen2.5:14b
   ```

*To specify a writer, use the `--writer` flag:*
```bash
python src/main.py run --output output books/the-1000000-bank-note.pdf --writer rule_based
```

## Running

### 1. Start the IndexTTS-2 Server (new terminal)

```bash
cd tts_service
bash setup_native.sh        # one-time setup
source .venv-native/bin/activate
python main.py
```

### 2. Generate Audio

Full pipeline (ingest → script → audio):
```bash
python src/main.py run --output output  books/the-1000000-bank-note.pdf --voice Heisenberg --voice-backend indextts2 
```

Step by step:
```bash
# Stage 1: Ingest book into a script
python src/main.py script --output output books/the-1000000-bank-note.pdf --writer rule_based

# Stage 2: Generate audio (server must be running)
python src/main.py audio --output output output/001_chapter-1_script.json --voice Heisenberg --voice-backend indextts2
```

### StyleTTS2 (no server needed)

```bash
python src/main.py run --output output books/the-1000000-bank-note.pdf --voice joe --voice-backend styletts2
```

## Output Structure

```
output/
  book-slug/
    chapters/
      001_chapter-1.json   # emotional script
      001_chapter-1.wav    # generated audio
```

## Notes

- Model weights (~5.9 GB) are stored in `~/tts-weights/` — run `python tts_service/download_weights.py` to download.
- Sentiment/emotion analysis requires Ollama running at `localhost:11434`.
