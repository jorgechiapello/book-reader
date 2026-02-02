# Book Reader (Local TTS)

Local audiobook generator that parses TXT/EPUB/PDF into chapters and generates per-chapter audio using voice cloning.

## Setup

1. Create a virtual environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip3 install -r requirements.txt`
2. Install PyTorch (required by TTS engines):
   - Follow the official instructions at https://pytorch.org/get-started/locally/

## TTS Backends

This project supports two TTS backends:

| Backend | Strengths | Use Case |
|---------|-----------|----------|
| **XTTS** (default) | Stable, multilingual, good voice cloning | General audiobook generation |
| **StyleTTS2** | Superior expressiveness, style/emotion control | When you want varied sentiments |

## Usage

### Basic (XTTS backend)

    # End-to-end generation
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --language en run

    # Or step by step
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output ingest
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe speak

### With StyleTTS2 (more expressive)

    # Basic StyleTTS2 - uses voice sample for both voice and style
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --tts-backend styletts2 run

    # With separate style reference (e.g., excited reading)
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe \
        --tts-backend styletts2 \
        --style-ref voices/joe_excited/sample.wav \
        run

### With LLM Sentiment Analysis (most expressive)

Enable automatic emotion detection using a local LLM. The LLM analyzes the text, segments it by emotion, and applies different TTS parameters for each segment.

**Prerequisites - Install Ollama:**

    brew install ollama

**Start Ollama (before generating audio):**

    brew services start ollama

**Download a model (one-time):**

    ollama pull llama3.2:3b

**Stop Ollama (when done, to free resources):**

    brew services stop ollama

**Two sentiment backends available:**

1. **Direct Ollama (default)** - Fast, single-pass emotion analysis
2. **CrewAI Multi-Agent** - Sophisticated 3-agent workflow with iterative refinement

**Run with sentiment analysis (direct Ollama):**

```bash
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --tts-backend styletts2 --sentiment run
```

**Run with CrewAI multi-agent workflow:**

```bash
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --tts-backend styletts2 --sentiment --sentiment-backend crewai run
```

**With a specific Ollama model:**

```bash
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe \
    --tts-backend styletts2 --sentiment \
    --ollama-model llama3.2:3b \
    run
```

### CrewAI Multi-Agent Workflow

The CrewAI backend uses three specialized agents working sequentially:

1. **Emotional Analyst** - Analyzes text for subtext, mood, and narrative pacing
2. **Voice Director** - Converts text + mood map into SSML with prosody, breaks, and emphasis
3. **SSML Critic** - Validates markup and ensures natural, non-robotic pacing

This produces both raw SSML and emotion-tagged segments compatible with StyleTTS2.

**Test the CrewAI workflow standalone:**

```bash
python src/sentiment_crewai.py
```

### Recommended Ollama Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `llama3.2:3b` | ~2GB | Fast | Good | **Apple M1/M2/M3 (recommended)** |
| `llama3.2:1b` | ~1GB | Fastest | OK | Low memory systems |
| `llama3:8b-instruct-q4_0` | ~4.7GB | Medium | Best | Systems with 16GB+ RAM |
| `phi3:mini` | ~2.2GB | Fast | Good | Alternative lightweight option |

**For Apple Silicon (M1/M2/M3):** Use `llama3.2:3b` for the best balance of speed and quality.

    ollama pull llama3.2:3b

### StyleTTS2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| --style-ref | (voice sample) | Path to style reference audio |
| --style-alpha | 0.3 | 0=more target speaker, 1=more reference style |
| --style-beta | 0.7 | Style strength |
| --diffusion-steps | 5 | More steps = better quality, slower |
| --sentiment | off | Enable LLM-powered emotion analysis |
| --sentiment-backend | ollama | Backend: ollama (fast) or crewai (sophisticated) |
| --ollama-model | llama3.2 | Ollama model for sentiment analysis |

### Tips for Better Sentiment

1. **Use --sentiment flag**: Automatically varies emotion per segment based on text content.

2. **Record expressive samples**: Create different samples for different moods:
   - voices/joe_calm/sample.wav - relaxed narration
   - voices/joe_excited/sample.wav - energetic reading
   - voices/joe_dramatic/sample.wav - intense moments

3. **Use style references**: Point --style-ref to an audio that demonstrates the emotion you want.

4. **Adjust alpha/beta**: Higher alpha blends more of the style reference's emotion.

## Output Structure

    output/
      book-slug/
        manifest.json
        chapters/
          001_chapter-1.txt
          001_chapter-1.wav
          002_chapter-2.txt
          002_chapter-2.wav

## Notes

- Chapter splitting is heuristic, especially for PDFs. If headings are missing, chapters are split by size.
- Audio output is WAV for reliable concatenation across chunks.
- StyleTTS2 requires more VRAM than XTTS (~6GB vs ~4GB).
- Sentiment analysis requires Ollama running locally (localhost:11434).
