# Book Reader (Local TTS)

Local audiobook generator that parses TXT/EPUB/PDF into chapters and generates per-chapter audio using voice cloning.

## Setup

1. Create a virtual environment and install dependencies:
   - `python3.11 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Install PyTorch (required by TTS engines):
   - Follow the official instructions at https://pytorch.org/get-started/locally/

## TTS Backends

This project supports three TTS backends:

| Backend | Strengths | Use Case |
|---------|-----------|----------|
| **StyleTTS2** (default) | Superior expressiveness, style/emotion control | Local high-quality narration |
| **IndexTTS-2** | Natural language "soft instructions", duration control | Most expressive, cinematic narration |
| **Qwen-TTS** | Dialogue specialized, long scripts | Books with heavy dialogue |

## Usage

### StyleTTS2 (Local & Expressive)

    # End-to-end generation (uses CrewAI sentiment analysis by default)
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --tts-backend styletts2 run

    # Or step by step
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output ingest
    python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --tts-backend styletts2 speak

### IndexTTS-2 (Docker Server Integration)

IndexTTS-2 uses "soft instructions" (natural language descriptions) to control emotion. This implementation runs as a Docker service with the voice reference and neural network pre-loaded.

**Prerequisites:**
1. Download the IndexTTS-2 model (CosyVoice):
   ```bash
   # Create model directory
   mkdir -p pretrained_models
   
   # Download CosyVoice-300M-SFT model
   # Visit: https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT
   # Or use modelscope CLI to download automatically
   ```

2. Build and start the Docker server:
   ```bash
   # Rebuild to install new dependencies
   docker-compose build
   
   # Start the server
   docker-compose up
   ```
   This starts the IndexTTS-2 server at `http://localhost:8001` with the voice and model pre-loaded.

**Run Workflow:**

Full pipeline (ingest + speak):
```bash
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice Heisenberg --tts-backend indextts2 --ollama-model qwen2.5:14b run
```

Or step by step:
```bash
# 1. Ingest the book into chapters
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output ingest

# 2. Generate audio (requires Docker server running)
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice Heisenberg --tts-backend indextts2 --ollama-model qwen2.5:14b speak
```

**Test the workflow:**
```bash
python -m src.workflows.indextts2.workflow
```

**How it works:**
1. CrewAI agents analyze text for emotional content and generate SSML
2. Each text segment is sent to the Docker server at `http://localhost:8001/generate`
3. The server synthesizes audio using the pre-loaded IndexTTS-2 model and voice
4. Audio bytes are returned and saved locally
5. All segments are merged into the final audio file

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
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe --tts-backend styletts2 run
```

**Run with CrewAI multi-agent workflow (enabled by default):**

```bash
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice Heisenberg --tts-backend styletts2 run
```

**With a specific Ollama model:**

```bash
python src/main.py --input books/the-1000000-bank-note.pdf --output-dir output --voice joe \
    --tts-backend styletts2 \
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
- StyleTTS2 is the recommended local backend (~6GB VRAM).
- IndexTTS-2 runs as a direct Python library from a sibling repository.
- Qwen-TTS still requires ComfyUI running at `localhost:8188`.
- Sentiment analysis requires Ollama running locally (`localhost:11434`).
