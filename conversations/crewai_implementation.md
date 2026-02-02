# CrewAI Emotional SSML Implementation

**Status**: ✅ Complete  
**Date**: February 2, 2026

## Overview

Successfully implemented a CrewAI-based multi-agent workflow for generating emotional, paced SSML from raw book text. This provides an alternative sentiment analysis backend alongside the existing direct Ollama approach.

## Implementation Details

### Files Added/Modified

1. **requirements.txt** - Added `crewai` and `litellm` dependencies
2. **src/sentiment_crewai.py** (NEW) - Complete 3-agent CrewAI workflow
3. **src/main.py** - Added `--sentiment-backend` flag with ollama/crewai options
4. **README.md** - Updated documentation with CrewAI usage examples

### Three-Agent Architecture

```
Raw Text → Emotional Analyst → Voice Director → SSML Critic → Final SSML
                ↓                    ↓                ↓
           Mood Map         Initial SSML      Validated SSML
```

1. **Emotional Analyst** (Narrative Psychologist)
   - Analyzes emotional subtext and narrative pacing
   - Creates detailed mood map with emotions, intensity, pacing notes
   
2. **Voice Director** (Audiobook Voice Director)
   - Converts text + mood map into SSML markup
   - Uses `<prosody>`, `<break>`, `<emphasis>` tags for cinematic delivery
   
3. **SSML Critic** (SSML Quality Auditor)
   - Validates markup structure and naturalness
   - Checks for robotic patterns and missing emotional beats

### Output Format

The workflow produces **both**:

1. **Raw SSML** (`.ssml` file) - Complete `<speak>` document for TTS
2. **Metadata JSON** (`.json` file) - Emotion segments with alpha/beta params for StyleTTS2

### Usage

**Direct Ollama (existing, fast)**:
```bash
python src/main.py --input book.pdf --tts-backend styletts2 \
  --sentiment --sentiment-backend ollama run
```

**CrewAI Multi-Agent (new, sophisticated)**:
```bash
python src/main.py --input book.pdf --tts-backend styletts2 \
  --sentiment --sentiment-backend crewai run
```

**Standalone Test**:
```bash
python src/sentiment_crewai.py
```

### Test Results

Successfully tested with sample text from "The Million Pound Bank Note":
- ✅ Generated valid SSML with prosody tags
- ✅ Created 3 emotion segments (neutral, dramatic, excited)
- ✅ Proper JSON format compatible with existing pipeline
- ✅ All three agents completed their tasks sequentially

### Key Features

- **No OpenAI API Key Required** - Uses local Ollama via LiteLLM
- **Iterative Refinement** - Critic can request revisions from Director
- **Dual Output** - Both SSML and JSON for flexibility
- **Backward Compatible** - Works with existing StyleTTS2 pipeline
- **Extensible** - Easy to add more agents (e.g., Pacing Coach, Character Voice Selector)

## Configuration

- Model: `llama3.2:3b` (default, tested)
- Ollama URL: `http://localhost:11434`
- API Key: Set to "NA" (bypasses OpenAI requirement)

## Performance Notes

- CrewAI approach is slower than direct Ollama (multi-pass vs single-pass)
- Provides more sophisticated SSML with better dramatic pacing
- Suitable for high-quality audiobook production where expressiveness matters
- Recommended for narrative fiction with emotional depth
