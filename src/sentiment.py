"""
LLM-powered sentiment analysis for expressive TTS.

Uses Ollama to analyze text and segment it by emotion, then maps
emotions to StyleTTS2 parameters for varied expressiveness.
"""

import json
import re
from dataclasses import asdict, dataclass
from typing import List, Optional

import requests


@dataclass
class EmotionSegment:
    """A text segment with associated emotion and TTS parameters."""
    text: str
    emotion: str
    alpha: float
    beta: float
    diffusion_steps: int
    prosody: dict = None  # New: SSML-like prosody (rate, pitch)

    def to_dict(self):
        return asdict(self)


# Emotion to StyleTTS2 parameter mapping
EMOTION_PARAMS = {
    "excited": {"alpha": 0.1, "beta": 0.9, "diffusion_steps": 10, "prosody": {"rate": "medium", "pitch": "high"}},
    "sad": {"alpha": 0.5, "beta": 0.5, "diffusion_steps": 8, "prosody": {"rate": "x-slow", "pitch": "low"}},
    "calm": {"alpha": 0.3, "beta": 0.6, "diffusion_steps": 5, "prosody": {"rate": "slow", "pitch": "medium"}},
    "tense": {"alpha": 0.2, "beta": 0.8, "diffusion_steps": 10, "prosody": {"rate": "medium", "pitch": "medium"}},
    "humorous": {"alpha": 0.2, "beta": 0.7, "diffusion_steps": 7, "prosody": {"rate": "medium", "pitch": "high"}},
    "neutral": {"alpha": 0.3, "beta": 0.7, "diffusion_steps": 5, "prosody": {"rate": "slow", "pitch": "medium"}},
    "dramatic": {"alpha": 0.15, "beta": 0.85, "diffusion_steps": 10, "prosody": {"rate": "x-slow", "pitch": "medium"}},
    "hopeful": {"alpha": 0.25, "beta": 0.75, "diffusion_steps": 7, "prosody": {"rate": "slow", "pitch": "medium"}},
    "desperate": {"alpha": 0.4, "beta": 0.8, "diffusion_steps": 10, "prosody": {"rate": "medium", "pitch": "low"}},
}

SENTIMENT_PROMPT = """You are an expert at analyzing text for audiobook narration using SSML-like instructions. Your task is to:
1. Split the text into segments based on emotional tone.
2. Identify the dominant emotion for each segment.
3. Enhance the text with SSML tags for prosody (rate, pitch) and breaks.

Available emotions: excited, sad, calm, tense, humorous, neutral, dramatic, hopeful, desperate
Available SSML tags:
- <prosody rate="slow|medium|fast" pitch="low|medium|high">text</prosody>
- <break time="200ms|500ms|1s"/>
- <emphasis level="reduced|moderate|strong">text</emphasis>

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{
  "segments": [
    {
      "text": "The original text with <prosody rate='slow'>SSML tags</prosody> added where appropriate...",
      "emotion": "emotion_name"
    }
  ]
}

Rules:
- Keep segments reasonably sized (1-3 sentences each).
- Add <break/> tags for dramatic pauses or between distinct thoughts.
- Use <prosody> and <emphasis> to highlight key emotional words or phrases.
- Do not add new content, only enhance with SSML tags.

Text to analyze:
"""


def analyze_sentiment(
    text: str,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
) -> List[EmotionSegment]:
    """
    Analyze text sentiment using Ollama and return emotion-tagged segments with SSML.
    """
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": SENTIMENT_PROMPT + text,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 4096,
                }
            },
            timeout=120,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Warning: Ollama request failed: {e}")
        return [_create_segment(text, "neutral")]

    result = response.json()
    llm_output = result.get("response", "")
    return _parse_llm_response(llm_output, text)


def _parse_llm_response(llm_output: str, original_text: str) -> List[EmotionSegment]:
    """Parse LLM response and extract emotion segments."""
    json_match = re.search(r'\{[\s\S]*"segments"[\s\S]*\}', llm_output)
    
    if json_match:
        try:
            data = json.loads(json_match.group())
            segments = []
            for seg in data.get("segments", []):
                text = seg.get("text", "").strip()
                # Handle cases where emotion might be null or missing
                emotion_val = seg.get("emotion")
                emotion = emotion_val.lower() if isinstance(emotion_val, str) else "neutral"
                
                if text:
                    segments.append(_create_segment(text, emotion))
            if segments:
                return segments
        except json.JSONDecodeError:
            pass
    
    return [_create_segment(original_text, "neutral")]


def _create_segment(text: str, emotion: str) -> EmotionSegment:
    """Create an EmotionSegment with appropriate TTS parameters."""
    emotion = emotion.lower().strip()
    if emotion not in EMOTION_PARAMS:
        emotion = "neutral"
    
    params = EMOTION_PARAMS[emotion]
    
    return EmotionSegment(
        text=text,
        emotion=emotion,
        alpha=params["alpha"],
        beta=params["beta"],
        diffusion_steps=params["diffusion_steps"],
        prosody=params["prosody"]
    )


def process_chapter_with_sentiment(
    text: str,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 2000,
) -> List[EmotionSegment]:
    """Process a full chapter into emotion segments."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    all_segments: List[EmotionSegment] = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            print(f"  Analyzing sentiment for {len(current_chunk)} characters...")
            segments = analyze_sentiment(current_chunk, model, ollama_url)
            all_segments.extend(segments)
            current_chunk = para
        else:
            current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
    
    if current_chunk:
        print(f"  Analyzing sentiment for {len(current_chunk)} characters...")
        segments = analyze_sentiment(current_chunk, model, ollama_url)
        all_segments.extend(segments)
    
    return all_segments
