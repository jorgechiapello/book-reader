"""
StyleTTS2 TTS backend with style/emotion and SSML support.
"""

import re
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import soundfile as sf

from dataclasses import asdict, dataclass

# Lazy import to avoid loading heavy deps if not used
_tts_model = None


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



def _get_model():
    """Lazy-load StyleTTS2 model."""
    global _tts_model
    if _tts_model is None:
        try:
            from styletts2 import tts as styletts2_tts
            _tts_model = styletts2_tts.StyleTTS2()
        except ImportError:
            raise ImportError(
                "StyleTTS2 not installed. Install with: pip install styletts2"
            )
    return _tts_model


def split_text(text: str, max_chars: int = 500) -> List[str]:
    """Split text into smaller chunks for processing, preserving SSML tags."""
    # Heuristic: split by sentence but keep tags intact
    sentences = []
    current = ""
    # Simple split by sentence-ending punctuation followed by space
    parts = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    
    for part in parts:
        if not part.strip():
            continue
        if len(current) + len(part) + 1 > max_chars and current:
            sentences.append(current.strip())
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        sentences.append(current.strip())
    return sentences


def parse_ssml_and_generate(
    model,
    text: str,
    voice_sample_path: str,
    base_alpha: float,
    base_beta: float,
    base_steps: int,
    embedding_scale: float,
    sample_rate: int = 24000
) -> np.ndarray:
    """
    Parses a string for SSML-like tags and generates audio segments.
    Currently supports: <break time="..."/>, <prosody>, <emphasis>.
    """
    # 1. Handle <break/> tags by splitting the text
    # 2. Strip other tags for the actual TTS engine (StyleTTS2 doesn't support them natively)
    # but use them to adjust alpha/beta/steps for that specific chunk.
    
    # For now, we'll implement a simplified version: 
    # - Strip all tags for the actual speech
    # - If <break time="1s"/> is found, add silence
    
    clean_text = re.sub(r'<[^>]+>', '', text).strip()
    
    # Detect breaks
    breaks = re.findall(r'<break\s+time=["\']?(\d+)(ms|s)["\']?\s*/>', text)
    
    # If there is no actual text to speak, start with an empty array
    if not clean_text:
        wav = np.array([], dtype=np.float32)
    else:
        # Generate main audio
        wav = model.inference(
            text=clean_text,
            target_voice_path=voice_sample_path,
            output_sample_rate=sample_rate,
            alpha=base_alpha,
            beta=base_beta,
            diffusion_steps=base_steps,
            embedding_scale=embedding_scale,
        )
    
    # Add silence for each break found (simplified: append to end)
    if breaks:
        for val, unit in breaks:
            duration = float(val) / 1000.0 if unit == 'ms' else float(val)
            silence = np.zeros(int(duration * sample_rate))
            wav = np.concatenate([wav, silence])
            
    return wav


def generate_audio_styletts2(
    text: str,
    output_path: Path,
    voice_sample_path: Path,
    style_ref_path: Optional[Path] = None,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
) -> None:
    """Generate audio using StyleTTS2."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _get_model()
    ref_path = style_ref_path if style_ref_path else voice_sample_path
    chunks = split_text(text)
    sample_rate = 24000

    all_audio = []
    for chunk in chunks:
        wav = parse_ssml_and_generate(
            model, chunk, str(ref_path), alpha, beta, diffusion_steps, embedding_scale, sample_rate
        )
        all_audio.append(wav)

    combined = np.concatenate(all_audio)
    sf.write(str(output_path), combined, sample_rate)


def generate_audio_with_emotions(
    segments: List["EmotionSegment"],
    output_path: Path,
    voice_sample_path: Path,
    embedding_scale: float = 1.0,
) -> None:
    """Generate audio from emotion-tagged segments with SSML support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _get_model()
    sample_rate = 24000
    all_audio = []

    for i, segment in enumerate(segments):
        print(f"  Generating segment {i+1}/{len(segments)} [{segment.emotion}]: {segment.text[:50]}...")
        
        # StyleTTS2 doesn't support SSML tags natively, so we:
        # 1. Extract breaks for silence
        # 2. Strip tags for the inference call
        # 3. Use the segment's emotion-based alpha/beta
        
        chunks = split_text(segment.text)
        for chunk in chunks:
            wav = parse_ssml_and_generate(
                model, 
                chunk, 
                str(voice_sample_path), 
                segment.alpha, 
                segment.beta, 
                segment.diffusion_steps, 
                embedding_scale, 
                sample_rate
            )
            all_audio.append(wav)

    if not all_audio:
        raise ValueError("No audio segments generated")

    combined = np.concatenate(all_audio)
    sf.write(str(output_path), combined, sample_rate)
