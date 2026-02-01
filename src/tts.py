import os
import tempfile
import wave
from pathlib import Path
from typing import Iterable, List, Optional

from TTS.api import TTS


DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def split_text(text: str, max_chars: int = 1200) -> List[str]:
    sentences = []
    current = ""
    for part in text.replace("\n", " ").split(". "):
        if not part.strip():
            continue
        sentence = part.strip()
        if not sentence.endswith("."):
            sentence += "."
        if len(current) + len(sentence) + 1 > max_chars and current:
            sentences.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        sentences.append(current.strip())
    return sentences


def concat_wavs(parts: Iterable[Path], output_path: Path) -> None:
    parts = list(parts)
    if not parts:
        raise ValueError("No audio parts to concatenate.")
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for part in parts[1:]:
        with wave.open(str(part), "rb") as wav_file:
            if wav_file.getparams() != params:
                raise ValueError("Audio parameters do not match for concatenation.")
            frames.append(wav_file.readframes(wav_file.getnframes()))
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for frame in frames:
            out.writeframes(frame)


def load_tts(model_name: str = DEFAULT_MODEL, device: Optional[str] = None) -> TTS:
    if device:
        return TTS(model_name).to(device)
    return TTS(model_name)


def generate_audio(
    text: str,
    output_path: Path,
    voice_sample_path: Path,
    language: str = "en",
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts = load_tts(model_name=model_name, device=device)
    chunks = split_text(text)
    if len(chunks) == 1:
        tts.tts_to_file(
            text=chunks[0],
            file_path=str(output_path),
            speaker_wav=str(voice_sample_path),
            language=language,
        )
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_paths: List[Path] = []
        for index, chunk in enumerate(chunks):
            temp_path = Path(temp_dir) / f"part_{index}.wav"
            tts.tts_to_file(
                text=chunk,
                file_path=str(temp_path),
                speaker_wav=str(voice_sample_path),
                language=language,
            )
            temp_paths.append(temp_path)
        concat_wavs(temp_paths, output_path)
