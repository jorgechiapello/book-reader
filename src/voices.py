from pathlib import Path
from typing import Optional


def resolve_voice_sample(voice: Optional[str], voices_dir: Path) -> Path:
    if voice is None:
        raise ValueError("Voice name or sample path is required.")

    voice_path = Path(voice)
    if voice_path.exists():
        return voice_path

    candidate_files = [
        voices_dir / f"{voice}.wav",
        voices_dir / voice / "sample.wav",
        voices_dir / voice / "sample.mp3",
    ]
    for candidate in candidate_files:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Voice sample not found for '{voice}'. "
        f"Checked: {', '.join(str(c) for c in candidate_files)}"
    )
