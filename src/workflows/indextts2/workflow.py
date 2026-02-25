import json
import os
from pathlib import Path
from typing import List

from pydub import AudioSegment

from text_utils import split_text_smartly
from .integration import generate_audio_with_indextts2
from .loader import load_segments
from .schema import Segment, SegmentsDocument

# --- Module-level defaults ---
DEFAULT_EMO_ALPHA = 0.45
DEFAULT_CHUNK_SIZE = 500
DEFAULT_TTS_URL = "http://localhost:8001"
DEFAULT_VOICE = "Heisenberg"
DEFAULT_USE_EMO_TEXT = True


def generate_segments(
    text: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
    emo_alpha: float = DEFAULT_EMO_ALPHA,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Split text into segments and save to a JSON artifact.

    Creates a segments document containing text chunks produced by rule-based
    splitting. The output includes configuration for IndexTTS-2 emotional
    inference (use_emo_text).

    Returns:
        str: The path to the generated segments file.
    """
    output_dir = os.path.normpath(output_dir)
    print(f"  Splitting text into segments for: {chapter_title}")

    chunks = split_text_smartly(text, max_chunk_size=chunk_size)
    segments = [Segment(text=chunk) for chunk in chunks]

    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = os.path.join(output_dir, f"{artifact_base}_segments.json")

    document = {
        "use_emo_text": DEFAULT_USE_EMO_TEXT,
        "emo_alpha": emo_alpha,
        "segments": [{"text": seg.text} for seg in segments],
    }

    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)

    print(f"  ✓ Saved {len(segments)} segments to: {segments_path}")
    return segments_path


def synthesize_from_segments(
    voice: str | None,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
    tts_url: str = DEFAULT_TTS_URL,
) -> None:
    """
    Load segments from a JSON file and synthesize audio via IndexTTS-2.

    Processes each segment in the provided document and merges the resulting
    audio files into a single output WAV.
    """
    output_dir = Path(output_dir)
    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = output_dir / f"{artifact_base}_segments.json"

    if not segments_path.exists():
        raise FileNotFoundError(
            f"Segments file not found: {segments_path}. Run 'segments' first."
        )

    doc = load_segments(segments_path)
    voice_name = Path(voice).stem if voice else DEFAULT_VOICE
    temp_files: List[Path] = []
    print(f"  Generating audio for {chapter_title} ({len(doc.segments)} segments)...")

    for idx, segment in enumerate(doc.segments):
        temp_path = output_dir / f"temp_{idx}.wav"

        print(f"  [{idx + 1}/{len(doc.segments)}] use_emo_text")
        print(f"  Text: {segment.text}")
        success = generate_audio_with_indextts2(
            text=segment.text,
            output_path=temp_path,
            voice=voice_name,
            filename=temp_path.name,
            use_emo_text=DEFAULT_USE_EMO_TEXT,
            emo_alpha=doc.emo_alpha,
            interval_silence=doc.interval_silence,
            tts_url=tts_url,
        )

        if success and temp_path.exists():
            temp_files.append(temp_path)
        else:
            print(f"  ⚠ Warning: Failed to generate audio for segment {idx}")

    if temp_files:
        final_audio_path = output_dir / chapter_filename.replace(".txt", ".wav")
        print(f"  Merging {len(temp_files)} segments into {final_audio_path}...")

        combined = AudioSegment.empty()
        for idx, p in enumerate(temp_files):
            try:
                seg_audio = AudioSegment.from_wav(str(p))
                combined += seg_audio
                silence_ms = (
                    doc.segments[idx].interval_silence
                    if doc.segments[idx].interval_silence is not None
                    else doc.interval_silence
                )
                if idx < len(temp_files) - 1:
                    combined += AudioSegment.silent(duration=silence_ms)
            except Exception as e:
                print(f"  ⚠ Error loading {p}: {e}")

        combined.export(str(final_audio_path), format="wav")

        for p in temp_files:
            try:
                os.remove(p)
            except Exception:
                pass

        print(f"  ✓ IndexTTS-2 Audio generated for {chapter_title}: {final_audio_path}")
    else:
        print("  ✗ Error: No audio segments generated.")


def run_indextts2_workflow(
    text: str,
    voice: str | None,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
    emo_alpha: float = DEFAULT_EMO_ALPHA,
    tts_url: str = DEFAULT_TTS_URL,
) -> None:
    """
    Full IndexTTS-2 workflow: generate segments, then synthesize audio.
    """
    generate_segments(
        text=text,
        output_dir=output_dir,
        chapter_title=chapter_title,
        chapter_filename=chapter_filename,
        emo_alpha=emo_alpha,
    )
    synthesize_from_segments(
        voice=voice,
        output_dir=output_dir,
        chapter_title=chapter_title,
        chapter_filename=chapter_filename,
        tts_url=tts_url,
    )


if __name__ == "__main__":
    test_text = """It was a dark and stormy night. The wind howled through the trees.
    'Who's there?' she cried out in terror. Her voice trembled with fear.
    Suddenly, a figure appeared in the doorway. It was only her cat."""

    run_indextts2_workflow(
        text=test_text,
        voice="Heisenberg",
        output_dir="output/test_indextts2",
        chapter_title="Test Chapter",
        chapter_filename="test.txt",
    )
