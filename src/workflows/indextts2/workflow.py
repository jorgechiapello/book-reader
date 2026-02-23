import json
import os
from pathlib import Path
from typing import List

from pydub import AudioSegment

from text_utils import split_text_smartly
from .integration import generate_audio_with_indextts2
from .loader import load_segments
from .schema import Segment, SegmentsDocument, merge_segment_params


def generate_segments(
    text: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
    emo_alpha: float = 0.5,
    interval_silence: int = 200,
    chunk_size: int = 500,
) -> str:
    """
    Phase 1: Split text into segments and save *_segments.json.

    Emotion is handled by IndexTTS-2's built-in use_emo_text inference —
    no LLM calls required. Each segment is a plain prose chunk produced by
    the rule-based split_text_smartly splitter.

    Returns the path to the segments file.
    """
    output_dir = os.path.normpath(output_dir)
    print(f"  Splitting text into segments for: {chapter_title}")

    chunks = split_text_smartly(text, max_chunk_size=chunk_size)
    segments = [Segment(text=chunk) for chunk in chunks]

    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = os.path.join(output_dir, f"{artifact_base}_segments.json")

    document = {
        "use_emo_text": True,
        "emo_alpha": emo_alpha,
        "interval_silence": interval_silence,
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
    tts_url: str = "http://localhost:8001",
) -> None:
    """
    Phase 2: Load segments from JSON and synthesize audio via IndexTTS-2 server.
    Requires existing *_segments.json file.
    """
    output_dir = Path(output_dir)
    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = output_dir / f"{artifact_base}_segments.json"

    if not segments_path.exists():
        raise FileNotFoundError(
            f"Segments file not found: {segments_path}. Run 'segments' first."
        )

    doc = load_segments(segments_path)
    voice_name = Path(voice).stem if voice else "Heisenberg"
    doc = SegmentsDocument(
        segments=doc.segments,
        use_emo_text=doc.use_emo_text,
        emo_alpha=doc.emo_alpha,
        interval_silence=doc.interval_silence,
    )
    temp_files: List[Path] = []
    print(f"  Generating audio for {chapter_title} ({len(doc.segments)} segments)...")

    for idx, segment in enumerate(doc.segments):
        temp_path = output_dir / f"temp_{idx}.wav"
        params = merge_segment_params(doc, segment)

        preview = (
            f"emo_vector={params['emo_vector'][:4]}..."
            if params["emo_vector"]
            else "use_emo_text"
        )
        print(f"  [{idx + 1}/{len(doc.segments)}] {preview}")

        success = generate_audio_with_indextts2(
            text=params["text"],
            output_path=temp_path,
            voice=voice_name,
            filename=temp_path.name,
            use_emo_text=params["use_emo_text"],
            emo_alpha=params["emo_alpha"],
            interval_silence=params["interval_silence"],
            emo_vector=params["emo_vector"],
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
    emo_alpha: float = 1,
    interval_silence: int = 200,
    tts_url: str = "http://localhost:8001",
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
        interval_silence=interval_silence,
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
