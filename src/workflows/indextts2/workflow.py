import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterator, List

from crewai import Crew
from pydub import AudioSegment

from agents.emotional_analyst import analysis_task, emotional_analyst
from agents.indextts2_interpreter import (
    indextts2_interpreter,
    indextts2_retry_task,
    indextts2_task,
)
from agents.utils import local_llm

from ..styletts2.workflow import split_text_smartly
from .integration import generate_audio_with_indextts2
from .loader import load_segments
from .schema import Segment, SegmentsDocument, merge_segment_params, normalize_emo_vector, validate_emo_vector

MAX_EMO_VECTOR_RETRIES = 2


def _parse_interpreter_json(json_str: str) -> list[dict] | None:
    """Parse JSON from interpreter output. Returns None on failure."""
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _validate_and_retry_segment(
    segment: dict,
    interpreter,
    t1,
) -> dict:
    """Validate emo_vector; retry with interpreter if invalid. Returns corrected segment."""
    emo_vec = segment.get("emo_vector")
    if emo_vec:
        segment["emo_vector"] = normalize_emo_vector(emo_vec)
        emo_vec = segment["emo_vector"]
        
    valid, error = validate_emo_vector(emo_vec)
    if valid:
        return segment

    for attempt in range(MAX_EMO_VECTOR_RETRIES):
        feedback = error or "Invalid emo_vector."
        retry_task = indextts2_retry_task(
            interpreter, segment, feedback, context=[t1]
        )
        crew = Crew(agents=[interpreter], tasks=[retry_task], verbose=True)
        result = crew.kickoff()
        parsed = _parse_interpreter_json(str(result))
        if parsed and len(parsed) >= 1:
            fixed = parsed[0]
            fixed_vec = fixed.get("emo_vector")
            if fixed_vec:
                fixed["emo_vector"] = normalize_emo_vector(fixed_vec)
                fixed_vec = fixed["emo_vector"]
                
            valid, err = validate_emo_vector(fixed_vec)
            if valid:
                print(f"  ✓ Fixed invalid emo_vector on attempt {attempt+1}")
                return fixed
            else:
                error = err
        else:
            error = "Could not parse corrected segment."

    print(f"  ⚠ emo_vector invalid after {MAX_EMO_VECTOR_RETRIES} retries, using fallback")
    return {**segment, "emo_vector": None}


def process_chapter_with_indextts2(
    text: str,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 1000,
) -> Iterator[List[dict]]:
    """
    Process text using 2-agent CrewAI workflow:
    1. Emotional Analyst - Mood Map
    2. IndexTTS-2 Interpreter - segments with emo_vector
    Validates emo_vector and retries invalid segments.
    """
    llm = local_llm(model=model, base_url=ollama_url)

    print("  Splitting text into chunks...")
    text_chunks = split_text_smartly(text, max_chunk_size=chunk_size)

    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")

        analyst = emotional_analyst(llm)
        interpreter = indextts2_interpreter(llm)

        t1 = analysis_task(analyst, chunk)
        t2 = indextts2_task(interpreter, context=[t1])

        crew = Crew(agents=[analyst, interpreter], tasks=[t1, t2], verbose=True)
        result = crew.kickoff()

        parsed = _parse_interpreter_json(str(result))
        if not parsed:
            print(f"  ⚠ Error parsing JSON from chunk {i}, using fallback")
            yield [{
                "text": chunk,
                "emo_vector": None,
                "role": "Narrator",
            }]
            continue

        chunk_segments = []
        for seg in parsed:
            validated = _validate_and_retry_segment(seg, interpreter, t1)
            chunk_segments.append(validated)

        print(f"  ✓ Generated {len(parsed)} segments from chunk {i}")
        yield chunk_segments


def generate_segments(
    text: str,
    ollama_model: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
    use_emo_text: bool = False,
    emo_alpha: float = 1,
    interval_silence: int = 200,
) -> str:
    """
    Phase 1: Generate emotional segments using CrewAI. Saves *_segments.json.
    Returns path to the segments file.
    """
    output_dir = os.path.normpath(output_dir)
    print(f"  Running 2-agent workflow for: {chapter_title}")

    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = os.path.join(output_dir, f"{artifact_base}_segments.json")
    
    raw_segments: list[dict] = []
    
    # Process chunks and save incrementally
    for chunk_segments in process_chapter_with_indextts2(text, model=ollama_model):
        raw_segments.extend(chunk_segments)
        
        document = {
            "use_emo_text": use_emo_text,
            "emo_alpha": emo_alpha,
            "interval_silence": interval_silence,
            "segments": raw_segments,
        }
        
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)

    print(f"  ✓ Saved {len(raw_segments)} segments to: {segments_path}")
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
    ollama_model: str,
    voice: str | None,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
    use_emo_text: bool = False,
    emo_alpha: float = 1,
    interval_silence: int = 200,
    tts_url: str = "http://localhost:8001",
) -> None:
    """
    Full IndexTTS-2 workflow: generate segments, then synthesize audio.
    """
    generate_segments(
        text=text,
        ollama_model=ollama_model,
        output_dir=output_dir,
        chapter_title=chapter_title,
        chapter_filename=chapter_filename,
        use_emo_text=use_emo_text,
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
        ollama_model="qwen2.5:14b",
        voice="Heisenberg",
        output_dir="output/test_indextts2",
        chapter_title="Test Chapter",
        chapter_filename="test.txt",
    )
