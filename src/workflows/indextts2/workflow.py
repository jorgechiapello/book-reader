import json
import os
from pathlib import Path
from typing import List
from pydub import AudioSegment

from crewai import Crew

from agents.emotional_analyst import emotional_analyst, analysis_task
from agents.indextts2_interpreter import indextts2_interpreter, indextts2_task
from agents.utils import local_llm

from .integration import generate_audio_with_indextts2
from ..styletts2.workflow import split_text_smartly


def process_chapter_with_indextts2(
    text: str,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 1000,
) -> List[dict]:
    """
    Process text using 2-agent CrewAI workflow:
    1. Emotional Analyst - analyzes text for emotional content
    2. IndexTTS-2 Interpreter - converts analysis into soft instructions
    """
    llm = local_llm(model=model, base_url=ollama_url)
    
    print(f"  Splitting text into chunks...")
    text_chunks = split_text_smartly(text, max_chunk_size=chunk_size)
    
    all_segments = []
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")
        
        # Create agents
        analyst = emotional_analyst(llm)
        interpreter = indextts2_interpreter(llm)
        
        # Create tasks
        t1 = analysis_task(analyst, chunk)
        t2 = indextts2_task(interpreter, context=[t1])
        
        # Run crew
        crew = Crew(
            agents=[analyst, interpreter],
            tasks=[t1, t2],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Parse JSON results from interpreter
        try:
            json_str = str(result)
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            segment_data = json.loads(json_str)
            all_segments.extend(segment_data)
            print(f"  ✓ Generated {len(segment_data)} segments from chunk {i}")
        except Exception as e:
            print(f"  ⚠ Error parsing JSON from chunk {i}: {e}")
            # Fallback segment
            all_segments.append({
                "text": chunk,
                "soft_instruction": "Neutral narration",
                "emotion": "neutral",
                "role": "Narrator"
            })
    
    return all_segments


def generate_segments(
    text: str,
    ollama_model: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
) -> str:
    """
    Phase 1: Generate emotional segments using CrewAI. Saves *_segments.json.
    Returns path to the segments file.
    """
    output_dir = os.path.normpath(output_dir)
    print(f"  Running 2-agent workflow for: {chapter_title}")
    segments = process_chapter_with_indextts2(text, model=ollama_model)
    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = os.path.join(output_dir, f"{artifact_base}_segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)
    print(f"  ✓ Saved {len(segments)} segments to: {segments_path}")
    return segments_path


def synthesize_from_segments(
    voice_sample_path: str | None,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
) -> None:
    """
    Phase 2: Load segments from JSON and synthesize audio via IndexTTS-2 server.
    Requires existing *_segments.json file.
    """
    output_dir = Path(output_dir)
    artifact_base = chapter_filename.replace(".txt", "")
    segments_path = output_dir / f"{artifact_base}_segments.json"
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_path}. Run 'segments' first.")

    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    voice_name = Path(voice_sample_path).stem if voice_sample_path else None
    temp_files = []
    print(f"  Generating audio for {len(segments)} segments...")

    for idx, seg in enumerate(segments):
        temp_path = output_dir / f"temp_{idx}.wav"
        instruction = seg.get("soft_instruction", "Neutral narration")
        print(f"  [{idx+1}/{len(segments)}] {instruction[:50]}...")

        success = generate_audio_with_indextts2(
            text=seg["text"],
            output_path=temp_path,
            voice=voice_name,
            soft_instruction=instruction,
        )

        if success and temp_path.exists():
            temp_files.append(temp_path)
        else:
            print(f"  ⚠ Warning: Failed to generate audio for segment {idx}")

    if temp_files:
        final_audio_path = output_dir / chapter_filename.replace(".txt", ".wav")
        print(f"  Merging {len(temp_files)} segments into {final_audio_path}...")

        combined_audio = AudioSegment.empty()
        for p in temp_files:
            try:
                seg_audio = AudioSegment.from_wav(str(p))
                combined_audio += seg_audio
                combined_audio += AudioSegment.silent(duration=200)
            except Exception as e:
                print(f"  ⚠ Error loading {p}: {e}")

        combined_audio.export(str(final_audio_path), format="wav")

        for p in temp_files:
            try:
                os.remove(p)
            except Exception:
                pass

        print(f"  ✓ IndexTTS-2 Audio generated: {final_audio_path}")
    else:
        print("  ✗ Error: No audio segments generated.")


def run_indextts2_workflow(
    text: str,
    ollama_model: str,
    voice_sample_path: str | None,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
) -> None:
    """
    Full IndexTTS-2 workflow: generate segments, then synthesize audio.
    """
    generate_segments(text, ollama_model, output_dir, chapter_title, chapter_filename)
    synthesize_from_segments(voice_sample_path, output_dir, chapter_title, chapter_filename)


if __name__ == "__main__":
    # Test script
    test_text = """It was a dark and stormy night. The wind howled through the trees.
    'Who's there?' she cried out in terror. Her voice trembled with fear.
    Suddenly, a figure appeared in the doorway. It was only her cat."""
    
    run_indextts2_workflow(
        text=test_text,
        ollama_model="qwen2.5:14b",
        voice_sample_path="voices/Heisenberg.wav",
        output_dir="output/test_indextts2",
        chapter_title="Test Chapter",
        chapter_filename="test.txt"
    )
