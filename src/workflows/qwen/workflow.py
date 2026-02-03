import json
import os
import re
from pathlib import Path
from typing import List, Optional

from crewai import Crew

from agents.emotional_analyst import emotional_analyst, analysis_task
from agents.qwen_interpreter import qwen_interpreter, qwen_script_task
from agents.utils import local_llm
from .engine import generate_audio_with_qwen
from ..styletts2.workflow import split_text_smartly


def process_chapter_with_qwen(
    text: str,
    moodmap_path: Optional[str] = None,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 1500, # Qwen can handle larger scripts
) -> tuple[str, str]:
    """
    Process a chapter using the CrewAI multi-agent workflow for Qwen-TTS.
    """
    llm = local_llm(model=model, base_url=ollama_url)
    
    print(f"  Splitting text into chunks...")
    text_chunks = split_text_smartly(text, max_chunk_size=chunk_size)
    
    all_scripts = []
    all_mood_maps = []
    
    # Agents
    analyst_agent = emotional_analyst(llm)
    interpreter_agent = qwen_interpreter(llm)

    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")
        
        # Tasks
        t1 = analysis_task(analyst_agent, chunk)
        t2 = qwen_script_task(interpreter_agent, context=[t1])
        
        crew = Crew(
            agents=[analyst_agent, interpreter_agent],
            tasks=[t1, t2],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Capture outputs
        chunk_mood_map = str(t1.output) if hasattr(t1, 'output') else ""
        header = f"\n=== CHUNK {i}/{len(text_chunks)} ===\n\n"
        all_mood_maps.append(header + chunk_mood_map)
        
        # Qwen script segment
        script_part = str(result)
        # Clean up possible markdown
        if "```" in script_part:
            script_part = script_part.split("```")[1].split("```")[0].strip()
        all_scripts.append(script_part)

    combined_script = "\n".join(all_scripts)
    combined_moodmap = "\n\n".join(all_mood_maps)
    
    if moodmap_path:
        with open(moodmap_path, "w", encoding="utf-8") as f:
            f.write(combined_moodmap)

    return combined_script, combined_moodmap


def run_qwen_workflow(
    text: str,
    ollama_model: str,
    voice_sample_path: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
) -> None:
    """
    Full Qwen-TTS workflow.
    """
    output_dir = os.path.normpath(output_dir)
    moodmap_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".moodmap"))
    
    # 1. Generate Script
    print(f"  Generating Qwen-TTS Script...")
    script, mood_map = process_chapter_with_qwen(
        text=text,
        moodmap_path=moodmap_path,
        model=ollama_model,
    )
    
    # 2. Save Script for debugging
    script_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".qwen_script"))
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Saved Qwen script: {script_path}")

    # 3. Generate Audio
    audio_path = Path(output_dir) / chapter_filename.replace(".txt", ".wav")
    generate_audio_with_qwen(
        script=script,
        output_path=audio_path,
        voice_name="Serena", # Preserving voice parameter logic later
        speaker_audio_path=voice_sample_path if voice_sample_path else None
    )

    print(f"Done generating Qwen audio: {audio_path}")
