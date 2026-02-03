import json
import os
import re
from pathlib import Path
from typing import List, Optional

from crewai import Crew

from agents.emotional_analyst import emotional_analyst, analysis_task
from agents.ssml_transcriber import ssml_transcriber, ssml_task
from agents.ssml_critic import ssml_critic, validation_task
from agents.indextts2_interpreter import indextts2_interpreter, indextts2_task
from agents.utils import local_llm

from .engine import generate_audio_with_indextts2
from ..styletts2.workflow import split_text_smartly

def process_chapter_with_indextts2(
    text: str,
    moodmap_path: Optional[str] = None,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 1000,
) -> tuple[List[dict], str, str]:
    """
    Process a chapter using the CrewAI multi-agent workflow for IndexTTS-2.
    """
    llm = local_llm(model=model, base_url=ollama_url)
    
    print(f"  Splitting text into chunks...")
    text_chunks = split_text_smartly(text, max_chunk_size=chunk_size)
    
    all_segments = []
    all_ssml_parts = []
    all_mood_maps = []
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")
        
        # Agents
        analyst_agent = emotional_analyst(llm)
        transcriber_agent = ssml_transcriber(llm)
        critic_agent = ssml_critic(llm)
        interpreter_agent = indextts2_interpreter(llm)
        
        # Tasks
        t1 = analysis_task(analyst_agent, chunk)
        t2 = ssml_task(transcriber_agent, chunk, context=[t1])
        t3 = validation_task(critic_agent, context=[t1, t2])
        t4 = indextts2_task(interpreter_agent, context=[t1, t2, t3])
        
        crew = Crew(
            agents=[analyst_agent, transcriber_agent, critic_agent, interpreter_agent],
            tasks=[t1, t2, t3, t4],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Capture mood map
        chunk_mood_map = str(t1.output) if hasattr(t1, 'output') else ""
        header = f"\n=== CHUNK {i}/{len(text_chunks)} ===\n\n"
        all_mood_maps.append(header + chunk_mood_map)
        
        # Capture SSML
        chunk_ssml = str(t3.output) if hasattr(t3, 'output') else ""
        chunk_ssml_clean = re.sub(r'^<speak[^>]*>\s*|\s*</speak>$', '', chunk_ssml, flags=re.DOTALL)
        all_ssml_parts.append(chunk_ssml_clean)
        
        # Parse JSON results
        try:
            json_str = str(result)
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            segment_data = json.loads(json_str)
            all_segments.extend(segment_data)
        except Exception as e:
            print(f"  Error parsing JSON: {e}")
            # Fallback segment
            all_segments.append({
                "text": chunk,
                "soft_instruction": "Neutral narration",
                "emotion": "neutral",
                "role": "Narrator"
            })

    combined_ssml = '<speak>\n' + '\n'.join(all_ssml_parts) + '\n</speak>'
    combined_moodmap = '\n\n'.join(all_mood_maps)
    
    if moodmap_path:
        with open(moodmap_path, "w", encoding="utf-8") as f:
            f.write(combined_moodmap)

    return all_segments, combined_ssml, combined_moodmap


def run_indextts2_workflow(
    text: str,
    ollama_model: str,
    voice_sample_path: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
) -> None:
    """
    Full IndexTTS-2 workflow implementation.
    """
    output_dir = os.path.normpath(output_dir)
    moodmap_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".moodmap"))
    
    # 1. Analyze and Generate segments
    print(f"  Running IndexTTS-2 Analysis...")
    segments, raw_ssml, mood_map = process_chapter_with_indextts2(
        text=text,
        moodmap_path=moodmap_path,
        model=ollama_model,
    )
    
    # 2. Save Artifacts
    artifact_base = chapter_filename.replace(".txt", "")
    with open(os.path.join(output_dir, f"{artifact_base}.json"), "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "ssml": raw_ssml}, f, indent=2)
    
    with open(os.path.join(output_dir, f"{artifact_base}.ssml"), "w", encoding="utf-8") as f:
        f.write(raw_ssml)

    # 3. Generate Audio
    import soundfile as sf
    import numpy as np
    from pydub import AudioSegment
    
    temp_files = []
    print(f"  Generating audio for {len(segments)} segments...")
    
    for idx, seg in enumerate(segments):
        temp_path = Path(output_dir) / f"temp_{idx}.wav"
        generate_audio_with_indextts2(
            text=seg["text"],
            output_path=temp_path,
            soft_instruction=seg["soft_instruction"],
            reference_audio_path=voice_sample_path
        )
        temp_files.append(temp_path)
        
    # 4. Merge Audio
    if temp_files:
        final_audio_path = Path(output_dir) / chapter_filename.replace(".txt", ".wav")
        print(f"  Merging {len(temp_files)} segments into {final_audio_path}...")
        
        combined_audio = AudioSegment.empty()
        for p in temp_files:
            seg_audio = AudioSegment.from_wav(str(p))
            combined_audio += seg_audio
            # Add a small crossfade or silence if needed?
            # combined_audio += AudioSegment.silent(duration=100)
            
        combined_audio.export(str(final_audio_path), format="wav")
        
        # Cleanup
        for p in temp_files:
            try:
                os.remove(p)
            except:
                pass
                
        print(f"  IndexTTS-2 Audio generated: {final_audio_path}")
    else:
        print("  Warning: No audio segments generated.")

if __name__ == "__main__":
    # Test script
    test_text = "It was a dark and stormy night. 'Who's there?' she cried out in terror."
    run_indextts2_workflow(
        text=test_text,
        ollama_model="qwen2.5:14b",
        voice_sample_path="voices/Narrator.wav", # Change as needed
        output_dir="output/test_indextts2",
        chapter_title="Test Chapter",
        chapter_filename="test.txt"
    )
