"""
CrewAI-powered sentiment analysis for expressive TTS.

Uses a multi-agent workflow with three specialized agents:
1. Emotional Analyst - Analyzes text for emotional subtext and pacing
2. Voice Director - Converts text + mood map into SSML with prosody tags
3. SSML Critic - Validates SSML structure and checks for robotic pacing

The agents work sequentially to produce cinematic, emotionally-aware SSML.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from crewai import Crew

from .engine import EmotionSegment, parse_ssml_to_segments, generate_audio_with_emotions

from agents.emotional_analyst import emotional_analyst, analysis_task
from agents.ssml_transcriber import ssml_transcriber, ssml_task
from agents.ssml_critic import ssml_critic, validation_task
from agents.styletts_interpreter import styletts_interpreter, style_parameters_task
from agents.utils import local_llm


def split_text_smartly(text: str, max_chunk_size: int = 1000) -> list[str]:
    """
    Split text into chunks while preserving sentence and paragraph boundaries.
    
    This ensures the Emotional Analyst receives text with natural structure,
    which helps it produce better mood maps.
    
    Args:
        text: The full text to split
        max_chunk_size: Maximum characters per chunk (soft limit)
    
    Returns:
        List of text chunks with preserved structure
    """
    # Split by paragraphs first (preserves major structure)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If the paragraph itself is larger than max_chunk_size, we MUST split it
        if len(paragraph) > max_chunk_size:
            # If we had a pending chunk, save it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split the large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Save the last part of the split paragraph
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            continue

        # If adding this paragraph would exceed the limit
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            # Save current chunk and start new one with this paragraph
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]


def process_chapter_with_crewai(
    text: str,
    moodmap_path: Optional[str] = None,
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 1000,
) -> tuple[List[EmotionSegment], str, str]:
    """
    Process a chapter using the CrewAI multi-agent workflow.
    Optionally updates the moodmap file incrementally if moodmap_path is provided.
    
    Args:
        text: The text to process
        moodmap_path: Optional path to the moodmap file for incremental updates
        model: Ollama model name
        ollama_url: Ollama API endpoint
        chunk_size: Maximum chunk size for processing
    
    Returns:
        Tuple of (all_segments, combined_ssml, combined_mood_map)
    """
    # Initialize LLM
    llm = local_llm(model=model, base_url=ollama_url)
    
    print(f"  Text is {len(text)} chars, splitting into paragraph-based chunks...")
    text_chunks = split_text_smartly(text, max_chunk_size=chunk_size)
    print(f"  Split into {len(text_chunks)} chunks")
    
    all_segments = []
    all_ssml_parts = []
    all_mood_maps = []
    
    # Initialize moodmap file if provided
    if moodmap_path:
        with open(moodmap_path, "w", encoding="utf-8") as f:
            f.write(f"# Mood Map for Chapter\n\n")

    for i, chunk in enumerate(text_chunks, 1):
        print(f"  Processing chunk {i}/{len(text_chunks)}...")
        
        # Create agents
        analyst_agent = emotional_analyst(llm)
        transcriber_agent = ssml_transcriber(llm)
        critic_agent = ssml_critic(llm)
        interpreter_agent = styletts_interpreter(llm)
        
        # Create tasks for this chunk with explicit context passing
        t1 = analysis_task(analyst_agent, chunk)
        t2 = ssml_task(transcriber_agent, chunk, context=[t1])
        t3 = validation_task(critic_agent, context=[t1, t2])
        t4 = style_parameters_task(interpreter_agent, context=[t1, t2, t3])
        
        # Create crew with sequential process
        crew = Crew(
            agents=[analyst_agent, transcriber_agent, critic_agent, interpreter_agent],
            tasks=[t1, t2, t3, t4],
            verbose=True
        )
        
        # Execute the workflow
        result = crew.kickoff()
        
        # Capture the mood map from the analysis task
        chunk_mood_map = str(t1.output) if hasattr(t1, 'output') else ""
        header = f"\n=== CHUNK {i}/{len(text_chunks)} ===\n\n"
        full_chunk_mood_map = header + chunk_mood_map
        all_mood_maps.append(full_chunk_mood_map)
        
        if moodmap_path:
            with open(moodmap_path, "a", encoding="utf-8") as f:
                f.write(full_chunk_mood_map + "\n\n")
            print(f"  Updated mood map: {moodmap_path}")
        
        # Capture the SSML from the critic task
        chunk_ssml = str(t3.output) if hasattr(t3, 'output') else ""
        # Remove output <speak> tags for merging logic
        chunk_ssml_clean = re.sub(r'^<speak[^>]*>\s*|\s*</speak>$', '', chunk_ssml, flags=re.DOTALL)
        all_ssml_parts.append(chunk_ssml_clean)
        
        # Parse the JSON result from the interpreter
        chunk_segments = []
        try:
            # Cleaner JSON extraction just in case
            json_str = str(result)
            # If wrapped in markdown blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            segment_data = json.loads(json_str)
            
            for item in segment_data:
                chunk_segments.append(EmotionSegment(
                    text=item.get("text", ""),
                    emotion=item.get("emotion", "neutral"),
                    alpha=float(item.get("alpha", 0.3)),
                    beta=float(item.get("beta", 0.7)),
                    diffusion_steps=int(item.get("diffusion_steps", 5)),
                    prosody=None # Interpreter handles params directly
                ))
        except Exception as e:
            print(f"Error parsing JSON from interpreter: {e}")
            print(f"Raw output: {result}")
            # Fallback to regex parsing if JSON fails (using the SSML)
            print("Falling back to regex parsing...")
            chunk_segments = parse_ssml_to_segments(f'<speak>{chunk_ssml_clean}</speak>', chunk)
        
        all_segments.extend(chunk_segments)
    
    # Combine all SSML parts and mood maps
    raw_ssml = '<speak>\n' + '\n'.join(all_ssml_parts) + '\n</speak>'
    mood_map = '\n\n'.join(all_mood_maps)
    
    return all_segments, raw_ssml, mood_map


def run_styletts2_workflow(
    text: str,
    ollama_model: str,
    voice_sample_path: str,
    output_dir: os.PathLike,
    chapter_title: str,
    chapter_filename: str,
) -> None:
    """
    Run the full StyleTTS2 workflow:
    1. Analyze text with CrewAI
    2. Save analysis artifacts (JSON, SSML, MoodMap)
    3. Generate audio using StyleTTS2
    """
    output_dir = os.path.normpath(output_dir)
    print(f"  Analyzing sentiment with CrewAI + Ollama ({ollama_model})...")
    
    moodmap_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".moodmap"))
    
    segments, raw_ssml, mood_map = process_chapter_with_crewai(
        text=text,
        moodmap_path=moodmap_path,
        model=ollama_model,
    )
    print(f"  Found {len(segments)} emotion segments")
    
    # Show emotion breakdown
    emotions = {}
    for seg in segments:
        emotions[seg.emotion] = emotions.get(seg.emotion, 0) + 1
    print(f"  Emotions: {emotions}")

    # Save sentiment analysis to JSON file with SSML
    sentiment_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".json"))
    sentiment_data = {
        "chapter_title": chapter_title,
        "segments": [seg.to_dict() for seg in segments],
        "raw_ssml": raw_ssml
    }
    with open(sentiment_path, "w", encoding="utf-8") as f:
        json.dump(sentiment_data, f, indent=2)
    print(f"  Saved sentiment analysis: {sentiment_path}")
    
    # Save raw SSML separately
    ssml_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".ssml"))
    with open(ssml_path, "w", encoding="utf-8") as f:
        f.write(raw_ssml)
    print(f"  Saved SSML: {ssml_path}")
    
    # Save mood map from the Narrative Psychologist (final version, though already saved incrementally if moodmap_path was provided)
    with open(moodmap_path, "w", encoding="utf-8") as f:
        f.write(mood_map)
    print(f"  Saved final mood map: {moodmap_path}")

    # Generate Audio
    audio_path = Path(output_dir) / chapter_filename.replace(".txt", ".wav")
    generate_audio_with_emotions(
        segments=segments,
        output_path=audio_path,
        voice_sample_path=Path(voice_sample_path),
    )

    print(f"Wrote audio: {audio_path}")


def main():
    """
    Standalone test function for the CrewAI workflow.
    Run with: python src/sentiment_crewai.py
    """
    print("=" * 80)
    print("CrewAI Emotional SSML Generator - Standalone Test")
    print("=" * 80)
    print()
    
    sample_text = """
    "I can't believe it!" she whispered, her voice trembling. "After all these years?"
    He looked away, unable to meet her gaze. "I had no choice," he replied flatly.
    """
    
    print("Sample Text:")
    print("-" * 80)
    print(sample_text.strip())
    print("-" * 80)
    print()
    
    # Process the sample text (no moodmap_path for standalone test)
    segments, raw_ssml, mood_map = process_chapter_with_crewai(
        text=sample_text.strip(),
        model="qwen2.5:14b",
    )
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print("1. MOOD MAP (from Narrative Psychologist):")
    print("-" * 80)
    print(mood_map)
    print("-" * 80)
    print()
    
    print("2. RAW SSML OUTPUT:")
    print("-" * 80)
    print(raw_ssml)
    print("-" * 80)
    print()
    
    print("3. EMOTION SEGMENTS (JSON):")
    print("-" * 80)
    output_data = {
        "chapter_title": "Sample Text",
        "segments": [seg.to_dict() for seg in segments],
        "raw_ssml": raw_ssml
    }
    print(json.dumps(output_data, indent=2))
    print("-" * 80)
    print()
    
    print(f"Total segments: {len(segments)}")
    emotions = {}
    for seg in segments:
        emotions[seg.emotion] = emotions.get(seg.emotion, 0) + 1
    print(f"Emotion breakdown: {emotions}")
    print()


if __name__ == "__main__":
    main()
