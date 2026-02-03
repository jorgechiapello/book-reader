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
from typing import List

from crewai import Crew

from .tts_styletts2 import EmotionSegment, parse_ssml_to_segments, generate_audio_with_emotions

from agents.emotional_analyst import emotional_analyst, analysis_task
from agents.ssml_transcriber import ssml_transcriber, ssml_task
from agents.ssml_critic import ssml_critic, validation_task
from agents.styletts_interpreter import styletts_interpreter, style_parameters_task
from agents.utils import local_llm

















def split_text_smartly(text: str, max_chunk_size: int = 3000) -> list[str]:
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
        # If adding this paragraph would exceed the limit
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            # Try to split the paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            # Add sentences to current chunk until we hit the limit
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    # Save current chunk and start new one
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
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
    model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 3000,
) -> tuple[List[EmotionSegment], str, str]:
    """
    Process a chapter using the CrewAI multi-agent workflow.
    
    Args:
        text: The text to process
        model: Ollama model name
        ollama_url: Ollama API endpoint
        chunk_size: Maximum chunk size for processing (soft limit)
    
    Returns:
        Tuple of (emotion_segments, raw_ssml, mood_map)
    """
    # Initialize LLM
    llm = local_llm(model=model, base_url=ollama_url)
    
    # Check if text needs to be split
    if len(text) > chunk_size:
        print(f"  Text is {len(text)} chars, splitting into chunks preserving sentence/paragraph boundaries...")
        text_chunks = split_text_smartly(text, max_chunk_size=chunk_size)
        print(f"  Split into {len(text_chunks)} chunks")
        
        # Process each chunk separately
        all_segments = []
        all_ssml_parts = []
        all_mood_maps = []
        
        for i, chunk in enumerate(text_chunks, 1):
            print(f"  Processing chunk {i}/{len(text_chunks)}...")
            
            # Create agents
            analyst_agent = emotional_analyst(llm)
            transcriber_agent = ssml_transcriber(llm)
            critic_agent = ssml_critic(llm)
            interpreter_agent = styletts_interpreter(llm)
            
            # Create tasks for this chunk
            t1 = analysis_task(analyst_agent, chunk)
            t2 = ssml_task(transcriber_agent, chunk)
            t3 = validation_task(critic_agent)
            t4 = style_parameters_task(interpreter_agent)
            
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
            all_mood_maps.append(f"\n=== CHUNK {i}/{len(text_chunks)} ===\n\n{chunk_mood_map}")
            
            # Capture the SSML from the critic task
            chunk_ssml = str(t3.output) if hasattr(t3, 'output') else ""
            # Remove output <speak> tags for merging logic
            chunk_ssml_clean = re.sub(r'^<speak[^>]*>\s*|\s*</speak>$', '', chunk_ssml, flags=re.DOTALL)
            all_ssml_parts.append(chunk_ssml_clean)
            
            # Parse the JSON result from the interpreter
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
                    all_segments.append(EmotionSegment(
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
        segments = all_segments
        
    else:
        # Process as single chunk (original behavior)
        print(f"  Processing text as single chunk ({len(text)} chars)...")
        
        # Create agents
        analyst_agent = emotional_analyst(llm)
        transcriber_agent = ssml_transcriber(llm)
        critic_agent = ssml_critic(llm)
        interpreter_agent = styletts_interpreter(llm)
        
        # Create tasks
        t1 = analysis_task(analyst_agent, text)
        t2 = ssml_task(transcriber_agent, text)
        t3 = validation_task(critic_agent)
        t4 = style_parameters_task(interpreter_agent)
        
        # Create crew with sequential process
        crew = Crew(
            agents=[analyst_agent, transcriber_agent, critic_agent, interpreter_agent],
            tasks=[t1, t2, t3, t4],
            verbose=True
        )
        
        # Execute the workflow
        print("  Starting CrewAI multi-agent workflow...")
        result = crew.kickoff()
        
        # Capture the mood map from the analysis task
        mood_map = str(t1.output) if hasattr(t1, 'output') else ""
        
        # Capture SSML from critic
        raw_ssml = str(t3.output) if hasattr(t3, 'output') else ""
        
        # Ensure SSML has proper <speak> wrapper
        if not raw_ssml.startswith('<speak>'):
            raw_ssml = f'<speak>\n{raw_ssml}\n</speak>'
            
        # Parse the JSON result from the interpreter
        try:
            # Cleaner JSON extraction just in case
            json_str = str(result)
            # If wrapped in markdown blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            segment_data = json.loads(json_str)
            
            segments = []
            for item in segment_data:
                segments.append(EmotionSegment(
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
            segments = parse_ssml_to_segments(raw_ssml, text)
    
    return segments, raw_ssml, mood_map


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
    
    segments, raw_ssml, mood_map = process_chapter_with_crewai(
        text=text,
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
    
    # Save mood map from the Narrative Psychologist
    moodmap_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".moodmap"))
    with open(moodmap_path, "w", encoding="utf-8") as f:
        f.write(mood_map)
    print(f"  Saved mood map: {moodmap_path}")

    # Generate Audio
    audio_path = os.path.join(output_dir, chapter_filename.replace(".txt", ".wav"))
    generate_audio_with_emotions(
        segments=segments,
        output_path=audio_path,
        voice_sample_path=voice_sample_path,
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
    
    # Process the sample text
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
