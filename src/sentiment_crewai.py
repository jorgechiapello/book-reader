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
from dataclasses import asdict, dataclass
from typing import List, Optional

from crewai import Agent, Crew, LLM, Task

from tts_styletts2 import EmotionSegment, parse_ssml_to_segments

from agents.emotional_analyst import emotional_analyst, analysis_task
from agents.voice_director import voice_director, ssml_task
from agents.ssml_critic import ssml_critic, validation_task
from agents.utils import local_llm


# Sample dramatic text for testing
SAMPLE_TEXT = """
Now, something had been happening there a little before, which I did not 
know anything about until a good many days afterwards, but I will tell you 
about it now. Those two old brothers had been having a pretty hot argument 
a couple of days before, and had ended by agreeing to decide it by a bet, 
which is the English way of settling everything.
"""



















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
            director_agent = voice_director(llm)
            critic_agent = ssml_critic(llm)
            
            # Create tasks for this chunk
            t1 = analysis_task(analyst_agent, chunk)
            t2 = ssml_task(director_agent, chunk)
            t3 = validation_task(critic_agent)
            
            # Create crew with sequential process
            crew = Crew(
                agents=[analyst_agent, director_agent, critic_agent],
                tasks=[t1, t2, t3],
                verbose=True
            )
            
            # Execute the workflow
            result = crew.kickoff()
            
            # Capture the mood map from the analysis task
            chunk_mood_map = str(t1.output) if hasattr(t1, 'output') else ""
            all_mood_maps.append(f"\n=== CHUNK {i}/{len(text_chunks)} ===\n\n{chunk_mood_map}")
            
            # Extract SSML (remove outer <speak> tags for merging)
            chunk_ssml = str(result).strip()
            chunk_ssml = re.sub(r'^<speak[^>]*>\s*|\s*</speak>$', '', chunk_ssml, flags=re.DOTALL)
            all_ssml_parts.append(chunk_ssml)
            
            # Parse this chunk's SSML into segments
            chunk_segments = parse_ssml_to_segments(f'<speak>{chunk_ssml}</speak>', chunk)
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
        director_agent = voice_director(llm)
        critic_agent = ssml_critic(llm)
        
        # Create tasks
        t1 = analysis_task(analyst_agent, text)
        t2 = ssml_task(director_agent, text)
        t3 = validation_task(critic_agent)
        
        # Create crew with sequential process
        crew = Crew(
            agents=[analyst_agent, director_agent, critic_agent],
            tasks=[t1, t2, t3],
            verbose=True
        )
        
        # Execute the workflow
        print("  Starting CrewAI multi-agent workflow...")
        result = crew.kickoff()
        
        # Capture the mood map from the analysis task
        mood_map = str(t1.output) if hasattr(t1, 'output') else ""
        
        # Extract the final SSML
        raw_ssml = str(result).strip()
        
        # Ensure SSML has proper <speak> wrapper
        if not raw_ssml.startswith('<speak>'):
            raw_ssml = f'<speak>\n{raw_ssml}\n</speak>'
        
        # Parse SSML into segments
        segments = parse_ssml_to_segments(raw_ssml, text)
    
    return segments, raw_ssml, mood_map


def main():
    """
    Standalone test function for the CrewAI workflow.
    Run with: python src/sentiment_crewai.py
    """
    print("=" * 80)
    print("CrewAI Emotional SSML Generator - Standalone Test")
    print("=" * 80)
    print()
    
    print("Sample Text:")
    print("-" * 80)
    print(SAMPLE_TEXT.strip())
    print("-" * 80)
    print()
    
    # Process the sample text
    segments, raw_ssml, mood_map = process_chapter_with_crewai(
        text=SAMPLE_TEXT.strip(),
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
