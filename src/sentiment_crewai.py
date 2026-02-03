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

from tts_styletts2 import EMOTION_PARAMS, EmotionSegment


# Sample dramatic text for testing
SAMPLE_TEXT = """
Now, something had been happening there a little before, which I did not 
know anything about until a good many days afterwards, but I will tell you 
about it now. Those two old brothers had been having a pretty hot argument 
a couple of days before, and had ended by agreeing to decide it by a bet, 
which is the English way of settling everything.
"""


def create_local_llm(model: str = "qwen2.5:14b", base_url: str = "http://localhost:11434") -> LLM:
    """
    Create a local Ollama LLM connection via CrewAI.
    
    Args:
        model: Ollama model name (e.g., "llama3.2:3b", "llama3", "mistral")
        base_url: Ollama API endpoint
    
    Returns:
        Configured LLM instance
    """
    # Set environment variable to bypass OpenAI API key requirement
    os.environ["OPENAI_API_KEY"] = "NA"
    
    return LLM(
        model=f"ollama/{model}",
        base_url=base_url,
        api_key="NA"
    )

def create_emotional_analyst(llm: LLM) -> Agent:
    return Agent(
        role="Narrative Emotion Annotator",
        goal="""
        Convert narrative text into a structured emotional timeline using
        Plutchik emotions and Valence–Arousal–Dominance (VAD) scores.
        The output must be suitable for expressive text-to-speech synthesis.
        """,
        backstory="""
        You are an expert in narrative psychology and affective computing.

        You MUST:
        - Classify emotions ONLY using Plutchik's 8 core emotions:
          joy, trust, fear, surprise, sadness, disgust, anger, anticipation.
        - Assign an intensity value between 0.0 and 1.0 for each emotion.
        - Assign VAD values:
            valence:   -1.0 (very negative) to 1.0 (very positive)
            arousal:    0.0 (calm) to 1.0 (excited)
            dominance:  0.0 (submissive) to 1.0 (dominant)

        Think like a voice director:
        - How should the line sound?
        - Where does emotion shift?
        - What should the listener feel?

        DO NOT write literary analysis.
        DO NOT explain your reasoning.
        Output ONLY structured data.
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_voice_director(llm: LLM) -> Agent:
    return Agent(
        role="Audiobook Voice Director",
        goal="""
        Convert structured emotional segments into SSML.
        You DO NOT infer emotion; you execute it literally.
        Handle multiple characters, transitions, and prosody.
        """,
        backstory="""
        You are a professional audiobook director.
        Responsibilities:
        - Map VAD → SSML prosody (rate, pitch, volume)
        - Respect emotional intensity and dominance
        - Use pauses according to transitions:
            soft: 200-300ms
            sharp: 400-700ms
            break: 800ms+
        - Emphasis:
            Only if intensity >= 0.7 and dominance >= 0.4
            Apply sparingly: 1–2 phrases per sentence
        - Handle multiple characters by switching voice tags
        - Include micro-pauses to simulate breathing
        - Output well-formed SSML for XTTS
        - No explanations or extra text
        """,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_ssml_critic(llm: LLM) -> Agent:
    """
    Agent 3: The SSML Critic
    
    Role: SSML Quality Auditor who validates markup structure and
    checks for robotic or unnatural pacing.
    """
    return Agent(
        role="SSML XML Validator",
        goal="Validate SSML markup syntax and ensure proper XML structure",
        backstory="""You are a technical quality control specialist for SSML/XML markup.
        Your expertise is in validating XML syntax, not story content or creative writing.
        
        You check SSML documents for:
        - Proper tag closure and nesting
        - Valid attribute values
        - Appropriate use of prosody and breaks
        
        You NEVER analyze story content, provide writing tips, or suggest narrative changes.
        You ONLY validate XML markup structure.
        
        You output ONLY valid SSML - no explanations, no commentary, no analysis.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )


def create_analysis_task(agent: Agent, text: str) -> Task:
    """
    Task 1: Analyze text and produce a Mood Map
    
    The Emotional Analyst creates a detailed mood map with:
    - Emotional tone for each segment
    - Intensity level (1-10)
    - Pacing notes (fast, medium, slow)
    - Key phrases that need emphasis
    """
    return Task(
        description=f"""Analyze the following text for emotional content and create a Mood Map.
        
Text to analyze:
{text}

Available emotions: excited, sad, calm, tense, humorous, neutral, dramatic, hopeful, desperate

Create a detailed Mood Map with the following structure for each segment (1-3 sentences):
1. Text segment (exact quotes from the original)
2. Dominant emotion
3. Intensity (1-10 scale)
4. Suggested pacing (fast/medium/slow)
5. Key phrases that need emphasis
6. Any dramatic pauses needed

Format your response as a numbered list of segments with clear labels for each field.""",
        agent=agent,
        expected_output="A detailed Mood Map breaking down the text into emotional segments with intensity, pacing, and emphasis notes"
    )


def create_ssml_task(agent: Agent, text: str) -> Task:
    """
    Task 2: Convert text + Mood Map into SSML
    
    The Voice Director uses the Mood Map to create SSML with:
    - <prosody> tags for rate and pitch control
    - <break> tags for dramatic pauses
    - <emphasis> tags for key phrases
    """
    return Task(
        description=f"""Using the Mood Map from the Emotional Analyst, add SSML markup to this text.

*** CRITICAL: TEXT PRESERVATION RULES - READ CAREFULLY ***

You MUST include EVERY SINGLE WORD from the original text below, starting from the VERY FIRST WORD.
The text may start mid-sentence - this is INTENTIONAL. Do NOT skip it.

FORBIDDEN ACTIONS (WILL CAUSE FAILURE):
- Skipping ANY words, even if they seem incomplete
- Changing, correcting, or "fixing" any words
- Paraphrasing or summarizing
- Removing sentences or phrases for any reason
- Adding new content not in the original
- Reordering sentences

The output MUST start with the same words as the original text starts with.
Every line of the original MUST appear in your SSML output.

=== ORIGINAL TEXT (include EVERY word, starting from "{text.split()[0] if text.split() else ''}") ===
{text}
=== END OF ORIGINAL TEXT ===

SSML Tags to use:
- <prosody rate="x-slow|slow|medium|fast|x-fast" pitch="x-low|low|medium|high|x-high">text</prosody>
- <break time="200ms|500ms|1s|2s"/>
- <emphasis level="reduced|moderate|strong">text</emphasis>

Guidelines:
1. Use <break/> tags for dramatic pauses between thoughts or at emotional beats
2. Adjust rate based on pacing (slow=contemplative, fast=urgent)
3. Adjust pitch based on emotion (high=excited, low=somber)
4. Use <emphasis> SPARINGLY - only for 1-2 KEY phrases per sentence maximum
5. Most text should have NO emphasis tags - use prosody for general emotion instead
6. Wrap everything in a <speak> root element

Return ONLY the complete SSML document with ALL original text preserved.""",
        agent=agent,
        expected_output="A complete, well-formatted SSML document with EVERY word from the original text preserved verbatim, enhanced with prosody, breaks, and emphasis tags"
    )


def create_validation_task(agent: Agent) -> Task:
    """
    Task 3: Validate and critique the SSML
    
    The SSML Critic checks for:
    - Proper XML structure and tag nesting
    - Robotic patterns (too uniform, missing variation)
    - Missing emotional beats
    - Technical errors in attribute values
    """
    return Task(
        description="""CRITICAL: You are an SSML MARKUP validator with TWO responsibilities:

1. VALIDATE TEXT PRESERVATION (MOST IMPORTANT)
   - The SSML output must contain ALL words from the original text
   - No words should be changed, removed, or paraphrased
   - If text has been modified, this is a CRITICAL ERROR
   - If text is missing or altered, you MUST reject and note the issue

2. VALIDATE XML SYNTAX
   - All <prosody>, <break>, <emphasis> tags properly closed
   - Root <speak> element present
   - Valid attributes: rate (x-slow|slow|medium|fast|x-fast), pitch (x-low|low|medium|high|x-high), time (e.g., 500ms, 1s)
   - Emphasis used SPARINGLY (1-2 phrases per sentence max, not individual words)
   - If you see multiple <emphasis> tags on consecutive words, REMOVE them - this is over-tagging

DO NOT:
- Analyze story content
- Provide writing advice
- Suggest narrative improvements
- Rewrite or modify ANY text content

Output Format:
- If SSML is valid AND text is preserved: output it EXACTLY as received
- If SSML has XML errors: fix ONLY the XML tags, preserve all text
- If text was modified/removed: output the SSML but note that text integrity was compromised
- NEVER add explanations, analysis, or commentary
- NEVER include anything except the <speak>...</speak> XML document

Return ONLY valid SSML markup. Nothing else.""",
        agent=agent,
        expected_output="Valid SSML document with original text preserved and no explanations or commentary"
    )


def parse_ssml_to_segments(ssml: str, text: str) -> List[EmotionSegment]:
    """
    Parse the SSML output and convert it into EmotionSegment objects
    compatible with the existing TTS pipeline.
    
    Args:
        ssml: The SSML markup from the Voice Director
        text: Original text
    
    Returns:
        List of EmotionSegment objects with inline SSML
    """
    segments = []
    
    # Remove the outer <speak> tags
    content = re.sub(r'^<speak>\s*|\s*</speak>$', '', ssml.strip(), flags=re.DOTALL)
    
    # Split by break tags to find natural segments
    parts = re.split(r'<break\s+time="[^"]+"\s*/>', content)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Infer emotion from prosody attributes
        emotion = "neutral"
        
        # Check for pitch/rate patterns that suggest emotions
        if 'pitch="high"' in part or 'pitch="x-high"' in part:
            if 'rate="fast"' in part or 'rate="x-fast"' in part:
                emotion = "excited"
            else:
                emotion = "hopeful"
        elif 'pitch="low"' in part or 'pitch="x-low"' in part:
            if 'rate="slow"' in part or 'rate="x-slow"' in part:
                emotion = "sad"
            else:
                emotion = "tense"
        elif '<emphasis level="strong">' in part:
            emotion = "dramatic"
        elif 'rate="slow"' in part:
            emotion = "calm"
        
        # Extract clean text for length estimation (rough)
        clean_text = re.sub(r'<[^>]+>', '', part).strip()
        
        if clean_text:
            # Get emotion parameters
            params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])
            
            segments.append(EmotionSegment(
                text=part,  # Keep SSML tags inline
                emotion=emotion,
                alpha=params["alpha"],
                beta=params["beta"],
                diffusion_steps=params["diffusion_steps"],
                prosody=params["prosody"]
            ))
    
    # If no segments were created, create one neutral segment
    if not segments:
        params = EMOTION_PARAMS["neutral"]
        segments.append(EmotionSegment(
            text=text,
            emotion="neutral",
            alpha=params["alpha"],
            beta=params["beta"],
            diffusion_steps=params["diffusion_steps"],
            prosody=params["prosody"]
        ))
    
    return segments


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
    llm = create_local_llm(model=model, base_url=ollama_url)
    
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
            analyst = create_emotional_analyst(llm)
            director = create_voice_director(llm)
            critic = create_ssml_critic(llm)
            
            # Create tasks for this chunk
            analysis_task = create_analysis_task(analyst, chunk)
            ssml_task = create_ssml_task(director, chunk)
            validation_task = create_validation_task(critic)
            
            # Create crew with sequential process
            crew = Crew(
                agents=[analyst, director, critic],
                tasks=[analysis_task, ssml_task, validation_task],
                verbose=True
            )
            
            # Execute the workflow
            result = crew.kickoff()
            
            # Capture the mood map from the analysis task
            chunk_mood_map = str(analysis_task.output) if hasattr(analysis_task, 'output') else ""
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
        analyst = create_emotional_analyst(llm)
        director = create_voice_director(llm)
        critic = create_ssml_critic(llm)
        
        # Create tasks
        analysis_task = create_analysis_task(analyst, text)
        ssml_task = create_ssml_task(director, text)
        validation_task = create_validation_task(critic)
        
        # Create crew with sequential process
        crew = Crew(
            agents=[analyst, director, critic],
            tasks=[analysis_task, ssml_task, validation_task],
            verbose=True
        )
        
        # Execute the workflow
        print("  Starting CrewAI multi-agent workflow...")
        result = crew.kickoff()
        
        # Capture the mood map from the analysis task
        mood_map = str(analysis_task.output) if hasattr(analysis_task, 'output') else ""
        
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
