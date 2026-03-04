import json
from pathlib import Path
from crewai import Agent, Task, Crew, Process

from agents.utils import local_llm
from config import Config
from script_schema import ChapterScript, ScriptSegment
from text_utils import split_text_smartly
from .base import ScriptWriter


class EmotionalAnalystWriter(ScriptWriter):
    """
    Uses the CrewAI emotional_analyst agent to read text chunks and produce
    the annotated ChapterScript JSON.
    """
    def __init__(self):
        self.llm = local_llm(model=Config.DEFAULT_OLLAMA_MODEL, base_url=Config.DEFAULT_OLLAMA_URL)

    def write(self, text: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        print(f"  [EmotionalAnalystWriter] Initializing LLM Analysis via CrewAI...")

        # For long texts, we should do initial extraction of characters globally
        # But this can be a basic implementation doing chunk-by-chunk for segments
        
        # Step 1: Extract summary (single task)
        summary = self._extract_summary(text)
        
        # Step 2: Split and interpret segments
        chunks = split_text_smartly(text, max_chunk_size=1500)
        segments = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"  [EmotionalAnalystWriter] Interpreting chunk {i}/{len(chunks)}...")
            chunk_segments = self._interpret_chunk(chunk)
            segments.extend(chunk_segments)

        chapter_title = output_path.stem.replace("_script", "").replace(".json", "")

        script = ChapterScript(
            version=1,
            chapter_title=chapter_title,
            summary=summary,
            segments=segments
        )
        
        script.save(output_path)
        print(f"  [EmotionalAnalystWriter] ✓ Saved script to {output_path}")
        return output_path

    def _extract_summary(self, full_text: str) -> str:
        agent = Agent(
            role="Narrative Context Analyzer",
            goal="Identify the overall summary of the chapter.",
            backstory="You are a dramaturg setting up the context for voice actors.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
        
        # Limit text for context extraction to first 3000 chars to save time
        sample_text = full_text[:3000]
        
        task = Task(
            description=f'''
            Based on the following text sample, write a one-sentence summary of what is happening.
            Return ONLY the text of the summary. No JSON, no quotes.
            
            Text sample:
            {sample_text}
            ''',
            agent=agent,
            expected_output="A single sentence summary."
        )
        
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        return str(result).strip()

    def _interpret_chunk(self, chunk: str) -> list[ScriptSegment]:
        from agents.emotional_analyst import emotional_analyst
        
        agent = emotional_analyst(self.llm)
        
        task = Task(
            description=f'''
            You are a Voice Director creating an annotated script for a TTS engine.
            Read the text and break it into segments (1-3 sentences).
            For each segment, provide the EXACT original text but add inline pauses like [pause 500ms] and emphasis like [emphasis]word[/emphasis].
            Then provide a human-readable "interpretation" note explaining how the actor should read it.
            
            DO NOT MISS ANY ORIGINAL TEXT. The text MUST be exactly the original text, just with inline brackets added.
            
            Return ONLY valid JSON matching this structure (a single array of objects):
            [
              {{
                "text": "The original text here [pause 300ms] with annotations [emphasis]added[/emphasis].",
                "interpretation": "A note for the Voice Actor on how to read this."
              }}
            ]
            
            Text to process:
            {chunk}
            ''',
            agent=agent,
            expected_output="Valid JSON array of segment objects."
        )
        
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        try:
            json_str = str(result)
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
                
            data = json.loads(json_str)
            return [ScriptSegment(**item) for item in data]
        except Exception as e:
            print(f"  [Warning] Failed to parse chunk JSON, falling back to raw: {e}")
            return [ScriptSegment(text=chunk, interpretation="Parsing failed; default interpretation.")]
