import os
from pathlib import Path

from script_schema import ChapterScript, ScriptSegment
from text_utils import split_text_smartly
from .base import ScriptWriter


class RuleBasedWriter(ScriptWriter):
    """
    A fast, deterministic script writer that uses sentence boundaries
    and simple heuristics instead of an LLM.
    """

    def write(self, text: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        print(f"  [RuleBasedWriter] Splitting text into segments...")
        
        # We reuse the existing split logic which chunks nicely by punctuation
        chunks = split_text_smartly(text, max_chunk_size=500)
        
        segments = []
        for chunk in chunks:
            # Very basic heuristic for pacing: long chunks = fast, short = slow
            if len(chunk) > 300:
                pacing = "fast pacing, keep the momentum"
            elif len(chunk) < 50:
                pacing = "slow pacing, deliberate delivery"
            else:
                pacing = "medium pacing, natural storytelling"
                
            # Default interpretation
            interpretation = f"Standard narrative delivery. {pacing}."
            
            # Identify quotes
            if '"' in chunk or "'" in chunk:
                interpretation = f"Contains dialogue. {pacing}. Distinguish voices if necessary."

            segment = ScriptSegment(
                text=chunk,
                interpretation=interpretation
            )
            segments.append(segment)

        summary = "Summary not extracted (Rule-Based Writer)"

        chapter_title = output_path.stem.replace("_script", "").replace(".json", "")

        script = ChapterScript(
            version=1,
            chapter_title=chapter_title,
            summary=summary,
            segments=segments
        )
        
        script.save(output_path)
        print(f"  [RuleBasedWriter] ✓ Saved {len(segments)} segments to {output_path}")
        return output_path
