from pathlib import Path
from typing import List

from script_schema import ChapterScript, ScriptSegment
from .base import ScriptWriter


class RuleBasedWriter(ScriptWriter):
    """
    A fast, deterministic script writer that uses sentence boundaries
    and simple heuristics instead of an LLM.
    """

    def write(self, segments: List[str], output_path: Path) -> Path:
        output_path = Path(output_path)
        print(f"  [RuleBasedWriter] Processing {len(segments)} segments...")
        
        chunks = segments
        
        script_segments = []
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
            script_segments.append(segment)

        summary = "Summary not extracted (Rule-Based Writer)"

        chapter_title = output_path.stem.replace("_script", "").replace(".json", "")

        script = ChapterScript(
            version=1,
            chapter_title=chapter_title,
            summary=summary,
            segments=script_segments
        )
        
        script.save(output_path)
        print(f"  [RuleBasedWriter] ✓ Saved {len(script_segments)} segments to {output_path}")
        return output_path
