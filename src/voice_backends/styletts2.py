import json
from pathlib import Path

import numpy as np
import soundfile as sf
from crewai import Agent, Task, Crew

from agents.utils import local_llm
from config import Config
from script_schema import ChapterScript, ScriptSegment
from .base import VoiceBackend, SynthesisContext

# Lazy load to avoid importing heavy PyTorch/StyleTTS2 deps on CLI start
_tts_model = None

def _get_model():
    global _tts_model
    if _tts_model is None:
        try:
            from styletts2 import tts as styletts2_tts
            _tts_model = styletts2_tts.StyleTTS2()
        except ImportError:
            raise ImportError("StyleTTS2 not installed. Install with: pip install styletts2")
    return _tts_model


class StyleTTS2Synth(VoiceBackend):
    """
    Translates annotated script segments into exact StyleTTS2 inference parameters
    (alpha, beta, diffusion_steps) and synthesizes audio locally.
    """

    def synthesize(self, script_path: Path, output_path: Path, ctx: SynthesisContext) -> Path:
        print(f"  [StyleTTS2Synth] Loading script from {script_path}...")
        script = ChapterScript.load(script_path)
        
        voice_sample = ctx.voice_sample_path or Path(Config.DEFAULT_STYLE_VOICE_SAMPLE)
        if not voice_sample.exists():
            raise FileNotFoundError(f"Voice sample for StyleTTS2 not found: {voice_sample}")

        model = _get_model()
        
        # Step 1: Translate script segments to StyleTTS2 params
        payloads = self._translate_to_payloads(script)
        
        all_audio = []
        sample_rate = 24000
        
        # Step 2: Generate audio for each translated segment
        for idx, payload in enumerate(payloads):
            print(f"  [StyleTTS2Synth] [{idx+1}/{len(payloads)}] Generating: {payload['text'][:50]}...")
            
            # Create pause silence if requested
            if payload["interval_silence"] > 0:
                silence_frames = int((payload["interval_silence"] / 1000.0) * sample_rate)
                silence = np.zeros(silence_frames, dtype=np.float32)
                all_audio.append(silence)
                
            try:
                wav = model.inference(
                    text=payload["text"],
                    target_voice_path=str(voice_sample),
                    output_sample_rate=sample_rate,
                    alpha=payload["alpha"],
                    beta=payload["beta"],
                    diffusion_steps=payload["diffusion_steps"],
                    embedding_scale=1.0,
                )
                all_audio.append(wav)
            except Exception as e:
                print(f"  [StyleTTS2Synth] ⚠ Failed to generate segment {idx+1}: {e}")

        if not all_audio:
            raise RuntimeError("No audio segments were successfully generated.")

        combined = np.concatenate(all_audio)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), combined, sample_rate)
        
        print(f"  [StyleTTS2Synth] ✓ Audio successfully combined and saved to {output_path}")
        return output_path

    def _translate_to_payloads(self, script: ChapterScript) -> list[dict]:
        """Use an LLM to translate script annotations into StyleTTS2 params."""
        print(f"  [StyleTTS2Synth] Translating {len(script.segments)} segments to StyleTTS2 parameters using LLM...")
        
        llm = local_llm(model=Config.DEFAULT_OLLAMA_MODEL, base_url=Config.DEFAULT_OLLAMA_URL)
        
        agent = Agent(
            role="StyleTTS2 Parameter Specialist",
            goal="Translate human annotations into exact JSON payloads for StyleTTS2.",
            backstory=(
                "You are an expert audio engineer specializing in StyleTTS2. You map emotions to 3 parameters:\n"
                "- Alpha (timbre/speaker identity): 0.1 (strong target) to 0.8 (styled).\n"
                "- Beta (prosody strength): 0.4 (flat) to 0.9 (strong emotion).\n"
                "- Diffusion Steps: 5 (neutral/standard) to 15 (complex high-quality emotion)."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        
        payloads = []
        batch_size = 5
        for i in range(0, len(script.segments), batch_size):
            batch = script.segments[i:i+batch_size]
            
            batch_input = []
            for j, seg in enumerate(batch):
                batch_input.append(f"Segment {i+j+1}:\nText: {seg.text}\nInterpretation: {seg.interpretation}\n")
            batch_text = "\n".join(batch_input)
            
            task = Task(
                description=f'''
                Convert the annotated text segments into StyleTTS2 configuration.
                
                For each segment:
                1. Strip ALL inline annotations from the text (like [pause 500ms] or [emphasis]).
                2. Extract any pause durations (in ms) before the segment and set 'interval_silence' (default 0).
                3. Based on the interpretation, determine the optimal 'alpha' (0.1 - 0.8), 'beta' (0.4 - 0.9), and 'diffusion_steps' (5 - 15).
                
                Return exactly a valid JSON array of objects with this schema:
                [
                  {{
                    "text": "Clean text without brackets here.",
                    "alpha": 0.3,
                    "beta": 0.7,
                    "diffusion_steps": 5,
                    "interval_silence": 0
                  }}
                ]
                
                Segments to process:
                {batch_text}
                ''',
                agent=agent,
                expected_output="Valid JSON array of StyleTTS2 payload objects."
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
                payloads.extend(data)
            except Exception as e:
                print(f"  [StyleTTS2Synth] ⚠ Failed to parse LLM translation, falling back to neutral: {e}")
                for seg in batch:
                    import re
                    clean_text = re.sub(r'\[.*?\]', '', seg.text).strip()
                    payloads.append({
                        "text": clean_text,
                        "alpha": 0.3,
                        "beta": 0.7,
                        "diffusion_steps": 5,
                        "interval_silence": 0
                    })

        return payloads
