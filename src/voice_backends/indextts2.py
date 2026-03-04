import json
import requests
from pathlib import Path
from pydub import AudioSegment
from crewai import Agent, Task, Crew

from agents.utils import local_llm
from config import Config
from script_schema import ChapterScript, ScriptSegment
from .base import VoiceBackend, SynthesisContext


class IndexTTS2Synth(VoiceBackend):
    """
    Translates annotated script segments into IndexTTS-2 emo_vectors
    and generates audio by calling the IndexTTS-2 server.
    """

    def synthesize(self, script_path: Path, output_path: Path, ctx: SynthesisContext) -> Path:
        print(f"  [IndexTTS2Synth] Loading script from {script_path}...")
        script = ChapterScript.load(script_path)
        
        # We need a temp directory for individual segment WAVs
        temp_dir = output_path.parent / f"temp_{output_path.stem}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Translate script segments to IndexTTS-2 payloads
        # Use a checkpoint file to avoid re-running the LLM on resume
        payloads_path = script_path.with_suffix(".payloads.json")
        if payloads_path.exists():
            print(f"  [IndexTTS2Synth] Loading cached payloads from {payloads_path}...")
            with open(payloads_path, "r") as f:
                payloads = json.load(f)
        else:
            payloads = self._translate_to_payloads(script)
            with open(payloads_path, "w") as f:
                json.dump(payloads, f, indent=2)
            print(f"  [IndexTTS2Synth] Saved {len(payloads)} payloads to {payloads_path}")
        
        temp_files = []
        
        # Step 2: Generate audio for each translated segment
        for idx, payload in enumerate(payloads):
            temp_path = temp_dir / f"seg_{idx:03d}.wav"
            
            if temp_path.exists():
                print(f"  [IndexTTS2Synth] [{idx+1}/{len(payloads)}] Skipping already generated segment.")
                temp_files.append((temp_path, payload["interval_silence"]))
                continue
                
            print(f"  [IndexTTS2Synth] [{idx+1}/{len(payloads)}] Generating: {payload['text'][:50]}...")
            success = self._generate_audio(
                text=payload["text"],
                emo_vector=payload["emo_vector"],
                interval_silence=payload["interval_silence"],
                voice_name=ctx.voice_name or Config.DEFAULT_INDEXTTS2_VOICE,
                tts_url=ctx.tts_url or Config.DEFAULT_INDEXTTS2_URL,
                output_path=temp_path
            )
            
            if success and temp_path.exists():
                temp_files.append((temp_path, payload["interval_silence"]))
            else:
                print(f"  [IndexTTS2Synth] ⚠ Failed to generate audio for segment {idx+1}")
                
        # Step 3: Merge segments
        if temp_files:
            print(f"  [IndexTTS2Synth] Merging {len(temp_files)} segments into {output_path}...")
            combined = AudioSegment.empty()
            
            for i, (path, silence_ms) in enumerate(temp_files):
                try:
                    seg_audio = AudioSegment.from_wav(str(path))
                    combined += seg_audio
                    if i < len(temp_files) - 1:
                        combined += AudioSegment.silent(duration=silence_ms)
                except Exception as e:
                    print(f"  [IndexTTS2Synth] ⚠ Error decoding segment from {path}: {e}")
                    
            combined.export(str(output_path), format="wav")
            print(f"  [IndexTTS2Synth] ✓ Audio successfully combined and saved to {output_path}")
            return output_path
        else:
            raise RuntimeError("No audio segments were successfully generated.")


    def _translate_to_payloads(self, script: ChapterScript) -> list[dict]:
        """Use an LLM to translate script annotations into IndexTTS-2 inputs."""
        print(f"  [IndexTTS2Synth] Translating {len(script.segments)} segments to IndexTTS2 inputs using LLM...")
        
        llm = local_llm(model=Config.DEFAULT_OLLAMA_MODEL, base_url=Config.DEFAULT_OLLAMA_URL)
        
        agent = Agent(
            role="IndexTTS-2 Configuration Specialist",
            goal="Translate human annotations into exact JSON payloads for the IndexTTS-2 server.",
            backstory="You are an expert audio engineer who translates emotional intent into numerical parameters.",
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        
        payloads = []
        
        # Process in batches to avoid overwhelming LLM context
        batch_size = 5
        for i in range(0, len(script.segments), batch_size):
            batch = script.segments[i:i+batch_size]
            
            # Prepare batch input
            batch_input = []
            for j, seg in enumerate(batch):
                batch_input.append(f"Segment {i+j+1}:\nText: {seg.text}\nInterpretation: {seg.interpretation}\n")
            batch_text = "\n".join(batch_input)
            
            task = Task(
                description=f'''
                Convert the following annotated text segments into IndexTTS-2 configuration.
                
                IndexTTS-2 uses an 8-value emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
                Vector values must be floats between 0 and 1.
                
                For each segment:
                1. Strip ALL inline annotations from the text (like [pause 500ms] or [emphasis]).
                2. Extract any pause durations (in ms) to use as 'interval_silence' (default 200).
                3. Convert the interpretation into the 8-value 'emo_vector'.
                
                Return exactly a valid JSON array of objects with this schema:
                [
                  {{
                    "text": "Clean text without brackets here.",
                    "emo_vector": [0, 0, 0, 0, 0, 0, 0, 0.5],
                    "interval_silence": 200
                  }}
                ]
                
                Segments to process:
                {batch_text}
                ''',
                agent=agent,
                expected_output="Valid JSON array of IndexTTS-2 payload objects."
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
                print(f"  [IndexTTS2Synth] ⚠ Failed to parse LLM translation, falling back to neutral: {e}")
                # Fallback neutral
                for seg in batch:
                    # quick strip of simple brackets
                    import re
                    clean_text = re.sub(r'\[.*?\]', '', seg.text).strip()
                    payloads.append({
                        "text": clean_text,
                        "emo_vector": [0, 0, 0, 0, 0, 0, 0, 0.5],
                        "interval_silence": 200
                    })

        return payloads


    def _generate_audio(
        self,
        text: str,
        emo_vector: list[float],
        interval_silence: int,
        voice_name: str,
        tts_url: str,
        output_path: Path
    ) -> bool:
        """Call the IndexTTS-2 server API."""
        url = f"{tts_url.rstrip('/')}/generate"
        payload = {
            "text": text,
            "filename": output_path.name,
            "voice": voice_name,
            "use_emo_text": False,
            "emo_alpha": 1.0,
            "interval_silence": interval_silence,
            "emo_vector": emo_vector,
        }

        try:
            response = requests.post(url, json=payload, timeout=3600)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        except requests.exceptions.RequestException as e:
            print(f"  [IndexTTS2Synth] Request failed: {e}")
            return False
