import json
import os
import time
import requests
from pathlib import Path
from crewai import Agent, Task, Crew

from agents.utils import local_llm
from config import Config
from script_schema import ChapterScript
from .base import VoiceBackend, SynthesisContext


class QwenSynth(VoiceBackend):
    """
    Translates annotated script segments into a Qwen-TTS script format
    and sends the generation job to a ComfyUI workflow.
    """

    def synthesize(self, script_path: Path, output_path: Path, ctx: SynthesisContext) -> Path:
        print(f"  [QwenSynth] Loading script from {script_path}...")
        script = ChapterScript.load(script_path)
        
        # Step 1: Translate JSON script into a single plain-text Qwen conversational script
        qwen_script_text = self._translate_to_qwen_script(script)
        
        # Save a debug copy of the Qwen script
        debug_script_path = output_path.parent / f"{output_path.stem}.qwen_script"
        debug_script_path.write_text(qwen_script_text, encoding="utf-8")
        print(f"  [QwenSynth] Saved debug Qwen script to {debug_script_path}")
        
        comfy_url = ctx.tts_url or Config.DEFAULT_QWEN_COMFY_URL
        voice_name = ctx.voice_name or Config.DEFAULT_QWEN_VOICE
        
        # Step 2: Trigger ComfyUI workflow
        print(f"  [QwenSynth] Calling ComfyUI via {comfy_url} for voice: {voice_name}...")
        
        workflow = self._build_comfy_workflow(qwen_script_text, voice_name, ctx.voice_sample_path)
        
        try:
            response = requests.post(f"{comfy_url}/prompt", json={"prompt": workflow}, timeout=10)
            response.raise_for_status()
            prompt_id = response.json()["prompt_id"]
            
            print(f"  [QwenSynth] Workflow queued. Prompt ID: {prompt_id}. Waiting for completion...")
            
            while True:
                history_resp = requests.get(f"{comfy_url}/history/{prompt_id}", timeout=10)
                history = history_resp.json()
                if prompt_id in history:
                    break
                time.sleep(2)
                
            output = history[prompt_id]["outputs"].get("4", {}).get("audio", [{}])[0]
            filename = output.get("filename")
            if not filename:
                 raise Exception("Could not find generated audio filename in ComfyUI history")

            audio_url = f"{comfy_url}/view?filename={filename}&type=output"
            audio_data = requests.get(audio_url).content
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio_data)

            print(f"  [QwenSynth] ✓ Audio downloaded and saved to {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f"  [QwenSynth] ⚠ Failed to communicate with ComfyUI: {e}")
            raise RuntimeError(f"ComfyUI request failed: {e}")

    def _translate_to_qwen_script(self, script: ChapterScript) -> str:
        """Use an LLM to translate script annotations into a fluid Qwen-TTS script text block."""
        print(f"  [QwenSynth] Translating {len(script.segments)} segments to Qwen-TTS script using LLM...")
        
        llm = local_llm(model=Config.DEFAULT_OLLAMA_MODEL, base_url=Config.DEFAULT_OLLAMA_URL)
        
        agent = Agent(
            role="Qwen-TTS Script Adapter",
            goal="Translate an annotated chapter script into a fluid conversational text format suitable for Qwen-TTS.",
            backstory=(
                "You are an expert audio scriptwriter adapting annotated text into a raw text prompt for Qwen-TTS."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        
        # We can pass the whole script if it's not huge, otherwise batching.
        # But Qwen script is fundamentally a single solid block of dialogue/directions.
        script_json_dump = script.to_json()
        
        task = Task(
            description=f'''
            Convert the following JSON ChapterScript into a plain-text Qwen-TTS script.
            
            Rules:
            1. Combine the segment text into a cohesive flow.
            2. Remove our custom annotations like [pause 500ms].
            3. Use the interpretation notes ONLY as a guide to adjust punctuation (e.g., adding ellipses ... for dramatic pauses, or ALL CAPS for emphasis).
            4. Do not include JSON formatting in the output. The output MUST be just the plain text script that Qwen will read.
            
            ChapterScript:
            {script_json_dump[:4000]}... (truncated if too long)
            ''',
            agent=agent,
            expected_output="Plain text story script."
        )
        
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        return str(result).strip()

    def _build_comfy_workflow(self, qwen_script: str, voice_name: str, voice_sample_path: Path | None) -> dict:
        """Constructs the ComfyUI Qwen-TTS workflow dictionary."""
        workflow = {
            "1": {
                "inputs": {
                    "script": qwen_script,
                    "model_choice": "1.7B",
                    "attention": "auto",
                    "pause_seconds": 0.5,
                    "merge_outputs": True,
                    "batch_size": 4,
                    "unload_model_after_generate": False,
                    "role_bank": ["2", 0]
                },
                "class_type": "DialogueInferenceNode"
            },
            "2": {
                "inputs": {
                    "role_name_1": "Narrator",
                    "prompt_1": ["3", 0]
                },
                "class_type": "RoleBankNode"
            },
            "3": {
                "inputs": {
                    "speaker": voice_name
                },
                "class_type": "LoadSpeakerNode"
            },
            "4": {
                "inputs": {
                    "filename_prefix": "qwen_tts_out",
                    "images": ["1", 0] 
                },
                "class_type": "SaveAudio"
            }
        }

        # Override with clone node if a specific WAV sample is passed
        if voice_sample_path and voice_sample_path.exists():
            workflow["3"] = {
                "inputs": {
                    "ref_audio": str(voice_sample_path.absolute()),
                    "ref_text": "", 
                    "model_choice": "1.7B",
                    "unload_model_after_generate": False
                },
                "class_type": "VoiceClonePromptNode"
            }
            
        return workflow
