import json
import requests
import time
import os
from pathlib import Path

COMFY_URL = "http://localhost:8188"

def generate_audio_with_qwen(
    script: str,
    output_path: Path,
    voice_name: str = "Serena", # Default preset
    speaker_audio_path: str = None, # If cloning
):
    """
    Sends a script to ComfyUI for Qwen-TTS generation.
    """
    # 1. Define the ComfyUI Workflow
    # Note: IDs are arbitrary for our manual JSON construction
    workflow = {
        "1": {
            "inputs": {
                "script": script,
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
                "images": ["1", 0] # DialogueInferenceNode outputs audio, but Comfy calls it 'images' in standard Save nodes sometimes, but let's assume a dedicated audio save node if available or the Dialogue node handles it.
            },
            "class_type": "SaveAudio" # Custom nodes often have specific audio savers
        }
    }

    # If cloning is requested, override node 3
    if speaker_audio_path and os.path.exists(speaker_audio_path):
        workflow["3"] = {
            "inputs": {
                "ref_audio": speaker_audio_path,
                "ref_text": "", # Optional
                "model_choice": "1.7B",
                "unload_model_after_generate": False
            },
            "class_type": "VoiceClonePromptNode"
        }

    # 2. Trigger Prompt
    print(f"  Sending script to Qwen-TTS (ComfyUI at {COMFY_URL})...")
    response = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]

    # 3. Poll for result
    print(f"  Waiting for audio generation (ID: {prompt_id})...")
    while True:
        history_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}")
        history = history_resp.json()
        if prompt_id in history:
            break
        time.sleep(1)

    # 4. Download result
    # We assume 'SaveAudio' saved it to ComfyUI's output folder
    # We might need to find the filename from history
    output = history[prompt_id]["outputs"].get("4", {}).get("audio", [{}])[0]
    filename = output.get("filename")
    if not filename:
         raise Exception("Could not find generated audio filename in ComfyUI history")

    audio_url = f"{COMFY_URL}/view?filename={filename}&type=output"
    audio_data = requests.get(audio_url).content
    
    with open(output_path, "wb") as f:
        f.write(audio_data)

    print(f"  Audio saved to {output_path}")
