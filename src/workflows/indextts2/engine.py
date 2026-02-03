import json
import requests
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

COMFY_URL = "http://localhost:8188"

def generate_audio_with_indextts2(
    text: str,
    output_path: Path,
    soft_instruction: str = "",
    reference_audio_path: Optional[str] = None,
    mode: str = "Auto", # Auto, Duration, Tokens
    model_choice: str = "1.7B", # For IndexTTS-2
):
    """
    Sends text and soft instructions to ComfyUI for IndexTTS-2 generation.
    Uses the IndexTTS2EmotionTextNode.
    """
    
    # Define the ComfyUI Workflow
    # Note: Using IndexTTS2EmotionTextNode for soft instructions
    workflow = {
        "1": {
            "inputs": {
                "text": text,
                "mode": mode,
                "emotion_description": soft_instruction,
                "do_sample_mode": "on",
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 30,
                "seed": 0,
                "reference_audio": ["2", 0] if reference_audio_path else None
            },
            "class_type": "IndexTTS2EmotionTextNode"
        },
        "3": {
            "inputs": {
                "filename_prefix": "indextts2_out",
                "images": ["1", 0] # Standard audio output link in some Comfy setups
            },
            "class_type": "SaveAudio"
        }
    }

    # If reference audio is provided, we need a loader node
    if reference_audio_path and os.path.exists(reference_audio_path):
        workflow["2"] = {
            "inputs": {
                "filename": os.path.basename(reference_audio_path)
            },
            "class_type": "LoadSpeakerNode" # Or TimbreAudioLoader depending on setup
        }
    else:
        # Fallback or error? For IndexTTS2, reference is usually required
        # If no reference, node 1's reference_audio will be None
        pass

    # 2. Trigger Prompt
    print(f"  Sending to IndexTTS-2 (ComfyUI at {COMFY_URL})...")
    print(f"  Instruction: {soft_instruction}")
    
    try:
        response = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow})
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
    except Exception as e:
        print(f"  Error connecting to ComfyUI: {e}")
        raise

    # 3. Poll for result
    print(f"  Waiting for IndexTTS-2 generation (ID: {prompt_id})...")
    start_time = time.time()
    while True:
        try:
            history_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}")
            history = history_resp.json()
            if prompt_id in history:
                break
        except Exception:
            pass
        
        if time.time() - start_time > 120: # 2 minute timeout
            raise TimeoutError("IndexTTS-2 generation timed out")
            
        time.sleep(1)

    # 4. Download result
    outputs = history[prompt_id]["outputs"]
    # Find the node ID for SaveAudio (it was "3" in our workflow)
    save_node_id = "3"
    if save_node_id not in outputs:
        # Try to find any node that has audio output
        for node_id, node_output in outputs.items():
            if "audio" in node_output:
                save_node_id = node_id
                break
    
    if save_node_id not in outputs:
        raise Exception(f"Could not find audio output in ComfyUI history for prompt {prompt_id}")

    output = outputs[save_node_id].get("audio", [{}])[0]
    filename = output.get("filename")
    
    if not filename:
         # Some implementations might use result[0] for audio
         raise Exception("Could not find generated audio filename in ComfyUI response")

    audio_url = f"{COMFY_URL}/view?filename={filename}&type=output"
    audio_data = requests.get(audio_url).content
    
    with open(output_path, "wb") as f:
        f.write(audio_data)

    print(f"  Audio saved to {output_path}")

def combine_audio_segments(segments: List[Path], output_path: Path):
    """
    Utility to combine multiple wav files (if needed, though DialogueInference often merges).
    For IndexTTS2, we currently generate segment by segment.
    """
    import soundfile as sf
    import numpy as np
    
    all_data = []
    sample_rate = None
    
    for p in segments:
        data, sr = sf.read(str(p))
        if sample_rate is None:
            sample_rate = sr
        all_data.append(data)
        
    if all_data:
        combined = np.concatenate(all_data)
        sf.write(str(output_path), combined, sample_rate)
