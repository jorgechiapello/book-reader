import requests
from pathlib import Path

def generate_audio_with_indextts2(
    text: str,
    output_path: Path,
    voice: str | None = None,
):
    """
    Calls the IndexTTS-2 server to generate audio.

    Args:
        text: The text to convert to speech.
        output_path: Local path where the audio file should be saved.
        voice: Voice name (e.g. "Heisenberg", "joe"). Must match a .wav in the
               server's voices directory. If omitted, server uses "Heisenberg".

    Returns:
        bool: True if successful, False otherwise.
    """
    url = "http://localhost:8001/generate"

    params = {"text": text, "filename": output_path.name}
    if voice is not None and voice.strip():
        params["voice"] = voice.strip()

    try:
        response = requests.post(
            url,
            params=params,
            timeout=120  # Increased timeout for longer text
        )
        response.raise_for_status()
        
        # Save the audio response to the output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        print(f"  ✓ Generated audio: {output_path.name}")
        return True
        
    except requests.exceptions.Timeout:
        print(f"  [TTS Error] Request timed out for text: {text[:50]}...")
        return False
    except requests.exceptions.RequestException as e:
        print(f"  [TTS Error] Failed to connect to TTS server: {e}")
        return False
    except IOError as e:
        print(f"  [TTS Error] Failed to save audio file: {e}")
        return False