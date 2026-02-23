import requests
from pathlib import Path


def generate_audio_with_indextts2(
    text: str,
    output_path: Path,
    voice: str | None = None,
    filename: str | None = None,
    use_emo_text: bool = False,
    emo_alpha: float = 1,
    interval_silence: int = 200,
    emo_vector: list[float] | None = None,
    tts_url: str = "http://localhost:8001",
) -> bool:
    """
    Calls the IndexTTS-2 server to generate audio.

    Args:
        text: The text to convert to speech.
        output_path: Local path where the audio file should be saved.
        voice: Voice name (e.g. "Heisenberg"). Must match a .wav in the server's voices directory.
        filename: Output filename for the response. Defaults to output_path.name.
        use_emo_text: Use text-based emotion inference when emo_vector is null.
        emo_alpha: Emotion blend weight (default 1).
        interval_silence: Milliseconds of silence between segments.
        emo_vector: Emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
        tts_url: Base URL of the TTS server.

    Returns:
        True if successful, False otherwise.
    """
    url = f"{tts_url.rstrip('/')}/generate"
    payload = {
        "text": text,
        "filename": filename or output_path.name,
        "voice": voice.strip() if voice and voice.strip() else None,
        "use_emo_text": use_emo_text,
        "emo_alpha": emo_alpha,
        "interval_silence": interval_silence,
        "emo_vector": emo_vector,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=3600,  # 1 hour: IndexTTS-2 on CPU can be very slow
        )
        response.raise_for_status()

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
