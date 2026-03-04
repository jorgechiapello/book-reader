import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the current working directory, or specific paths if needed
load_dotenv()


class Config:
    """Central configuration managed via environment variables."""

    # Defaults
    DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # IndexTTS-2
    DEFAULT_INDEXTTS2_URL = os.getenv("INDEXTTS2_URL", "http://localhost:8001")
    DEFAULT_INDEXTTS2_VOICE = os.getenv("INDEXTTS2_VOICE", "Heisenberg")

    # StyleTTS2
    DEFAULT_STYLE_VOICE_SAMPLE = os.getenv("STYLE_VOICE_SAMPLE", "voices/Heisenberg.wav")

    # Qwen-TTS
    DEFAULT_QWEN_COMFY_URL = os.getenv("QWEN_COMFY_URL", "http://localhost:8188")
    DEFAULT_QWEN_VOICE = os.getenv("QWEN_VOICE", "Serena")

    # General Output Defaults
    DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
