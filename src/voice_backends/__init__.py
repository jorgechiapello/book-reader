from .base import VoiceBackend

def get_voice_backend(name: str) -> VoiceBackend:
    """Factory to get a VoiceBackend implementation."""
    if name == "indextts2":
        from .indextts2 import IndexTTS2Synth
        return IndexTTS2Synth()
    elif name == "styletts2":
        from .styletts2 import StyleTTS2Synth
        return StyleTTS2Synth()
    elif name == "qwen":
        from .qwen import QwenSynth
        return QwenSynth()
    else:
        raise ValueError(f"Unknown VoiceBackend: {name!r}")
