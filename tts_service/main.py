from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import os
import sys


class GenerateRequest(BaseModel):
    text: str
    filename: str = "output.wav"
    voice: str | None = None
    soft_instruction: str | None = None  # Emotion/pacing hint, e.g. "Calm and confident narration."

# Add IndexTTS to path
sys.path.insert(0, '/app/index-tts')

app = FastAPI()

# Global variables
tts_model = None
VOICES_DIR = "/app/voices"
DEFAULT_VOICE = "Heisenberg"

def load_model():
    """Load IndexTTS-2 model on startup"""
    global tts_model
    
    try:
        print("Loading IndexTTS-2 model...")
        
        # Import IndexTTS2
        from indextts.infer_v2 import IndexTTS2
        
        # Initialize the model
        tts_model = IndexTTS2(
            cfg_path="/app/index-tts/checkpoints/config.yaml",
            model_dir="/app/index-tts/checkpoints",
            use_fp16=True,  # Use FP16 for faster inference
            use_cuda_kernel=False,
            use_deepspeed=False
        )
            
        print("✓ IndexTTS-2 model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        tts_model = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup"""
    load_model()


def resolve_voice_path(voice: str | None) -> str | None:
    """
    Resolve voice name to a file path under /app/voices/.
    If no voice is passed, defaults to Heisenberg.
    """
    name = (voice or "").strip() or DEFAULT_VOICE
    # Try with .wav extension if not present
    for path in [
        os.path.join(VOICES_DIR, name),
        os.path.join(VOICES_DIR, f"{name}.wav"),
    ]:
        if os.path.exists(path):
            return path
    return None


@app.get("/")
def read_root():
    return {
        "status": "IndexTTS-2 Service Online",
        "model_loaded": tts_model is not None,
        "voices_dir": VOICES_DIR,
        "default_voice": DEFAULT_VOICE,
    }


@app.post("/generate")
async def generate_audio(req: GenerateRequest):
    """
    Generate audio using IndexTTS-2.

    Accepts JSON body: {"text": "...", "filename": "output.wav", "voice": "Heisenberg"}
    Use JSON body (not query params) to avoid URL length limits for long text.

    Returns:
        Audio file as WAV bytes.
    """
    text = req.text
    voice = req.voice
    filename = req.filename
    emo_text = req.soft_instruction
    if tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="text is required and cannot be empty"
        )
    
    voice_path = resolve_voice_path(voice)
    if not voice_path:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice or DEFAULT_VOICE}' not found in {VOICES_DIR}. "
                   f"Add a .wav file (e.g. {DEFAULT_VOICE}.wav) to the voices directory."
        )
    
    try:
        instr_suffix = f" [instruction: {emo_text[:50]}...]" if emo_text and len(emo_text) > 50 else (f" [instruction: {emo_text}]" if emo_text else "")
        print(f"Generating audio for: {text[:50]}... (voice: {voice_path}){instr_suffix}")
        
        # Generate temporary output file
        temp_output = f"/tmp/{filename}"
        
        # Generate audio using IndexTTS-2 with the requested voice
        # emo_text: soft instruction from CrewAI (e.g. "Calm and confident narration") guides emotion
        tts_model.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=temp_output,
            use_emo_text=True,
            emo_text=emo_text or text,  # Use soft_instruction if provided, else main text
            emo_alpha=0.6,
            verbose=False
        )
        
        # Read the generated audio file
        if not os.path.exists(temp_output):
            raise HTTPException(
                status_code=500,
                detail="Audio generation failed: output file not created"
            )
        
        with open(temp_output, "rb") as f:
            audio_bytes = f.read()
        
        # Cleanup
        try:
            os.remove(temp_output)
        except:
            pass
        
        print(f"✓ Generated {len(audio_bytes)} bytes of audio")
        
        # Return audio as WAV file
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    default_voice_path = resolve_voice_path(None)
    return {
        "status": "healthy",
        "model": "loaded" if tts_model else "not loaded",
        "default_voice": DEFAULT_VOICE,
        "default_voice_available": default_voice_path is not None,
    }