import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("tts_service")


class GenerateRequest(BaseModel):
    text: str
    filename: str = "output.wav"
    voice: str | None = None
    use_emo_text: bool = True
    emo_alpha: float = 1
    interval_silence: int = 200
    emo_vector: list[float] | None = None

# Add IndexTTS to path dynamically
TTS_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_TTS_DIR = os.path.join(TTS_SERVICE_DIR, "index-tts")
if os.path.exists(INDEX_TTS_DIR):
    sys.path.insert(0, INDEX_TTS_DIR)
elif os.path.exists('/app/index-tts'):
    sys.path.insert(0, '/app/index-tts')

app = FastAPI(title="IndexTTS-2 Service", version="1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request and its response status + duration."""
    start = time.perf_counter()
    body_bytes = await request.body()
    # Re-inject body so downstream handlers can read it
    async def receive():
        return {"type": "http.request", "body": body_bytes}
    request._receive = receive

    logger.info("→ %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("← %s %s  %d  %.0fms", request.method, request.url.path, response.status_code, elapsed)
    return response

# Global variables
tts_model = None
# Look for voices in the local directory or fallback to Docker /app/voices
VOICES_DIR = os.path.join(TTS_SERVICE_DIR, "voices")
if not os.path.exists(VOICES_DIR) and os.path.exists("/app/voices"):
    VOICES_DIR = "/app/voices"

# Model weights are in ~/tts-weights by default natively, or /app/index-tts/checkpoints in Docker
WEIGHTS_DIR = os.path.expanduser("~/tts-weights")
if not os.path.exists(WEIGHTS_DIR) and os.path.exists("/app/index-tts/checkpoints"):
    WEIGHTS_DIR = "/app/index-tts/checkpoints"

DEFAULT_VOICE = "Heisenberg"

def load_model():
    """Load IndexTTS-2 model on startup"""
    global tts_model
    
    try:
        import torch
        threads = max(1, os.cpu_count() or 1)
        torch.set_num_threads(threads)
        logger.info("Set PyTorch CPU threads to: %d", threads)
        
        logger.info("Loading IndexTTS-2 model...")
        
        # Determine best device (MPS OOMs on 18GB memory, use native ARM CPU instead)
        device = "cpu"
        logger.info("Using device: %s", device)

        # Import IndexTTS2
        from indextts.infer_v2 import IndexTTS2
        
        # Initialize the model
        tts_model = IndexTTS2(
            cfg_path=os.path.join(WEIGHTS_DIR, "config.yaml"),
            model_dir=WEIGHTS_DIR,
            device=device,
            use_fp16=True,  # Use FP16 for faster inference
            use_cuda_kernel=False,
            use_deepspeed=False
        )
            
        logger.info("IndexTTS-2 model loaded successfully")
        
    except Exception as e:
        logger.exception("Error loading model: %s", e)
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
def generate_audio(req: GenerateRequest):
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
    use_emo_text = req.use_emo_text
    emo_alpha = req.emo_alpha
    interval_silence = req.interval_silence
    emo_vector = req.emo_vector

    logger.info("Request received: %r (voice=%s, use_emo_text=%s)", text[:60], voice or DEFAULT_VOICE, use_emo_text)

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
        emo_vec_str = f" emo_vector={emo_vector}" if emo_vector else ""
        logger.info("Generating audio for: %s... (voice: %s, use_emo_text=%s)%s", text[:50], voice_path, use_emo_text, emo_vec_str)
        
        # Generate temporary output file in RAM disk if available, else local temp
        temp_output = f"/dev/shm/{filename}" if os.path.exists("/dev/shm") else f"/tmp/{filename}"
        
        tts_model.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=temp_output,
            use_emo_text=use_emo_text if emo_vector is None else False,
            emo_text=text,
            emo_alpha=emo_alpha,
            emo_vector=emo_vector,
            interval_silence=interval_silence,
            verbose=False,
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
        
        logger.info("Generated %d bytes of audio", len(audio_bytes))
        
        # Return audio as WAV file
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
            
    except Exception as e:
        logger.exception("Error generating audio: %s", e)
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


if __name__ == "__main__":
    import uvicorn
    import tomllib
    
    config_path = os.path.join(TTS_SERVICE_DIR, "uvicorn.toml")
    server_host = "0.0.0.0"
    server_port = 8000
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
                server_host = config.get("host", server_host)
                server_port = config.get("port", server_port)
        except Exception as e:
            logger.error("Failed to parse uvicorn.toml: %s", e)

    logger.info("Starting Uvicorn on %s:%s (from config)", server_host, server_port)
    uvicorn.run("main:app", host=server_host, port=server_port)