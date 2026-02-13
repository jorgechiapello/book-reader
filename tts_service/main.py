from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import os
import sys

# Add IndexTTS to path
sys.path.insert(0, '/app/index-tts')

app = FastAPI()

# Global variables
tts_model = None
voice_reference = None

def load_model():
    """Load IndexTTS-2 model on startup"""
    global tts_model, voice_reference
    
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
        
        # Set voice reference
        voice_reference = "/app/voices/Heisenberg.wav"
        
        if os.path.exists(voice_reference):
            print(f"✓ Voice reference loaded: {voice_reference}")
        else:
            print(f"⚠ Warning: Voice reference not found at {voice_reference}")
            voice_reference = None
            
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


@app.get("/")
def read_root():
    return {
        "status": "IndexTTS-2 Service Online",
        "model_loaded": tts_model is not None,
        "voice_loaded": voice_reference is not None
    }


@app.post("/generate")
async def generate_audio(text: str, filename: str = "output.wav"):
    """
    Generate audio using IndexTTS-2
    
    Args:
        text: Text to synthesize
        filename: Output filename
    
    Returns:
        Audio file as WAV bytes
    """
    if tts_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Text parameter is required and cannot be empty"
        )
    
    if voice_reference is None or not os.path.exists(voice_reference):
        raise HTTPException(
            status_code=503,
            detail="Voice reference not available"
        )
    
    try:
        print(f"Generating audio for: {text[:50]}...")
        
        # Generate temporary output file
        temp_output = f"/tmp/{filename}"
        
        # Generate audio using IndexTTS-2 with voice cloning
        tts_model.infer(
            spk_audio_prompt=voice_reference,
            text=text,
            output_path=temp_output,
            use_emo_text=True,  # Use text to guide emotions
            emo_alpha=0.6,  # Natural sounding speech
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
    return {
        "status": "healthy",
        "model": "loaded" if tts_model else "not loaded",
        "voice": "loaded" if voice_reference else "not loaded"
    }