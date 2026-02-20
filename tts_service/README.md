# IndexTTS-2 Service

Docker service for IndexTTS-2 text-to-speech engine with voice cloning support.

## Prerequisites

### Docker Memory Requirements
**Important:** This service requires at least **16 GB** of Docker memory to run.

**To increase Docker memory on macOS:**
1. Open **Docker Desktop**
2. Go to **Settings** → **Resources** → **Memory**
3. Set to **16 GB** (minimum)
4. Click **Apply & Restart**

### Apple Silicon (M-series) Note
This image is built for **AMD64/x86_64** architecture and runs via Rosetta 2 emulation on Apple Silicon Macs. This is necessary because the `pynini` dependency doesn't compile on ARM64. Performance is still good, with only ~10-20% overhead.

## Quick Start

### 1. Build the Docker Image

```bash
cd tts_service
docker build --platform linux/amd64 -t indextts-service .
```

**Build time:** 20-40 minutes (due to cross-platform compilation)  
**Image size:** ~4 GB (includes PyTorch CPU + dependencies)

### 2. Setup Virtual Environment

Create and activate a virtual environment in the project to keep dependencies isolated:

```bash
cd tts_service

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

This installs:
- `fastapi` - Web framework for TTS service
- `uvicorn` - ASGI server
- `huggingface-hub` - For downloading model weights

**Note:** You can deactivate the virtual environment later with:
```bash
deactivate
```

### 4. Download Model Weights

**Option A: Using the download script (automated)**

```bash
python3 download_weights.py
```

**Option B: Manual download**

```bash
python << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="IndexTeam/IndexTTS-2",
    local_dir="/Users/jorgealbertochiapellosaid/tts-weights"
)
print("✅ Download complete!")
EOF
```

**Custom directory:**
```bash
python3 download_weights.py /path/to/custom/dir
```

**Download info:**
- Size: ~5.9 GB
- Time: 10-30 minutes (depends on internet speed)
- Default location: `~/tts-weights`

### 5. Prepare Voice Reference (Optional)

For voice cloning, add WAV files to the `voices/` folder:

```bash
# Add your voice samples (WAV format recommended)
cp your-voice.wav tts_service/voices/
```

The service looks for `voices/Heisenberg.wav` by default, but you can use any voice file.

### 6. Run the Container

Mount the weights and voices folders, then start the service:

```bash
docker run -d \
  -p 8001:8001 \
  -v ~/tts-weights:/app/index-tts/checkpoints \
  -v ~/repos/book-reader/tts_service/voices:/app/voices \
  --name tts-service \
  indextts-service
```

**Command breakdown:**
- `-d` - Run in detached mode (background)
- `-p 8001:8001` - Map port 8001 (host:container)
- `-v ~/tts-weights:/app/index-tts/checkpoints` - Mount model weights (~5.9 GB)
- `-v ~/repos/book-reader/tts_service/voices:/app/voices` - Mount voice references for cloning
- `--name tts-service` - Container name
- `indextts-service` - Image name

**Note:** If you don't mount the voices folder, the service will use a random voice instead of voice cloning.

**Custom paths:**
Adjust the paths to match your setup:
```bash
docker run -d \
  -p 8001:8001 \
  -v /path/to/your/weights:/app/index-tts/checkpoints \
  -v /path/to/your/voices:/app/voices \
  --name tts-service \
  indextts-service
```

### 7. Check Logs

**Monitor startup progress:**

```bash
docker logs -f tts-service
```

**Startup process (takes 1-2 minutes):**
1. ⏳ Building Chinese text normalizer FST (~30-60 seconds)
2. ⏳ Loading model weights from mounted volume (~30-60 seconds)
3. ✅ "Application startup complete"
4. ✅ "Uvicorn running on http://0.0.0.0:8001"

**Expected log output:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
Loading IndexTTS-2 model...
>> GPT weights restored from: /app/index-tts/checkpoints/gpt.pth
>> s2mel weights restored from: /app/index-tts/checkpoints/s2mel.pth
✓ Voice reference loaded: /app/voices/Heisenberg.wav
✓ IndexTTS-2 model loaded successfully!
```

Press `Ctrl+C` to stop following logs.

## Container Management

**Stop the service:**
```bash
docker stop tts-service
```

**Start the service:**
```bash
docker start tts-service
```

**Remove the container:**
```bash
docker stop tts-service
docker rm tts-service
```

**Restart the service:**
```bash
docker restart tts-service
```

## API Usage

The service exposes the following endpoints:

### Check Service Status
```bash
curl http://localhost:8001/
```

**Response:**
```json
{
  "status": "IndexTTS-2 Service Online",
  "model_loaded": true,
  "voice_loaded": true
}
```

### Health Check
```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "voice": "loaded"
}
```

### Generate Speech (Voice Cloning)
```bash
curl -X POST "http://localhost:8001/generate?text=Hello%20world&filename=output.wav" \
  --output output.wav
```

**Parameters:**
- `text` (required): Text to synthesize
- `filename` (optional): Output filename (default: "output.wav")

**Response:** WAV audio file

**With voice cloning:** If `voices/Heisenberg.wav` is mounted, uses that voice  
**Without voice:** Uses random speaker characteristics

## Troubleshooting

### Container Exits with Code 137 (OOM - Out of Memory)
**Problem:** Docker runs out of memory while loading the 5.9 GB model.

**Solution:** Increase Docker memory to 16 GB (see Prerequisites above).

### Port 8001 Already in Use
**Problem:** Another service is using port 8001.

**Solution:** Use a different host port:
```bash
docker run -d -p 8002:8001 ...  # Maps host:8002 to container:8001
```

### Voice Reference Not Found
**Problem:** Logs show "Voice reference not found at /app/voices/Heisenberg.wav"

**Solution:** 
1. Check that `voices/Heisenberg.wav` exists in your project
2. Verify the voices volume mount in your `docker run` command
3. Or, let it use random voice (works without voice file)

### Slow Build Times
**Problem:** Docker build takes 30-40 minutes.

**Cause:** Cross-platform compilation (AMD64 on ARM64 Mac).

**Expected:** This is normal. Subsequent builds use cache and are faster.

### Model Loading Hangs
**Problem:** Container starts but model never finishes loading.

**Check:**
1. Docker has 16 GB memory allocated
2. Weights are properly mounted: `docker exec tts-service ls /app/index-tts/checkpoints`

## Image Optimization

This image uses a multi-stage build to minimize size:

- **Final image:** ~4 GB (PyTorch CPU + dependencies + IndexTTS-2)
- **Model weights:** 5.9 GB (kept separate, mounted as volume)
- **Total storage:** ~10 GB

**Why so large?**
- PyTorch CPU: ~1.5-2 GB (unavoidable for deep learning)
- Audio processing libraries (ffmpeg, librosa): ~300 MB
- Text normalization (pynini + wetextprocessing): ~200 MB
- Python dependencies: ~500 MB
- IndexTTS-2 application: ~100 MB

Optimizations applied:
- Multi-stage build (separate builder/runtime)
- `--no-cache-dir` for all pip installs
- Removed build tools from final image
- Using `opencv-python-headless` (no GUI dependencies)
- Clean Python bytecode after build

## Notes

- Model weights are **not included** in the Docker image
- Weights must be downloaded separately and mounted as a volume
- Weights persist on host machine, not in container
- Rebuilding the image won't require re-downloading weights
- **Virtual environment**: The download script creates a `.venv/` directory in `tts_service/` for isolated dependencies (already in `.gitignore`)

## Project Structure

After running the download script, you'll have:

```
tts_service/
├── .venv/                    # Virtual environment (gitignored)
├── voices/                   # Voice reference files for cloning
│   ├── Heisenberg.wav       # Default voice reference (~10 MB)
│   └── README.md            # Voice file instructions
├── download_weights.py       # Model weights download script
├── Dockerfile                # Multi-stage Docker build (AMD64)
├── main.py                   # FastAPI TTS service
├── requirements.txt          # Python dependencies for download script
└── README.md                 # This file

~/tts-weights/                # Model weights (mounted as volume)
├── config.yaml               # Model configuration
├── gpt.pth                   # GPT model (~3.5 GB)
├── s2mel.pth                 # Spectrogram model (~1.2 GB)
├── bpe.model                 # Tokenizer
└── ...                       # Other checkpoints & cache
```

## Additional Commands

**Check if volumes are mounted correctly:**
```bash
# Check weights
docker exec tts-service ls -lh /app/index-tts/checkpoints

# Check voices
docker exec tts-service ls -lh /app/voices
```

**Clean up Docker cache:**
```bash
docker system prune -a --volumes
```

**Rebuild without cache:**
```bash
docker build --no-cache --platform linux/amd64 -t indextts-service .
```
