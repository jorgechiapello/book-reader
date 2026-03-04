# IndexTTS-2 Service

Text-to-speech engine with voice cloning, powered by [IndexTTS-2](https://github.com/index-tts/index-tts).

## Setup

### 1. Download Model Weights (~5.9 GB)

```bash
cd tts_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 download_weights.py
```

Weights are saved to `~/tts-weights/` by default.

### 2. Add Voice References

Place `.wav` files in the `voices/` directory. The default voice is `Heisenberg.wav`.

## Running

```bash
cd tts_service
bash setup_native.sh
source .venv-native/bin/activate
python3 main.py
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service status |
| `/health` | GET | Health check |
| `/generate` | POST | Generate speech |

### Generate Speech

```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "Heisenberg"}' \
  --output output.wav
```
