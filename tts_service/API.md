# IndexTTS-2 Service API

REST API for the IndexTTS-2 text-to-speech engine with voice cloning and emotion control.

**Base URL:** `http://localhost:8001`

---

## Endpoints

### GET /

Returns service status.

**Response:**
```json
{
  "status": "IndexTTS-2 Service Online",
  "model_loaded": true,
  "voices_dir": "/app/voices",
  "default_voice": "Heisenberg"
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "default_voice": "Heisenberg",
  "default_voice_available": true
}
```

---

### POST /generate

Generates speech from text. Accepts JSON body and returns a WAV file.

**Request (JSON body):**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | — | Text to synthesize. Cannot be empty. |
| `voice` | string \| null | No | `"Heisenberg"` | Voice name. Must match a WAV file in the server's voices directory (e.g. `Heisenberg`, `Heisenberg.wav`). Resolved under `/app/voices/`. |
| `filename` | string | No | `"output.wav"` | Filename for the returned WAV (Content-Disposition). |
| `use_emo_text` | boolean | No | `true` | When true, emotion is inferred from the text. Ignored if `emo_vector` is provided. |
| `emo_alpha` | float | No | `1` | Emotion blend weight (0.0–1.0). Kept at 1; emotion controlled via text and emo_vector. |
| `interval_silence` | integer | No | `200` | Milliseconds of silence between synthesis segments. |
| `emo_vector` | array of 8 floats \| null | No | `null` | Emotion vector `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`. Each value 0–1. When provided, overrides `use_emo_text`. |

**Example:**
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world.", "voice": "Heisenberg", "filename": "hello.wav"}' \
  --output hello.wav
```

**Example with emo_vector:**
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "She whispered with fear.", "emo_vector": [0, 0, 0.2, 0.8, 0, 0, 0, 0.2]}' \
  --output output.wav
```

**Response:**
- **200:** WAV audio, `Content-Type: audio/wav`, `Content-Disposition: attachment; filename=<filename>`
- **400:** Invalid request (empty text, unknown voice)
- **503:** Model not loaded
- **500:** Synthesis error
