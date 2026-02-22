FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    gcc \
    wget \
    libfst-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
# Pin torch 2.5 / torchaudio 2.5: save() uses soundfile/sox (not TorchCodec).
# TorchCodec in 2.9+ can introduce static in WAV output.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download IndexTTS source
RUN wget -q https://github.com/index-tts/index-tts/archive/refs/heads/main.tar.gz && \
    tar -xzf main.tar.gz && \
    mv index-tts-main index-tts && \
    rm main.tar.gz

# Fix CUDA-only deps automatically (opencv, torch versions for CPU)
# hadolint ignore=SC2039,DL3025,DL4006
RUN cd index-tts && python3 - << 'EOF'
import toml
d = toml.load("pyproject.toml")
deps = d["project"]["dependencies"]
for i, x in enumerate(deps):
    if x.startswith("opencv-python=="):
        deps[i] = x.replace("opencv-python==", "opencv-python-headless==")
    elif "torch==" in x and "torchaudio" not in x:
        deps[i] = "torch>=2.0,<2.6"  # Match our pinned 2.5.0
    elif "torchaudio==" in x:
        deps[i] = "torchaudio>=2.0,<2.6"  # Match our pinned 2.5.0 (avoid 2.9+ TorchCodec)
toml.dump(d, open("pyproject.toml", "w"))
EOF

# Install IndexTTS
RUN cd index-tts && pip install --no-cache-dir -e .

# Copy API server and voices
COPY main.py .
COPY voices/ voices/

ENV PYTHONPATH="/app/index-tts"

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
