# ============================================
# Stage 1: Builder - Install and compile dependencies
# ============================================
FROM --platform=linux/amd64 python:3.10-slim as builder

WORKDIR /app

# Install build dependencies (will be removed from final image)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    ca-certificates \
    libfst-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Download IndexTTS-2 repository (using wget instead of git to save space)
RUN wget -q https://github.com/index-tts/index-tts/archive/refs/heads/main.tar.gz && \
    tar -xzf main.tar.gz && \
    mv index-tts-main /app/index-tts && \
    rm main.tar.gz

# Install CPU-only PyTorch (no cache to save space)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install toml library for editing pyproject.toml
RUN pip install --no-cache-dir toml

# Patch pyproject.toml to fix torch versions for CPU
RUN cd /app/index-tts && \
    python3 << 'EOF'
import toml

# Read the pyproject.toml
with open('pyproject.toml', 'r') as f:
    data = toml.load(f)

# Remove CUDA-specific torch versions and replace with generic versions
deps = data['project']['dependencies']
for i, dep in enumerate(deps):
    if 'torch==2.8.0+cu128' in dep:
        deps[i] = 'torch>=2.8.0'
    elif 'torchaudio==2.8.0+cu128' in dep:
        deps[i] = 'torchaudio>=2.8.0'
    # Replace opencv-python with headless version (no GUI dependencies)
    elif dep.startswith('opencv-python=='):
        deps[i] = dep.replace('opencv-python==', 'opencv-python-headless==')

# Write back
with open('pyproject.toml', 'w') as f:
    toml.dump(data, f)

print("Patched pyproject.toml:")
print("- Fixed torch versions for CPU")
print("- Replaced opencv-python with opencv-python-headless")
EOF

# Install IndexTTS-2 dependencies (includes wetextprocessing with pynini)
RUN cd /app/index-tts && \
    pip install --no-cache-dir -e ".[all]" || pip install --no-cache-dir -e .

# Copy and install TTS service requirements (FastAPI, uvicorn, etc.)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Clean up pip cache and Python bytecode
RUN find /usr/local/lib/python3.10 -name '*.pyc' -delete && \
    find /usr/local/lib/python3.10 -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy IndexTTS-2 application
COPY --from=builder /app/index-tts /app/index-tts

# Copy server code
COPY main.py /app/

# Set Python path to include IndexTTS
ENV PYTHONPATH="/app/index-tts"

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
