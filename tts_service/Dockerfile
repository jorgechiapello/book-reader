# ── Stage 1: Build OpenFst + patched pynini wheel ────────────────────────────
# All build tools stay here; only the .whl and runtime .so files are exported.
FROM python:3.10-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    autoconf \
    automake \
    libtool \
    zlib1g-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cython

# Build OpenFst 1.8.4 with --enable-grm (pynini requires the grm extensions).
RUN wget -q https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.4.tar.gz && \
    tar -xzf openfst-1.8.4.tar.gz && \
    cd openfst-1.8.4 && \
    ./configure --enable-grm && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && rm -rf openfst-1.8.4 openfst-1.8.4.tar.gz

# Patch pynini: OpenFst 1.8.4 renamed fst::StringJoin → fst::StrJoin but pynini's
# distributed stringmap.h still calls the old name. Build a native ARM64 wheel.
RUN pip download --no-deps --no-binary :all: pynini --dest /tmp/pynini_dl && \
    cd /tmp/pynini_dl && \
    tar -xzf pynini-*.tar.gz && \
    cd pynini-* && \
    sed -i 's/fst::StringJoin/fst::StrJoin/g' extensions/stringmap.h && \
    pip wheel --no-cache-dir --no-build-isolation . -w /wheels/ && \
    cd / && rm -rf /tmp/pynini_dl

# Build WeTextProcessing from GitHub (PyPI version 0.1.10 lacks tn.english).
# PR #325 modernized their repo to pyproject.toml, so standard pip wheel works.
# --no-deps prevents it from trying to rebuild pynini from source again.
RUN pip wheel --no-deps --no-cache-dir --find-links /wheels/ git+https://github.com/wenet-e2e/WeTextProcessing.git -w /wheels/

# ── Stage 2: Lean runtime image ───────────────────────────────────────────────
FROM python:3.10-slim-bookworm

WORKDIR /app

# Runtime system deps only — no build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenFst runtime libraries (pynini's .so links against them at runtime)
COPY --from=builder /usr/local/lib/ /usr/local/lib/
RUN ldconfig

# Copy our pre-built patched pynini ARM64 wheel from the builder stage
COPY --from=builder /wheels/ /wheels/

# Make /wheels/ visible to ALL pip invocations (including transitive dep resolution)
ENV PIP_FIND_LINKS=/wheels

# Install Python deps
COPY requirements.txt .
# Pin torch 2.5 / torchaudio 2.5: save() uses soundfile/sox (not TorchCodec).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Install our pre-built wheels directly by file path.
# Shell wildcard expansion grabs both pynini and WeTextProcessing regardless of casing.
# --no-deps bypasses WeTextProcessing's strict requirement for pynini < 2.1.7
RUN pip install --no-deps --no-cache-dir /wheels/*.whl

# Download IndexTTS source
RUN wget -q https://github.com/index-tts/index-tts/archive/refs/heads/main.tar.gz && \
    tar -xzf main.tar.gz && \
    mv index-tts-main index-tts && \
    rm main.tar.gz

# Fix CUDA-only deps (opencv, torch versions for CPU)
# hadolint ignore=SC2039,DL3025,DL4006
RUN cd index-tts && python3 - << 'EOF'
import toml
d = toml.load("pyproject.toml")

def filter_deps(deps):
    result = []
    for x in deps:
        if x.startswith("opencv-python=="):
            result.append(x.replace("opencv-python==", "opencv-python-headless=="))
        elif "torch==" in x and "torchaudio" not in x:
            result.append("torch>=2.0,<2.6")  # Match our pinned 2.5.0
        elif "torchaudio==" in x:
            result.append("torchaudio>=2.0,<2.6")  # Avoid 2.9+ TorchCodec
        elif "WeTextProcessing" in x or "wetext" in x:
            pass  # We install WeTextProcessing from GitHub, drop these completely
        else:
            result.append(x)
    return result

d["project"]["dependencies"] = filter_deps(d["project"].get("dependencies", []))
for key in d["project"].get("optional-dependencies", {}):
    d["project"]["optional-dependencies"][key] = filter_deps(
        d["project"]["optional-dependencies"][key]
    )
toml.dump(d, open("pyproject.toml", "w"))
print("Patched pyproject.toml successfully")
EOF

# Install pynini wheel version into a constraints file so pip never tries to
# rebuild it from source (even with build isolation re-enabled).
RUN pip show pynini | grep ^Version | awk '{print "pynini==" $2}' > /constraints.txt

# Install IndexTTS as a regular (non-editable) install.
# --constraint prevents pip from fetching a different pynini version.
RUN cd index-tts && pip install --no-cache-dir --constraint /constraints.txt .

# Copy API server and voices
COPY main.py .
COPY voices/ voices/

ENV PYTHONPATH="/app/index-tts"

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
