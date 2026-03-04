#!/bin/bash
# ── Native macOS Setup for IndexTTS-2 ────────────────────────────────────────
# This script mirrors the Dockerfile but targets native Apple Silicon execution.
# Running this script is sufficient to set up the entire TTS engine locally.
#
# Usage:
#   cd tts_service
#   python3 -m venv .venv-native
#   bash setup_native.sh
#
# After setup:
#   source .venv-native/bin/activate
#   python main.py
# ─────────────────────────────────────────────────────────────────────────────
set -ex

cd "$(dirname "$0")"

# Fix xcrun permission issues when run via automated tools
export TMPDIR=/tmp

VENV_DIR="$PWD/.venv-native"

# ── Step 0: Create venv if it doesn't exist ──────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install cython==3.0.11 wheel setuptools build toml

# ── Step 1: Build OpenFST from source (ARM64) ───────────────────────────────
# pynini requires OpenFST with --enable-grm. No homebrew formula provides this.
if [ ! -f "$VENV_DIR/bin/fstinfo" ]; then
    echo "Building OpenFST 1.8.4..."
    mkdir -p build_deps
    cd build_deps
    curl -sSL https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.4.tar.gz -o openfst-1.8.4.tar.gz
    tar -xzf openfst-1.8.4.tar.gz
    cd openfst-1.8.4
    ./configure --enable-grm --prefix="$VENV_DIR"
    make -j$(sysctl -n hw.ncpu)
    make install
    cd ../..
fi

# Set compiler flags so pynini finds our local openfst
export CFLAGS="-I$VENV_DIR/include"
export CXXFLAGS="-I$VENV_DIR/include"
export LDFLAGS="-L$VENV_DIR/lib -Wl,-rpath,$VENV_DIR/lib"

# ── Step 2: Build pynini from source (patched for OpenFST 1.8.4) ────────────
# OpenFST 1.8.4 renamed fst::StringJoin → fst::StrJoin but pynini 2.1.6
# still references the old name. We patch stringmap.h before building.
if ! python -c "import pynini" &> /dev/null; then
    echo "Building pynini 2.1.6..."
    mkdir -p build_deps/pynini_build
    cd build_deps/pynini_build
    pip download --no-deps --no-binary :all: pynini==2.1.6
    tar -xzf pynini-2.1.6.tar.gz
    cd pynini-2.1.6
    sed -i '' 's/fst::StringJoin/fst::StrJoin/g' extensions/stringmap.h
    pip install .
    cd ../../..
fi

# ── Step 3: Install WeTextProcessing (from GitHub, PyPI version lacks English) ─
if ! python -c "import tn" &> /dev/null; then
    echo "Installing WeTextProcessing..."
    pip install --no-deps git+https://github.com/wenet-e2e/WeTextProcessing.git@bb145729c903fac2d9fddf6b9077f352f3fc2816
fi

# ── Step 4: Install wetext (macOS-only text normalizer used by IndexTTS-2) ──
# On Linux, IndexTTS-2 uses WeTextProcessing directly.
# On macOS (sys_platform != 'linux'), it uses the "wetext" PyPI package instead.
if ! python -c "import wetext" &> /dev/null; then
    echo "Installing wetext..."
    pip install wetext
fi

# ── Step 5: Install PyTorch (native macOS ARM64 build) ──────────────────────
# Pin to 2.5.x (same as Dockerfile) — torchaudio 2.10+ defaults to "torchcodec"
# backend which produces garbled WAV output. Version 2.5 uses soundfile instead.
echo "Installing PyTorch..."
pip install 'torch>=2.5.0,<2.6' 'torchaudio>=2.5.0,<2.6' soundfile

# ── Step 6: Install server dependencies ──────────────────────────────────────
echo "Installing server dependencies..."
pip install fastapi uvicorn python-multipart huggingface-hub

# ── Step 7: Install IndexTTS-2 ──────────────────────────────────────────────
echo "Installing IndexTTS-2..."
if [ ! -d "index-tts" ]; then
    curl -sSL https://github.com/index-tts/index-tts/archive/1698b32033f38a034572891aed698609da2ff392.tar.gz -o indextts.tar.gz
    tar -xzf indextts.tar.gz
    mv index-tts-1698b32033f38a034572891aed698609da2ff392 index-tts
    rm indextts.tar.gz

    cd index-tts
    # Patch pyproject.toml: remove pinned torch/torchaudio versions and
    # WeTextProcessing (we install these separately above).
    python -c '
import toml
d = toml.load("pyproject.toml")
def filter_deps(deps):
    result = []
    for x in deps:
        if x.startswith("opencv-python=="):
            result.append(x.replace("opencv-python==", "opencv-python-headless=="))
        elif "torch==" in x or "torchaudio==" in x:
            pass
        elif "WeTextProcessing" in x or "wetext" in x:
            pass
        else:
            result.append(x)
    return result
d["project"]["dependencies"] = filter_deps(d["project"].get("dependencies", []))
for key in d["project"].get("optional-dependencies", {}):
    d["project"]["optional-dependencies"][key] = filter_deps(d["project"]["optional-dependencies"][key])
toml.dump(d, open("pyproject.toml", "w"))
'
    pip install -e .
    cd ..
fi

# ── Step 8: Ensure voices directory exists ───────────────────────────────────
mkdir -p voices

echo ""
echo "========================================="
echo "✅ Native macOS setup complete!"
echo ""
echo "To start the server:"
echo "  cd tts_service"
echo "  source .venv-native/bin/activate"
echo "  python main.py"
echo ""
echo "Make sure you have model weights in ~/tts-weights/"
echo "(run: python download_weights.py)"
echo "========================================="
