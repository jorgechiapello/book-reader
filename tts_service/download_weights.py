#!/usr/bin/env python3
"""
Script to download IndexTTS-2 model weights.
These weights will be mounted as a volume when running the Docker container.

Prerequisites:
1. Activate virtual environment: source .venv/bin/activate
2. Install dependencies: pip install -r requirements.txt
"""

import sys
from pathlib import Path


def check_huggingface_installed():
    """Check if huggingface-hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False


def download_weights(weights_dir: Path):
    """Download the IndexTTS-2 model weights using the Python API."""
    from huggingface_hub import snapshot_download
    
    print("\n⬇️  Downloading IndexTTS-2 model weights...")
    print("This may take 10-30 minutes depending on your internet speed...\n")
    
    try:
        print("Starting download from HuggingFace...")
        print(f"Repository: IndexTeam/IndexTTS-2")
        print(f"Target: {weights_dir}")
        print("")
        
        snapshot_download(
            repo_id="IndexTeam/IndexTTS-2",
            local_dir=str(weights_dir),
            local_dir_use_symlinks=False
        )
        print("")
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Error during download: {e}")
        sys.exit(1)


def main():
    """Main function."""
    # Check if huggingface-hub is installed
    if not check_huggingface_installed():
        print("❌ Error: huggingface-hub is not installed")
        print("\nPlease install it first:")
        print("  1. Activate virtual environment: source .venv/bin/activate")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Run this script again\n")
        sys.exit(1)
    
    # Get weights directory from command line or use default
    if len(sys.argv) > 1:
        weights_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        weights_dir = Path.home() / "tts-weights"
    
    print("=" * 50)
    print("IndexTTS-2 Model Weights Download Script")
    print("=" * 50)
    print(f"\nDownload size: ~5.9 GB")
    print(f"Target directory: {weights_dir}\n")
    
    # Create weights directory
    print("📁 Creating weights directory...")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Download weights
    download_weights(weights_dir)
    
    # Success message
    print("\n✅ Download complete!")
    print(f"\nModel weights saved to: {weights_dir}\n")
    print("=" * 50)
    print("Next Steps:")
    print("=" * 50)
    print("\nRun the Docker container with mounted weights:\n")
    print(f"  docker run -d \\")
    print(f"    -p 8001:8001 \\")
    print(f"    -v {weights_dir}:/app/index-tts/checkpoints \\")
    print(f"    --name tts-service \\")
    print(f"    indextts-service\n")
    print("Check container logs:")
    print("  docker logs -f tts-service\n")


if __name__ == "__main__":
    main()
