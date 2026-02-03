import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from workflows.indextts2.workflow import run_indextts2_workflow
    from voices import resolve_voice_sample
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure you are running this from the project root within the virtual environment.")
    sys.exit(1)

def main():
    # 1. Setup paths
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / "checkpoints" / "indextts2"
    cfg_path = checkpoints_dir / "config.yaml"
    
    # 2. Check for weights
    if not cfg_path.exists():
        print(f"\n[!] MISSSING WEIGHTS")
        print(f"IndexTTS-2 requires model weights to be placed in: {checkpoints_dir}")
        print("Please download 'config.yaml' and the model '.safetensors' to that folder.")
        print("You can find the models on the official IndexTTS repository or ModelScope.")
        return

    # 3. Resolve a voice
    try:
        voice_path = resolve_voice_sample("Heisenberg", project_root / "voices")
    except:
        voice_path = None
        print("[!] Warning: Heisenberg.wav not found, using default reference if available.")

    # 4. Run test
    test_text = "The quick brown fox jumps over the lazy dog. How expressive is this voice?"
    output_dir = project_root / "output" / "engine_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting IndexTTS-2 Engine Test ---")
    try:
        run_indextts2_workflow(
            text=test_text,
            ollama_model="llama3.2:3b",
            voice_sample_path=str(voice_path) if voice_path else "",
            output_dir=output_dir,
            chapter_title="Test Chapter",
            chapter_filename="test_output.txt"
        )
        print("\n[SUCCESS] Test completed! Check the output in: " + str(output_dir))
    except Exception as e:
        print(f"\n[ERROR] Engine test failed: {e}")

if __name__ == "__main__":
    main()
