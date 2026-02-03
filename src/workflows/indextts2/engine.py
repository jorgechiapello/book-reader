import os
import sys
from pathlib import Path
from typing import Optional

# Global lazy-loaded engine
_indextts2_engine = None

def _get_indextts2_engine():
    """Lazy-load the IndexTTS2 model as a standard package."""
    global _indextts2_engine
    if _indextts2_engine is None:
        try:
            from indextts.infer_v2 import IndexTTS2
            print("  Loading IndexTTS2 model weights into memory...")
            
            # Checkpoints should be placed in a 'checkpoints' folder in the project root
            # or a path defined by the user.
            base_dir = Path(__file__).parents[3] # Root of the project
            cfg_path = base_dir / "checkpoints/indextts2/config.yaml"
            model_dir = base_dir / "checkpoints/indextts2"
            
            if not cfg_path.exists():
                 print(f"  [WARNING] IndexTTS2 config not found at {cfg_path}")
                 print("  Ensure you have downloaded the checkpoints to 'checkpoints/indextts2/'")
            
            _indextts2_engine = IndexTTS2(
                cfg_path=str(cfg_path), 
                model_dir=str(model_dir), 
                use_fp16=True
            )
            print("  IndexTTS2 engine initialized.")
        except ImportError as e:
            raise ImportError(
                "IndexTTS2 package not found. Please run: pip install -r requirements.txt. "
                f"Error: {e}"
            )
        except Exception as e:
            print(f"  Error initializing IndexTTS2: {e}")
            raise
            
    return _indextts2_engine

def generate_audio_with_indextts2(
    text: str,
    output_path: Path,
    soft_instruction: str = "",
    reference_audio_path: Optional[str] = None,
    mode: str = "Auto", 
    model_choice: str = "1.7B", 
):
    """
    Synthesizes audio using the indextts library.
    """
    engine = _get_indextts2_engine()
    
    print(f"  Generating audio with IndexTTS-2 package...")
    print(f"  Instruction: {soft_instruction}")
    
    try:
        engine.infer(
            text=text,
            spk_audio_prompt=str(reference_audio_path) if reference_audio_path else None,
            emo_text=soft_instruction,
            use_emo_text=bool(soft_instruction),
            output_path=str(output_path)
        )
        
        if output_path.exists():
            print(f"  Audio successfully saved to {output_path}")
        else:
            raise RuntimeError(f"Engine reported success but {output_path} was not created.")
            
    except Exception as e:
        print(f"  Error during IndexTTS2 inference: {e}")
        raise

def combine_audio_segments(segments: list[Path], output_path: Path):
    """
    Utility to combine multiple wav files.
    """
    from pydub import AudioSegment
    
    combined = AudioSegment.empty()
    for p in segments:
        if p.exists():
            seg_audio = AudioSegment.from_wav(str(p))
            combined += seg_audio
    
    combined.export(str(output_path), format="wav")
