import argparse
import logging
from pathlib import Path

from config import Config
from pipeline import build_pipeline
from text_extractors import extract_chapters
from voice_backends.base import SynthesisContext

def build_parser():
    parser = argparse.ArgumentParser(description="TTS Service CLI - 3-Stage Pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default=Config.DEFAULT_OUTPUT_DIR,
        help="Base directory for output files (default: from .env OUTPUT_DIR)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: script
    # Generates a .json script from an input document.
    parser_script = subparsers.add_parser("script", help="Generate an annotated script from text.")
    parser_script.add_argument("input", type=str, help="Path to input file (txt, epub, pdf)")
    parser_script.add_argument(
        "--writer", 
        type=str, 
        choices=["rule_based", "emotional_analyst"],
        default="rule_based",
        help="ScriptWriter strategy to use (default: rule_based)"
    )

    # Command: audio
    # Generates .wav files from a .json script.
    parser_audio = subparsers.add_parser("audio", help="Generate audio from an annotated script.")
    parser_audio.add_argument("script", type=str, help="Path to the .json script file")
    parser_audio.add_argument(
        "--voice-backend", 
        type=str, 
        choices=["indextts2", "styletts2", "qwen"],
        default="indextts2",
        help="VoiceBackend strategy to use (default: indextts2)"
    )
    parser_audio.add_argument("--voice", type=str, help="Override default voice name in .env")
    parser_audio.add_argument("--voice-sample", type=str, help="Override default voice sample path in .env")

    # Command: run
    # Runs the full pipeline end-to-end.
    parser_run = subparsers.add_parser("run", help="Run full pipeline: ingest -> script -> audio")
    parser_run.add_argument("input", type=str, help="Path to input file (txt, epub, pdf)")
    parser_run.add_argument(
        "--writer", 
        type=str, 
        choices=["rule_based", "emotional_analyst"],
        default="rule_based",
        help="ScriptWriter strategy to use"
    )
    parser_run.add_argument(
        "--voice-backend", 
        type=str, 
        choices=["indextts2", "styletts2", "qwen"],
        default="indextts2",
        help="VoiceBackend strategy to use"
    )
    parser_run.add_argument("--voice", type=str, help="Override default voice name")
    parser_run.add_argument("--voice-sample", type=str, help="Override default voice sample path")

    return parser


def build_context(args, command: str) -> SynthesisContext:
    ctx = SynthesisContext()
    
    backend_name = getattr(args, "voice_backend", "indextts2")
    
    # Map CLI overrides or .env defaults based on the backend
    if backend_name == "indextts2":
        ctx.voice_name = getattr(args, "voice", None) or Config.DEFAULT_INDEXTTS2_VOICE
        ctx.tts_url = Config.DEFAULT_INDEXTTS2_URL
    elif backend_name == "styletts2":
        sample = getattr(args, "voice_sample", None) or Config.DEFAULT_STYLE_VOICE_SAMPLE
        ctx.voice_sample_path = Path(sample)
    elif backend_name == "qwen":
        ctx.voice_name = getattr(args, "voice", None) or Config.DEFAULT_QWEN_VOICE
        ctx.tts_url = Config.DEFAULT_QWEN_COMFY_URL
        # Optional clone sample
        sample = getattr(args, "voice_sample", None)
        if sample:
            ctx.voice_sample_path = Path(sample)

    return ctx


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "script":
        input_path = Path(args.input)
        print(f"Ingesting book from: {input_path}")
        chapters = extract_chapters(input_path)
        
        pipeline = build_pipeline(args.writer, "indextts2") # Backend doesn't matter for stage 1 only
        
        for i, chapter in enumerate(chapters, 1):
            print(f"Processing chapter {i}/{len(chapters)}: {chapter.title}")
            script_filename = f"{i:03d}_{chapter.title}_script.json"
            script_output_path = output_dir / script_filename
            pipeline.run_stage_1(chapter.segments, script_output_path)
            
        print(f"Success. Scripts saved to {output_dir}")

    elif args.command == "audio":
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"Error: Script file not found: {script_path}")
            return
            
        ctx = build_context(args, "audio")
        
        pipeline = build_pipeline("rule_based", args.voice_backend) # Writer doesn't matter for stage 2 only
        
        audio_filename = script_path.stem.replace("_script", "") + ".wav"
        audio_output_path = output_dir / audio_filename
        
        pipeline.run_stage_2(script_path, audio_output_path, ctx)
        print(f"Success. Audio saved to {audio_output_path}")

    elif args.command == "run":
        input_path = Path(args.input)
        print(f"Ingesting book from: {input_path}")
        chapters = extract_chapters(input_path)
        
        ctx = build_context(args, "run")
        pipeline = build_pipeline(args.writer, args.voice_backend)
        
        for i, chapter in enumerate(chapters, 1):
            print(f"Processing chapter {i}/{len(chapters)}: {chapter.title}")
            chapter_filename = f"{i:03d}_{chapter.title}.txt" # Dummy extension
            pipeline.run_pipeline(chapter.segments, output_dir, chapter_filename, ctx)
            
        print("Success. Full pipeline completed.")


if __name__ == "__main__":
    main()
