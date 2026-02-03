import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from formats import Chapter, extract_chapters, slugify
from voices import resolve_voice_sample


def build_output_dir(output_dir: Path, book_slug: str) -> Path:
    return output_dir / book_slug / "chapters"


def write_manifest(output_dir: Path, book_slug: str, chapters: List[Chapter]) -> Path:
    manifest = {
        "book_slug": book_slug,
        "chapters": [
            {"index": index + 1, "title": chapter.title, "file": f"{index+1:03d}_{slugify(chapter.title)}.txt"}
            for index, chapter in enumerate(chapters)
        ],
    }
    manifest_path = output_dir / book_slug / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_chapter_files(chapters: List[Chapter], chapters_dir: Path) -> List[Path]:
    chapters_dir.mkdir(parents=True, exist_ok=True)
    chapter_paths: List[Path] = []
    for index, chapter in enumerate(chapters, start=1):
        filename = f"{index:03d}_{slugify(chapter.title)}.txt"
        path = chapters_dir / filename
        path.write_text(chapter.text, encoding="utf-8")
        chapter_paths.append(path)
    return chapter_paths


def load_manifest(output_dir: Path, book_slug: str) -> Dict:
    manifest_path = output_dir / book_slug / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ingest_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    book_slug = slugify(args.title or input_path.stem)
    chapters = extract_chapters(input_path)
    chapters_dir = build_output_dir(output_dir, book_slug)
    write_chapter_files(chapters, chapters_dir)
    write_manifest(output_dir, book_slug, chapters)
    print(f"Wrote {len(chapters)} chapters to {chapters_dir}")


def speak_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    book_slug = slugify(args.title or input_path.stem)
    manifest = load_manifest(output_dir, book_slug)
    chapters_dir = build_output_dir(output_dir, book_slug)
    voice_sample = resolve_voice_sample(args.voice, Path(args.voices_dir))

    # Resolve style reference if provided
    style_ref: Optional[Path] = None
    if hasattr(args, "style_ref") and args.style_ref:
        style_ref = Path(args.style_ref)
        if not style_ref.exists():
            raise FileNotFoundError(f"Style reference not found: {style_ref}")

    ollama_model = getattr(args, "ollama_model", "llama3.2")

    for chapter in manifest["chapters"]:
        chapter_path = chapters_dir / chapter["file"]
        if not chapter_path.exists():
            raise FileNotFoundError(f"Missing chapter file: {chapter_path}")
        audio_path = chapters_dir / chapter["file"].replace(".txt", ".wav")
        text = chapter_path.read_text(encoding="utf-8")

        print(f"Processing: {chapter['title']}")

        # Use LLM-powered sentiment analysis
        from workflows.workflow_styletts2 import run_styletts2_workflow
        
        run_styletts2_workflow(
            text=text,
            ollama_model=ollama_model,
            voice_sample_path=voice_sample,
            output_dir=chapters_dir,
            chapter_title=chapter["title"],
            chapter_filename=chapter["file"],
        )
        



def run_command(args: argparse.Namespace) -> None:
    ingest_command(args)
    speak_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local audiobook generator")
    parser.add_argument("--input", required=True, help="Path to TXT/EPUB/PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--title", help="Optional book title override")
    parser.add_argument("--voices-dir", default="voices", help="Directory containing voice samples")
    parser.add_argument("--voice", help="Voice sample name or path")
    parser.add_argument("--language", default="en", help="Language code for TTS")
    parser.add_argument("--device", help="Device override (e.g., cpu, cuda)")

    # TTS backend selection
    parser.add_argument(
        "--tts-backend",
        choices=["xtts", "styletts2"],
        default="xtts",
        help="TTS backend to use (default: xtts)"
    )

    # XTTS-specific options
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="XTTS model name (only for xtts backend)"
    )

    # StyleTTS2-specific options
    parser.add_argument(
        "--style-ref",
        help="Path to style reference audio (for styletts2 backend)"
    )
    parser.add_argument(
        "--style-alpha",
        type=float,
        default=float(os.getenv("STYLETTS2_ALPHA", "0.3")),
        help="StyleTTS2 alpha: 0=more target speaker, 1=more reference style (default: 0.3)"
    )
    parser.add_argument(
        "--style-beta",
        type=float,
        default=float(os.getenv("STYLETTS2_BETA", "0.7")),
        help="StyleTTS2 beta: style strength (default: 0.7)"
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=int(os.getenv("STYLETTS2_DIFFUSION_STEPS", "5")),
        help="StyleTTS2 diffusion steps, more=better quality but slower (default: 5)"
    )


    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model for sentiment analysis (default: llama3.2)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Parse book into chapter text files")
    subparsers.add_parser("speak", help="Generate audio files for each chapter")
    subparsers.add_parser("run", help="Ingest and speak in one step")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "speak":
        speak_command(args)
    elif args.command == "run":
        run_command(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
