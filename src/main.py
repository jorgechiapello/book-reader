import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

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
    # IndexTTS2 uses voices from the TTS server; no local voice file needed
    if args.tts_backend == "indextts2":
        voice_sample = args.voice  # Voice name for API; None = server default
    else:
        voice_sample = resolve_voice_sample(args.voice, Path(args.voices_dir))

    ollama_model = getattr(args, "ollama_model", "llama3.2")

    for chapter in manifest["chapters"]:
        chapter_path = chapters_dir / chapter["file"]
        if not chapter_path.exists():
            raise FileNotFoundError(f"Missing chapter file: {chapter_path}")
        text = chapter_path.read_text(encoding="utf-8")

        print(f"Processing: {chapter['title']}")

        # Use LLM-powered sentiment analysis based on backend
        if args.tts_backend == "indextts2":
            from workflows.indextts2.workflow import run_indextts2_workflow
            run_indextts2_workflow(
                text=text,
                ollama_model=ollama_model,
                voice_sample_path=voice_sample,
                output_dir=chapters_dir,
                chapter_title=chapter["title"],
                chapter_filename=chapter["file"],
            )
        elif args.tts_backend == "styletts2":
            from workflows.styletts2.workflow import run_styletts2_workflow
            run_styletts2_workflow(
                text=text,
                ollama_model=ollama_model,
                voice_sample_path=voice_sample,
                output_dir=chapters_dir,
                chapter_title=chapter["title"],
                chapter_filename=chapter["file"],
            )
        else:
            # Fallback for other backends if they don't use this workflow structure
            print(f"Skipping CrewAI workflow for backend: {args.tts_backend}")


def segments_command(args: argparse.Namespace) -> None:
    """Generate emotional segments (CrewAI) for each chapter. IndexTTS-2 only."""
    if args.tts_backend != "indextts2":
        raise ValueError("segments command requires --tts-backend indextts2")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    book_slug = slugify(args.title or input_path.stem)
    manifest = load_manifest(output_dir, book_slug)
    chapters_dir = build_output_dir(output_dir, book_slug)
    ollama_model = getattr(args, "ollama_model", "llama3.2")

    from workflows.indextts2.workflow import generate_segments

    for chapter in manifest["chapters"]:
        chapter_path = chapters_dir / chapter["file"]
        if not chapter_path.exists():
            raise FileNotFoundError(f"Missing chapter file: {chapter_path}")
        text = chapter_path.read_text(encoding="utf-8")
        print(f"Processing: {chapter['title']}")
        generate_segments(
            text=text,
            ollama_model=ollama_model,
            output_dir=chapters_dir,
            chapter_title=chapter["title"],
            chapter_filename=chapter["file"],
        )


def synthesize_command(args: argparse.Namespace) -> None:
    """Synthesize audio from existing segments. IndexTTS-2 only."""
    if args.tts_backend != "indextts2":
        raise ValueError("synthesize command requires --tts-backend indextts2")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    book_slug = slugify(args.title or input_path.stem)
    manifest = load_manifest(output_dir, book_slug)
    chapters_dir = build_output_dir(output_dir, book_slug)
    voice_sample = args.voice  # Voice name for IndexTTS-2 API

    from workflows.indextts2.workflow import synthesize_from_segments

    for chapter in manifest["chapters"]:
        artifact_base = chapter["file"].replace(".txt", "")
        segments_path = chapters_dir / f"{artifact_base}_segments.json"
        if not segments_path.exists():
            print(f"Skipping {chapter['title']}: no segments file (run 'segments' first)")
            continue
        print(f"Processing: {chapter['title']}")
        synthesize_from_segments(
            voice_sample_path=voice_sample,
            output_dir=chapters_dir,
            chapter_title=chapter["title"],
            chapter_filename=chapter["file"],
        )


def run_command(args: argparse.Namespace) -> None:
    ingest_command(args)
    if args.tts_backend == "indextts2":
        segments_command(args)
        synthesize_command(args)
    else:
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
        choices=["styletts2", "indextts2"],
        default="styletts2",
        help="TTS backend to use (default: styletts2)"
    )

    # Sentiment analysis model




    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model for sentiment analysis (default: llama3.2)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Parse book into chapter text files")
    subparsers.add_parser("speak", help="Generate audio files for each chapter")
    subparsers.add_parser(
        "segments",
        help="Generate emotional segments (CrewAI). IndexTTS-2 only. Requires ingest.",
    )
    subparsers.add_parser(
        "synthesize",
        help="Synthesize audio from segments. IndexTTS-2 only. Requires segments.",
    )
    subparsers.add_parser("run", help="Ingest and speak (or segments+synthesize for IndexTTS-2)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "speak":
        speak_command(args)
    elif args.command == "segments":
        segments_command(args)
    elif args.command == "synthesize":
        synthesize_command(args)
    elif args.command == "run":
        run_command(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
