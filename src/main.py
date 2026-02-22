import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backends import ChapterContext, TTSBackend, get_backend
from formats import Chapter, extract_chapters, slugify
from voices import resolve_voice_sample


# ---------------------------------------------------------------------------
# Book / chapter helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_voice(args: argparse.Namespace) -> Path | None:
    """Resolve --voice name to a local Path, or return None."""
    if not args.voice:
        return None
    try:
        return resolve_voice_sample(args.voice, Path(args.voices_dir))
    except (ValueError, FileNotFoundError):
        return None


def _iter_chapters(args: argparse.Namespace) -> Iterator[ChapterContext]:
    """Yield one ChapterContext per chapter from the manifest."""
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    book_slug = slugify(args.title or input_path.stem)
    manifest = load_manifest(output_dir, book_slug)
    chapters_dir = build_output_dir(output_dir, book_slug)
    voice = _resolve_voice(args)
    ollama_model = getattr(args, "ollama_model", "llama3.2")
    tts_url = getattr(args, "tts_url", "http://localhost:8001")

    for chapter in manifest["chapters"]:
        chapter_path = chapters_dir / chapter["file"]
        if not chapter_path.exists():
            raise FileNotFoundError(f"Missing chapter file: {chapter_path}")
        text = chapter_path.read_text(encoding="utf-8")
        print(f"Processing: {chapter['title']}")
        yield ChapterContext(
            text=text,
            chapter_title=chapter["title"],
            chapter_filename=chapter["file"],
            output_dir=chapters_dir,
            voice=voice,
            ollama_model=ollama_model,
            tts_url=tts_url,
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def ingest_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    book_slug = slugify(args.title or input_path.stem)
    chapters = extract_chapters(input_path)
    chapters_dir = build_output_dir(output_dir, book_slug)
    write_chapter_files(chapters, chapters_dir)
    write_manifest(output_dir, book_slug, chapters)
    print(f"Wrote {len(chapters)} chapters to {chapters_dir}")


def _backend_command(
    args: argparse.Namespace,
    method: str,
) -> None:
    backend: TTSBackend = get_backend(args.tts_backend)
    backend_operation = getattr(backend, method)
    for chapter_context in _iter_chapters(args):
        backend_operation(chapter_context)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local audiobook generator")
    parser.add_argument("--input", required=True, help="Path to TXT/EPUB/PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--title", help="Optional book title override")
    parser.add_argument("--voices-dir", default="voices", help="Directory containing voice samples")
    parser.add_argument("--voice", help="Voice name (e.g., 'Heisenberg')")
    parser.add_argument("--language", default="en", help="Language code for TTS")
    parser.add_argument("--device", help="Device override (e.g., cpu, cuda)")
    parser.add_argument("--tts-url", default="http://localhost:8001", help="IndexTTS-2 server URL")

    parser.add_argument(
        "--tts-backend",
        choices=["styletts2", "indextts2"],
        default="styletts2",
        help="TTS backend to use (default: styletts2)",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "llama3.2"),
        help="Ollama model for sentiment analysis (default: llama3.2)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Parse book into chapter text files")
    subparsers.add_parser(
        "segments",
        help="Generate emotional segments (phase 1). IndexTTS-2 only. Requires ingest.",
    )
    subparsers.add_parser(
        "synthesize",
        help="Synthesize audio from segments (phase 2). IndexTTS-2 only. Requires segments.",
    )
    subparsers.add_parser(
        "run",
        help="Full pipeline: ingest then segments + synthesize (or one-shot for StyleTTS2).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_command(args)
        return

    if args.command == "run":
        ingest_command(args)

    # For 'run', delegate to backend.run(); for others, use the matching method name.
    method = "run" if args.command == "run" else args.command
    _backend_command(args, method)


if __name__ == "__main__":
    main()
