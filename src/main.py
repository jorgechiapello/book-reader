import argparse
import json
from pathlib import Path
from typing import Dict, List

from formats import Chapter, extract_chapters, slugify
from tts import generate_audio
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
    for chapter in manifest["chapters"]:
        chapter_path = chapters_dir / chapter["file"]
        if not chapter_path.exists():
            raise FileNotFoundError(f"Missing chapter file: {chapter_path}")
        audio_path = chapters_dir / chapter["file"].replace(".txt", ".wav")
        text = chapter_path.read_text(encoding="utf-8")
        generate_audio(
            text=text,
            output_path=audio_path,
            voice_sample_path=voice_sample,
            language=args.language,
            model_name=args.model,
            device=args.device,
        )
        print(f"Wrote audio: {audio_path}")


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
    parser.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2", help="XTTS model name")
    parser.add_argument("--language", default="en", help="Language code for TTS")
    parser.add_argument("--device", help="Device override (e.g., cpu, cuda)")

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
