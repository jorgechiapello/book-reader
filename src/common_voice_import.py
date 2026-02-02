import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def find_locale_root(dataset_root: Path, locale: str) -> Path:
    if (dataset_root / locale / "validated.tsv").exists():
        return dataset_root / locale
    if (dataset_root / "validated.tsv").exists():
        return dataset_root
    raise FileNotFoundError(
        f"Could not find validated.tsv under {dataset_root} (locale={locale})."
    )


def load_speaker_rows(validated_tsv: Path, speaker_id: str) -> List[dict]:
    with validated_tsv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get("client_id") == speaker_id]
    return rows


def copy_clips(
    clips_dir: Path,
    rows: List[dict],
    output_dir: Path,
    max_clips: int,
) -> Tuple[Path, List[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_out_dir = output_dir / "clips"
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for idx, row in enumerate(rows[:max_clips], start=1):
        clip_name = row.get("path")
        if not clip_name:
            continue
        src = clips_dir / clip_name
        if not src.exists():
            continue
        dst = clips_out_dir / f"clip_{idx:03d}{src.suffix}"
        dst.write_bytes(src.read_bytes())
        copied.append(dst)

    if not copied:
        raise FileNotFoundError("No clips copied. Check speaker_id and dataset path.")

    sample_path = output_dir / f"sample{copied[0].suffix}"
    sample_path.write_bytes(copied[0].read_bytes())
    return sample_path, copied


def write_license(output_dir: Path) -> Path:
    license_path = output_dir / "LICENSE.txt"
    license_text = (
        "This voice sample was extracted from Mozilla Common Voice.\n"
        "Dataset: https://commonvoice.mozilla.org/\n"
        "License: https://creativecommons.org/licenses/by/4.0/\n"
        "Make sure you comply with the CC BY 4.0 attribution requirements.\n"
    )
    license_path.write_text(license_text, encoding="utf-8")
    return license_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import one speaker from Mozilla Common Voice into voices/"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to extracted Common Voice dataset (or locale folder)",
    )
    parser.add_argument("--speaker", required=True, help="Common Voice client_id")
    parser.add_argument(
        "--voice-name",
        default="moz_speaker",
        help="Folder name to create under voices/",
    )
    parser.add_argument("--locale", default="en", help="Locale folder (e.g., en, es)")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=5,
        help="Number of clips to copy for reference",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    locale_root = find_locale_root(dataset_root, args.locale)
    validated_tsv = locale_root / "validated.tsv"
    clips_dir = locale_root / "clips"

    rows = load_speaker_rows(validated_tsv, args.speaker)
    if not rows:
        raise ValueError(f"No rows found for speaker '{args.speaker}'.")

    voices_dir = Path("voices") / args.voice_name
    sample_path, copied = copy_clips(clips_dir, rows, voices_dir, args.max_clips)
    license_path = write_license(voices_dir)

    print(f"Sample: {sample_path}")
    print(f"Clips copied: {len(copied)}")
    print(f"License: {license_path}")


if __name__ == "__main__":
    main()
