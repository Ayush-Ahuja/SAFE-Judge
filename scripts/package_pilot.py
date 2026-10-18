"""Package Stage 2 code and its frozen pair manifest for Colab; omit models/results."""

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/pilot_code.zip")
    args = parser.parse_args()
    files = [ROOT / name for name in ("configs/default.yaml", "requirements-pilot.txt", "README.md",
             "notebooks/stage2_colab.ipynb", "docs/failure_review.md", "docs/colab.md",
             "data/pilot/dl2019.jsonl", "data/pilot/dl2019.metadata.json")]
    for directory in ("src", "scripts", "tests"):
        files.extend(p for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Packaged {len(files)} source/config files: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
