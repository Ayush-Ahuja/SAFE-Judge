"""Validate or download the pinned public inputs for the human-review study."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.io import file_hash, write_json


def execute(download=False):
    settings = json.loads((ROOT/"configs/retrieval_sources.json").read_text())
    for source in settings["files"]:
        path = (ROOT/source["path"]).resolve()
        if not path.is_relative_to((ROOT/"cache/research").resolve()):
            raise ValueError("Input path must stay within the research cache")
        if not path.exists():
            if not download:
                raise ValueError(f"Missing {source['path']}; use --download")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+".partial")
            with urllib.request.urlopen(source["url"], timeout=60) as response, temporary.open("wb") as stream:
                while block := response.read(2**20):
                    stream.write(block)
            if file_hash(temporary) != source["sha256"]:
                raise ValueError("Downloaded public input has an unexpected checksum")
            temporary.replace(path)
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"Checksum mismatch: {source['path']}")
    metadata = ROOT/"cache/research/dl19_public_runs/downloads.json"
    expected = settings["retrieval_download_metadata"]
    if metadata.exists() and json.loads(metadata.read_text()) != expected:
        raise ValueError("Retrieval provenance differs")
    write_json(metadata, expected)
    print(f"Validated {len(settings['files'])} pinned public files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    execute(parser.parse_args().download)
