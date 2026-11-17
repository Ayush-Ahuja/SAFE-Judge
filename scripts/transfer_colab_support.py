"""Small, dependency-free helpers used by the uploadable transfer notebook."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import zipfile


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(root):
    root = Path(root)
    receipt = json.loads((root / "bundle_hashes.json").read_text())
    for relative, expected in receipt.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed bundle file: {relative}")
    return len(receipt)


def restore_run(local, durable):
    """Only restore into an empty working directory; retain newer local batches."""
    local, durable = Path(local), Path(durable)
    local.mkdir(parents=True, exist_ok=True)
    if not (local / "checkpoint.sqlite3").exists() and (durable / "checkpoint.sqlite3").exists():
        for path in durable.iterdir():
            if path.is_file() and path.name != "run.lock":
                shutil.copy2(path, local / path.name)
    lock = local / "run.lock"
    if lock.exists():
        info = json.loads(lock.read_text())
        if info["host"] == socket.gethostname():
            try:
                os.kill(int(info["pid"]), 0)
            except ProcessLookupError:
                lock.unlink()
            else:
                raise RuntimeError(f"A worker may still be running (PID {info['pid']}); stop it before resuming.")
        else:
            lock.unlink()


def archive_results(root, run_dir, destination):
    """Archive inputs and results, excluding downloaded model weights and live DBs.

    The durable directory holds consistent SQLite backups produced by the worker.
    Always archive that directory instead of a live local SQLite connection.
    """
    root, run_dir, destination = Path(root), Path(run_dir), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for parent, prefix in ((root / "data/transfer", "inputs"), (run_dir, "runs")):
            if parent.exists():
                for path in sorted(parent.rglob("*")):
                    if path.is_file() and path != destination and path != temporary and path.name not in ("input_bundle.zip", "run.lock"):
                        archive.write(path, f"{prefix}/{path.relative_to(parent).as_posix()}")
        for name in ("bundle_hashes.json", "configs/transfer_colab.yaml", "docs/qwen_transfer_colab.md"):
            path = root / name
            if path.exists():
                archive.write(path, f"provenance/{name}")
        receipt_path = root / "bundle_hashes.json"
        if receipt_path.exists():
            for name in json.loads(receipt_path.read_text()):
                if name.startswith(("src/", "scripts/", "notebooks/")):
                    archive.write(root / name, f"source/{name}")
    temporary.replace(destination)
    return destination


def safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            name = item.filename
            if ("\\" in name or ":" in name or PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe archive member: {name}")
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {name}")
        archive.extractall(destination)
